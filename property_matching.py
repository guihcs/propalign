import numpy as np
from nlp import filter_jj, get_core_concept

import torch
import torch.nn as nn
from om.match import onts, aligns
from om.ont import get_n, tokenize
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL
from rdflib.term import Literal, BNode, URIRef
from sklearn.feature_extraction.text import TfidfVectorizer
from py_stringmatching import SoftTfIdf, JaroWinkler
from termcolor import colored
from utils import metrics


def is_property(e, g):
    types = list(g.objects(e, RDF.type))
    tn = map(lambda x: get_n(x, g), types)
    have_prop = map(lambda x: 'property' in x.lower(), tn)

    return any(have_prop)


def get_docs(a_entities, g1):
    out = []
    slist = []
    for e in a_entities:
        ns = list(map(str.lower, tokenize(get_n(e, g1))))

        ds = []
        if (e, RDFS.domain, None) in g1:
            ds = list(map(str.lower, tokenize(get_n(g1.value(e, RDFS.domain), g1))))

        rs = []
        if (e, RDFS.range, None) in g1:
            rs = list(map(str.lower, tokenize(get_n(g1.value(e, RDFS.range), g1))))

        out.append(' '.join(ns + rs + ds))
        slist.append(ns)

    return out, slist


def get_gen_docs(g1):
    out = []
    for e in set(g1.subjects()):
        ns = list(map(str.lower, tokenize(get_n(e, g1))))

        if is_property(e, g1):

            ds = []
            if (e, RDFS.domain, None) in g1:
                ds = list(map(str.lower, tokenize(get_n(g1.value(e, RDFS.domain), g1))))

            rs = []
            if (e, RDFS.range, None) in g1:
                rs = list(map(str.lower, tokenize(get_n(g1.value(e, RDFS.range), g1))))

            out.append(' '.join(ns + rs + ds))
        else:
            out.append(' '.join(ns))

    return out


def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_document_similarity(domain_a, domain_b, m):
    if len(domain_a) <= 0 or len(domain_b) <= 0:

        domain_conf_a = 0
        domain_conf_b = 0
    else:
        v = m[1].transform([' '.join(domain_a), ' '.join(domain_b)])
        v = v.toarray()
        domain_conf_a = cosine_similarity(v[0], v[1])
        v = m[1].transform([' '.join(domain_b), ' '.join(domain_a)])
        v = v.toarray()
        domain_conf_b = cosine_similarity(v[0], v[1])

    return domain_conf_a, domain_conf_b


def get_prop(e, g, p):
    s = []
    for d in g.objects(e, p):
        s.extend(map(str.lower, tokenize(get_n(d, g))))

    return s


class PropertyMatcher:

    def __init__(self, class_model, sentence_model):
        super(PropertyMatcher, self).__init__()
        self.class_model = class_model
        self.sentence_model = sentence_model

    def match_property(self, e1, e2, g1, g2, m, ds):

        exact_label_a = list(map(str.lower, tokenize(get_n(e1, g1))))

        domain_a = get_prop(e1, g1, RDFS.domain)
        range_a = get_prop(e1, g1, RDFS.range)

        if len(range_a) == 1 and exact_label_a[-1] == range_a[0]:
            exact_label_a.pop(-1)

        string_a = get_core_concept(exact_label_a)

        exact_label_b = list(map(str.lower, tokenize(get_n(e2, g2))))

        domain_b = get_prop(e2, g2, RDFS.domain)
        range_b = get_prop(e2, g2, RDFS.range)

        if len(range_b) == 1 and exact_label_b[-1] == range_b[0]:
            exact_label_b.pop(-1)

        string_b = get_core_concept(exact_label_b)

        range_a = filter_jj(range_a)
        range_b = filter_jj(range_b)

        if len(string_a) <= 0 or len(string_b) <= 0:

            label_conf_a = 0
            label_conf_b = 0
        else:
            label_conf_a = m[0].get_raw_score(string_a, string_b)
            label_conf_b = m[0].get_raw_score(string_b, string_a)

        domain_conf_a, domain_conf_b = get_document_similarity(domain_a, domain_b, m)
        range_conf_a, range_conf_b = get_document_similarity(range_a, range_b, m)

        label_confidence = (label_conf_a + label_conf_b) / 2
        domain_confidence = (domain_conf_a + domain_conf_b) / 2
        range_confidence = (range_conf_a + range_conf_b) / 2

        if domain_confidence == 0 and len(exact_label_a) > 0 and len(exact_label_b) > 0:
            if len(domain_a) == 1 and len(domain_b) == 1:
                domain_confidence = self.class_model.sim(domain_a[0], domain_b[0])

        dsp = (g1.value(e1, RDFS.domain), g2.value(e2, RDFS.domain))
        if dsp in ds:
            domain_confidence += ds[dsp]

        rsp = (g1.value(e1, RDFS.range), g2.value(e2, RDFS.range))
        if rsp in ds:
            range_confidence += ds[rsp]

        if domain_confidence > 0.95 and range_confidence > 0.95 and label_confidence < 0.1:
            if len(string_a) <= 1 and len(string_b) <= 1:
                sr = [' '.join(domain_a + list(map(str.lower, tokenize(get_n(e1, g1)))) + range_a)]
                tg = [' '.join(domain_b + list(map(str.lower, tokenize(get_n(e2, g2)))) + range_b)]
                e1 = self.sentence_model.encode(sr, convert_to_tensor=True)
                e2 = self.sentence_model.encode(tg, convert_to_tensor=True)
                sim = nn.functional.cosine_similarity(e1, e2).item()
                if sim < 0.8:
                    sim = 0
                label_confidence = sim

        return min([label_confidence, domain_confidence, range_confidence])

    def match(self, o1, o2, th=0.65):
        correct = 0
        pred = 0
        total = 0
        iterations = 0
        for r, k1, k2 in onts(o1, o2):

            print('-' * 100)
            print(k1.split('/')[-1], k2.split('/')[-1])

            o1 = Graph().parse(k1)
            o2 = Graph().parse(k2)

            als = set(aligns(r))

            pa = set()

            for a1, a2 in als:

                if is_property(a1, o1) and is_property(a2, o2):
                    total += 1
                    pa.add((a1, a2))

                    d1 = o1.value(a1, RDFS.domain)
                    d2 = o2.value(a2, RDFS.domain)

                    r1 = o1.value(a1, RDFS.range)
                    r2 = o2.value(a2, RDFS.range)

                    print(colored('#', 'blue'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
                          get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))

            a_entities = set(filter(lambda x: is_property(x, o1), o1.subjects()))
            b_entities = set(filter(lambda x: is_property(x, o2), o2.subjects()))

            l1, s1 = get_docs(a_entities, o1)
            l2, s2 = get_docs(b_entities, o2)

            qlist = l1 + l2
            slist = s1 + s2

            prop_metric = TfidfVectorizer()

            prop_metric.fit(qlist)

            qlist = get_gen_docs(o1) + get_gen_docs(o2)

            general_metric = TfidfVectorizer()

            general_metric.fit(qlist)
            soft_metric = SoftTfIdf(slist, sim_func=JaroWinkler().get_raw_score, threshold=0.8)
            p = set()

            ds = {}

            pm = {}

            oi = 0

            for step in range(2):

                for e1 in set(o1.subjects()):
                    if not is_property(e1, o1):
                        continue
                    for e2 in set(o2.subjects()):
                        if not is_property(e2, o2):
                            continue

                        sim = self.match_property(e1, e2, o1, o2, (soft_metric, general_metric), ds)
                        iterations += 1
                        oi += 1
                        if sim > th:
                            if e1 in pm:
                                if pm[e1][1] >= sim:
                                    continue
                                elif pm[e1][1] < sim:
                                    p.discard((e1, pm[e1][0]))
                                    pm.pop(pm[e1][0])
                                    pm.pop(e1)

                            if e2 in pm:
                                if pm[e2][1] >= sim:
                                    continue
                                elif pm[e2][1] < sim:
                                    p.discard((pm[e2][0], e2))
                                    pm.pop(pm[e2][0])
                                    pm.pop(e2)

                            d1 = o1.value(e1, RDFS.domain)
                            d2 = o2.value(e2, RDFS.domain)
                            ds[(d1, d2)] = 0.66
                            p.add((e1, e2))
                            pm[e1] = (e2, sim)
                            pm[e2] = (e1, sim)
                            if (e1, OWL.inverseOf, None) in o1 and (e2, OWL.inverseOf, None) in o2:
                                d1 = o1.value(o1.value(e1, OWL.inverseOf), RDFS.domain)
                                d2 = o2.value(o2.value(e2, OWL.inverseOf), RDFS.domain)

                                ds[(d1, d2)] = 0.66
                                iv1, iv2 = o1.value(e1, OWL.inverseOf), o2.value(e2, OWL.inverseOf)
                                p.add((iv1, iv2))
                                pm[iv1] = (iv2, sim)
                                pm[iv2] = (iv1, sim)

            pred += len(p)
            correct += len(pa.intersection(p))

            for a1, a2 in pa.intersection(p):
                print(colored('✓', 'green'), get_n(a1, o1), get_n(a2, o2))

            for a1, a2 in p.difference(pa):
                d1 = o1.value(a1, RDFS.domain)
                d2 = o2.value(a2, RDFS.domain)

                r1 = o1.value(a1, RDFS.range)
                r2 = o2.value(a2, RDFS.range)
                print(colored('X', 'red'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
                      get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))

            print('ontology iterations:', oi)
        print(f'iterations: {iterations}, {metrics(correct, pred, total)}')
        return metrics(correct, pred, total)



    def match_with_model_filter(self, o1, o2, tokenizer, model):
        correct = 0
        pred = 0
        total = 0
        iterations = 0
        for r, k1, k2 in onts(o1, o2):  # [(rp, o1p, o2p)]:

            print('-' * 100)
            print(k1.split('/')[-1], k2.split('/')[-1])

            o1 = Graph().parse(k1)
            o2 = Graph().parse(k2)

            als = set(aligns(r))

            pa = set()

            for a1, a2 in als:

                if is_property(a1, o1) and is_property(a2, o2):
                    total += 1
                    pa.add((a1, a2))

                    d1 = o1.value(a1, RDFS.domain)
                    d2 = o2.value(a2, RDFS.domain)

                    r1 = o1.value(a1, RDFS.range)
                    r2 = o2.value(a2, RDFS.range)

                    print(colored('#', 'blue'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
                          get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))

            aEntities = set(filter(lambda x: is_property(x, o1), o1.subjects()))
            bEntities = set(filter(lambda x: is_property(x, o2), o2.subjects()))

            l1, s1 = get_docs(aEntities, o1)
            l2, s2 = get_docs(bEntities, o2)

            qlist = l1 + l2
            slist = s1 + s2

            propMetric = TfidfVectorizer()

            propMetric.fit(qlist)

            qlist = get_gen_docs(o1) + get_gen_docs(o2)

            generalMetric = TfidfVectorizer()

            generalMetric.fit(qlist)
            softMetric = SoftTfIdf(slist, sim_func=JaroWinkler().get_raw_score, threshold=0.8)
            p = set()

            ds = {}

            pm = {}

            p1 = []

            for e1 in set(o1.subjects()):
                if not is_property(e1, o1) or (e1, RDFS.domain, None) not in o1 or type(
                        o1.value(e1, RDFS.domain)) is BNode:
                    continue

                p1.append(e1)

            p2 = []

            for e2 in set(o2.subjects()):
                if not is_property(e2, o2) or (e2, RDFS.domain, None) not in o2 or type(
                        o2.value(e2, RDFS.domain)) is BNode:
                    continue

                p2.append(e2)

            p1d = [' '.join(map(str.lower, tokenize(get_n(o1.value(x, RDFS.domain), o1)))) for x in p1]
            p2d = [' '.join(map(str.lower, tokenize(get_n(o2.value(x, RDFS.domain), o2)))) for x in p2]

            tk = tokenizer(p1d, return_tensors='pt', padding=True)

            idx = tk['input_ids']
            atn = tk['attention_mask']

            with torch.no_grad():
                out1 = model(idx.cuda(0), atn.cuda(0)).exp().cpu()

            tk = tokenizer(p2d, return_tensors='pt', padding=True)

            idx = tk['input_ids']
            atn = tk['attention_mask']

            with torch.no_grad():
                out2 = model(idx.cuda(0), atn.cuda(0)).exp().cpu()

            cl1 = out1.argmax(dim=1)
            cl2 = out2.argmax(dim=1)
            sim = cl1.unsqueeze(1) == cl2.unsqueeze(0)



            nz = list(map(lambda x: (p1[x[0].item()], p2[x[1].item()]), sim.nonzero()))
            oi = 0
            for step in range(2):

                for e1, e2 in nz:


                    sim = self.match_property(e1, e2, o1, o2, (softMetric, generalMetric), ds)
                    iterations += 1
                    oi += 1
                    if sim > 0.65:

                        if e1 in pm:
                            if pm[e1][1] >= sim:
                                continue
                            elif pm[e1][1] < sim:
                                p.discard((e1, pm[e1][0]))
                                pm.pop(pm[e1][0])
                                pm.pop(e1)

                        if e2 in pm:
                            if pm[e2][1] >= sim:
                                continue
                            elif pm[e2][1] < sim:
                                p.discard((pm[e2][0], e2))
                                pm.pop(pm[e2][0])
                                pm.pop(e2)

                        d1 = o1.value(e1, RDFS.domain)
                        d2 = o2.value(e2, RDFS.domain)

                        ds[(d1, d2)] = 0.66
                        p.add((e1, e2))
                        pm[e1] = (e2, sim)
                        pm[e2] = (e1, sim)

                        if (e1, OWL.inverseOf, None) in o1 and (e2, OWL.inverseOf, None) in o2:
                            d1 = o1.value(o1.value(e1, OWL.inverseOf), RDFS.domain)
                            d2 = o2.value(o2.value(e2, OWL.inverseOf), RDFS.domain)

                            ds[(d1, d2)] = 0.66
                            iv1, iv2 = o1.value(e1, OWL.inverseOf), o2.value(e2, OWL.inverseOf)
                            p.add((iv1, iv2))
                            pm[iv1] = (iv2, sim)
                            pm[iv2] = (iv1, sim)

            pred += len(p)
            correct += len(pa.intersection(p))

            for a1, a2 in pa.intersection(p):
                print(colored('✓', 'green'), get_n(a1, o1), get_n(a2, o2))

            for a1, a2 in p.difference(pa):
                d1 = o1.value(a1, RDFS.domain)
                d2 = o2.value(a2, RDFS.domain)

                r1 = o1.value(a1, RDFS.range)
                r2 = o2.value(a2, RDFS.range)
                print(colored('X', 'red'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
                      get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))

            print('ontology iterations:', oi)
        print(f'iterations: {iterations}, {metrics(correct, pred, total)}')
