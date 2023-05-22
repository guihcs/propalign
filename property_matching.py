import numpy as np
from nlp import filter_jj, get_core_concept
from transformers import AutoTokenizer
import torch.nn as nn
from om.match import onts, aligns
from om.ont import get_n, tokenize
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL, DCTERMS, SKOS
from rdflib.term import Literal, BNode, URIRef
from sklearn.feature_extraction.text import TfidfVectorizer
from py_stringmatching import SoftTfIdf, JaroWinkler
from utils import metrics
from collections import Counter
from tqdm.auto import tqdm
import math

def get_type_h(e, g, ml=1):
    if type(e) is Literal:
        return [e.datatype]

    tp = g.value(e, DCTERMS.subject)

    if tp is None:
        return [e]

    h = [tp]

    for _ in range(ml):
        if g.value(tp, SKOS.broader) is None:
            break

        tp = g.value(tp, SKOS.broader)
        h.append(tp)

    return h


def most_common_dr_hist(g, ml=1, mh=5):
    props = set()
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    ng = Graph()

    pc = {}
    for prop in props:

        for s, p, o in g.triples((None, prop, None)):
            st = get_type_h(s, g, ml=ml)
            ot = get_type_h(o, g, ml=ml)

            if p not in pc:
                pc[p] = {'domain': Counter(), 'range': Counter()}

            for s in st:
                pc[p]['domain'][s] += 1

            for o in ot:
                pc[p]['range'][o] += 1

    for k in pc:
        c = pc[k]
        d = c['domain'].most_common(mh)
        r = c['range'].most_common(mh)

        jd = '_'.join([x[0].split('/')[-1].split('#')[-1].split(':')[-1] for x in d])
        jr = '_'.join([x[0].split('/')[-1].split('#')[-1].split(':')[-1] for x in r])

        ng.add((k, RDFS.domain, URIRef(jd)))
        ng.add((k, RDFS.range, URIRef(jr)))

    ng.namespace_manager = g.namespace_manager
    return ng


def get_type(e, g):
    if type(e) is Literal:
        return e.datatype

    tp = g.value(e, DCTERMS.subject)

    if tp is None:
        return e

    return tp


def most_common_pair(g):
    props = set()
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    ng = Graph()

    pc = {}
    for prop in props:

        for s, p, o in g.triples((None, prop, None)):
            st = get_type(s, g)
            ot = get_type(o, g)

            if p not in pc:
                pc[p] = Counter()

            pc[p][(st, ot)] += 1

    for k in pc:
        c = pc[k]
        d, r = c.most_common()[0][0]
        ng.add((k, RDFS.domain, d))
        ng.add((k, RDFS.range, r))

    ng.namespace_manager = g.namespace_manager
    return ng


def most_common_dr(g):
    props = set()
    for s, p, o in g.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    ng = Graph()

    pc = {}
    for prop in props:

        for s, p, o in g.triples((None, prop, None)):
            st = get_type(s, g)
            ot = get_type(o, g)

            if p not in pc:
                pc[p] = {'domain': Counter(), 'range': Counter()}

            pc[p]['domain'][st] += 1
            pc[p]['range'][ot] += 1

    for k in pc:
        c = pc[k]
        d = c['domain'].most_common()[0][0]
        r = c['range'].most_common()[0][0]

        ng.add((k, RDFS.domain, d))
        ng.add((k, RDFS.range, r))

    ng.namespace_manager = g.namespace_manager
    return ng


def is_property(e, g):
    return (e, RDFS.domain, None) in g and (e, RDFS.range, None) in g


def get_entity_label_docs(a_entities, g1):
    slist = []
    for e in a_entities:
        slist.append(list(map(str.lower, tokenize(get_n(e, g1)))))

    return slist


def flat_fr_chain(e, g):
    if g.value(e, RDF.rest) == RDF.nil:
        return [g.value(e, RDF.first)]
    else:
        return [g.value(e, RDF.first)] + flat_fr_chain(g.value(e, RDF.rest), g)


def get_cpe(e, g):
    cp = list(set(g.predicates(e)).difference({RDF.type}))
    objs = list(map(lambda x: get_n(x, g), cp + flat_fr_chain(g.value(e, cp[0]), g)))
    return '_'.join(objs), len(objs)


def is_joinable(e, g):
    preds = set(g.predicates(e)).difference({RDF.type})
    return len(preds) == 1 and OWL.unionOf in preds


def flat_restriction(e, g):
    nodes = []
    for s, p, o in g.triples((e, None, None)):
        if p == RDF.type:
            continue
        if type(o) is BNode:
            nodes.extend(flat_restriction(o, g))
        else:
            nodes.extend([p, o])

    return nodes


def is_restriction(e, g):
    return g.value(e, RDF.type) == OWL.Restriction


def join_nodes(nodes, g):
    return '_'.join(list(map(lambda x: get_n(x, g), nodes)))


def get_gen_docs(g1):
    out = []
    for e in set(g1.subjects()):

        if type(e) is BNode:
            if is_joinable(e, g1):
                label, _ = get_cpe(e, g1)
            elif is_restriction(e, g1):
                label = join_nodes(flat_restriction(e, g1), g1)
            else:
                label = get_n(e, g1)

        else:
            label = get_n(e, g1)

        ns = list(map(str.lower, tokenize(label)))

        if is_property(e, g1):

            ds = []
            if (e, RDFS.domain, None) in g1:
                domain = g1.value(e, RDFS.domain)
                if type(domain) is BNode and is_joinable(domain, g1):
                    dn, _ = get_cpe(domain, g1)
                else:
                    dn = get_n(domain, g1)
                ds = list(map(str.lower, tokenize(dn)))

            rs = []
            if (e, RDFS.range, None) in g1:
                rg = g1.value(e, RDFS.range)
                if type(rg) is BNode and is_joinable(rg, g1):
                    rn, _ = get_cpe(rg, g1)
                else:
                    rn = get_n(rg, g1)
                rs = list(map(str.lower, tokenize(rn)))

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
    objs = list(g.objects(e, p))
    objc = len(objs)
    for d in objs:
        if type(d) is BNode:
            if is_joinable(d, g):
                name, oc = get_cpe(d, g)
                objc += oc - 1
            elif is_restriction(d, g):
                name = join_nodes(flat_restriction(d, g), g)
            else:
                name = get_n(d, g)
        else:
            name = get_n(d, g)
        s.extend(map(str.lower, tokenize(name)))

    return s, objc





def build_tf_models(o1, o2):
    a_entities = set(filter(lambda x: is_property(x, o1), o1.subjects()))
    b_entities = set(filter(lambda x: is_property(x, o2), o2.subjects()))

    slist = get_entity_label_docs(a_entities, o1) + get_entity_label_docs(b_entities, o2)
    soft_metric = SoftTfIdf(slist, sim_func=JaroWinkler().get_raw_score, threshold=0.8)

    qlist = get_gen_docs(o1) + get_gen_docs(o2)

    general_metric = TfidfVectorizer()

    general_metric.fit(qlist)

    return soft_metric, general_metric


class PropertyMatcher:

    def __init__(self, class_model, sentence_model):
        super(PropertyMatcher, self).__init__()
        self.class_model = class_model
        self.sentence_model = sentence_model

    def match_property(self, e1, e2, g1, g2, m, ds, sim_weights=None, disable_dr=False):

        exact_label_a = list(map(str.lower, tokenize(get_n(e1, g1))))

        domain_a, dca = get_prop(e1, g1, RDFS.domain)
        range_a, rca = get_prop(e1, g1, RDFS.range)

        if len(range_a) == 1 and exact_label_a[-1] == range_a[0]:
            exact_label_a.pop(-1)

        string_a = get_core_concept(exact_label_a)

        exact_label_b = list(map(str.lower, tokenize(get_n(e2, g2))))

        domain_b, dcb = get_prop(e2, g2, RDFS.domain)
        range_b, rcb = get_prop(e2, g2, RDFS.range)

        if len(range_b) == 1 and exact_label_b[-1] == range_b[0]:
            exact_label_b.pop(-1)

        string_b = get_core_concept(exact_label_b)

        range_a = filter_jj(range_a)
        range_b = filter_jj(range_b)

        if exact_label_a == exact_label_b:
            label_conf_a = 1
            label_conf_b = 1
        elif len(string_a) <= 0 or len(string_b) <= 0:

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

        if disable_dr:
            domain_confidence = 0
            range_confidence = 0
            sim_weights = [1]

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


        if sim_weights:
            conf = []
            if 0 in sim_weights:
                conf.append(domain_confidence)
            if 1 in sim_weights:
                conf.append(label_confidence)

            if 2 in sim_weights:
                conf.append(range_confidence)
        else:
            conf = [label_confidence, domain_confidence, range_confidence]
        return min(conf)

    def match(self, base, ref, th=0.65, process_strategy=None, sim_weights=None, steps=2, disable_dr=False, tr=None):
        correct = 0
        pred = 0
        total = 0
        iterations = 0

        if tr is not None:
            trm = [[0, 0] for _ in tr]


        for r, k1, k2 in tqdm(list(onts(base, ref))):

            print('-' * 100)
            print(k1.split('/')[-1], k2.split('/')[-1])

            print('Loading o1')
            o1 = Graph().parse(k1)
            if process_strategy is not None:
                o1 = process_strategy(o1)

            print('Loading o2')
            o2 = Graph().parse(k2)
            if process_strategy is not None:
                o2 = process_strategy(o2)
            als = set(aligns(r))

            pa = set()
            current_total = 0
            for a1, a2 in als:

                if is_property(a1, o1) and is_property(a2, o2):
                    total += 1
                    current_total += 1
                    pa.add((a1, a2))

                    # d1 = o1.value(a1, RDFS.domain)
                    # d2 = o2.value(a2, RDFS.domain)
                    #
                    # r1 = o1.value(a1, RDFS.range)
                    # r2 = o2.value(a2, RDFS.range)
                    #
                    # print(colored('#', 'blue'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
                    #       get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))
            print(current_total)
            a_entities = set(filter(lambda x: is_property(x, o1), o1.subjects()))
            b_entities = set(filter(lambda x: is_property(x, o2), o2.subjects()))
            p, it = self.match_ontologies(o1, o2, th, sim_weights=sim_weights, steps=steps, disable_dr=disable_dr)
            iterations += it
            oi = it
            current_pred = len(p)
            current_correct = len(pa.intersection(set(p.keys())))
            pred += len(p)
            correct += len(pa.intersection(set(p.keys())))

            if tr is not None:
                for i, t in enumerate(tr):
                    cp = set()
                    for pair, sim in p.items():
                        if sim >= t:
                            cp.add(pair)

                    trm[i][0] += len(pa.intersection(cp))
                    trm[i][1] += len(cp)
                    print(f'ontology iterations: {oi}, {metrics(len(pa.intersection(cp)), len(cp), current_total)}, aligns: {current_total}, po1: {len(a_entities)}, po2: {len(b_entities)}')

            # for a1, a2 in pa.intersection(p):
            #     print(colored('✓', 'green'), get_n(a1, o1), get_n(a2, o2))
            #
            # for a1, a2 in p.difference(pa):
            #     d1 = o1.value(a1, RDFS.domain)
            #     d2 = o2.value(a2, RDFS.domain)
            #
            #     r1 = o1.value(a1, RDFS.range)
            #     r2 = o2.value(a2, RDFS.range)
            #     print(colored('X', 'red'), get_n(d1, o1), get_n(a1, o1), get_n(r1, o1), colored('<>', 'green'),
            #           get_n(d2, o2), get_n(a2, o2), get_n(r2, o2))

            # print(
            #     f'ontology iterations: {oi}, {metrics(current_correct, current_pred, current_total)}, aligns: {current_total}, po1: {len(a_entities)}, po2: {len(b_entities)}')
        print(f'iterations: {iterations}, {metrics(correct, pred, total)}')
        if tr is not None:
            res = []
            for q, w in trm:
                res.append(metrics(q, w, total))

            return res

        return metrics(correct, pred, total)

    def match_ontologies(self, o1, o2, th, sim_weights=None, steps=2, disable_dr=False):

        soft_metric, general_metric = build_tf_models(o1, o2)
        p = {}

        ds = {}

        pm = {}

        iterations = 0
        for step in range(steps):
            for e1 in set(o1.subjects()):
                if not is_property(e1, o1):
                    continue
                for e2 in set(o2.subjects()):
                    if not is_property(e2, o2):
                        continue

                    sim = self.match_property(e1, e2, o1, o2, (soft_metric, general_metric), ds,
                                              sim_weights=sim_weights, disable_dr=disable_dr)

                    iterations += 1
                    if sim <= th:
                        continue

                    if e1 in pm:
                        if pm[e1][1] >= sim:
                            continue
                        elif pm[e1][1] < sim:
                            p.pop((e1, pm[e1][0]))
                            pm.pop(pm[e1][0])
                            pm.pop(e1)

                    if e2 in pm:
                        if pm[e2][1] >= sim:
                            continue
                        elif pm[e2][1] < sim:
                            p.pop((pm[e2][0], e2))
                            pm.pop(pm[e2][0])
                            pm.pop(e2)

                    d1 = o1.value(e1, RDFS.domain)
                    d2 = o2.value(e2, RDFS.domain)
                    ds[(d1, d2)] = 0.66
                    p[(e1, e2)] = sim
                    pm[e1] = (e2, sim)
                    pm[e2] = (e1, sim)
                    if (e1, OWL.inverseOf, None) in o1 and (e2, OWL.inverseOf, None) in o2:
                        d1 = o1.value(o1.value(e1, OWL.inverseOf), RDFS.domain)
                        d2 = o2.value(o2.value(e2, OWL.inverseOf), RDFS.domain)

                        ds[(d1, d2)] = 0.66
                        iv1, iv2 = o1.value(e1, OWL.inverseOf), o2.value(e2, OWL.inverseOf)
                        p[(iv1, iv2)] = sim
                        pm[iv1] = (iv2, sim)
                        pm[iv2] = (iv1, sim)


        return p, iterations

