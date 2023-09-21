import numpy as np
from nlp import filter_adjectives, get_core_concept
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


def get_type_hierarchy(entity, graph, max_depth=1):
    """
    Get the type hierarchy of an entity in a graph up to a certain depth.
    :param entity: RDF entity
    :param graph: RDFLib graph
    :param max_depth: maximum depth of the hierarchy.
    :return: list of types
    """
    if type(entity) is Literal:
        return [entity.datatype]

    entity_parent = graph.value(entity, DCTERMS.subject)

    if entity_parent is None:
        return [entity]

    hierarchy = [entity_parent]

    for _ in range(max_depth):
        if graph.value(entity_parent, SKOS.broader) is None:
            break

        entity_parent = graph.value(entity_parent, SKOS.broader)
        hierarchy.append(entity_parent)

    return hierarchy


def most_common_domain_range_pair(graph, max_depth=1, most_common_count=5):
    """
    Get the most common domain and range for each property in a graph.
    :param graph:
    :param max_depth:
    :param most_common_count:
    :return:
    """
    properties = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        properties.add(s)

    new_graph = Graph()

    pair_counter = {}
    for prop in properties:

        for s, p, o in graph.triples((None, prop, None)):
            subject_type = get_type_hierarchy(s, graph, max_depth=max_depth)
            object_type = get_type_hierarchy(o, graph, max_depth=max_depth)

            if p not in pair_counter:
                pair_counter[p] = {'domain': Counter(), 'range': Counter()}

            for s in subject_type:
                pair_counter[p]['domain'][s] += 1

            for o in object_type:
                pair_counter[p]['range'][o] += 1

    for pair in pair_counter:
        count = pair_counter[pair]
        domain = count['domain'].most_common(most_common_count)
        rang = count['range'].most_common(most_common_count)

        joined_domain = join_entities(domain)
        joined_range = join_entities(rang)

        new_graph.add((pair, RDFS.domain, URIRef(joined_domain)))
        new_graph.add((pair, RDFS.range, URIRef(joined_range)))

    new_graph.namespace_manager = graph.namespace_manager
    return new_graph


def join_entities(entities):
    return '_'.join([x[0].split('/')[-1].split('#')[-1].split(':')[-1] for x in entities])


def get_type(entity, graph):
    """
    Get the type of entity in a graph.
    :param entity:
    :param graph:
    :return:
    """
    if type(entity) is Literal:
        return entity.datatype

    parent = graph.value(entity, DCTERMS.subject)

    if parent is None:
        return entity

    return parent


def most_common_pair(graph):
    """
    Get the most common domain and range for each property in a graph.
    :param graph:
    :return:
    """
    props = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    new_graph = Graph()

    pair_counter = {}
    for prop in props:

        for s, p, o in graph.triples((None, prop, None)):
            subject_type = get_type(s, graph)
            object_type = get_type(o, graph)

            if p not in pair_counter:
                pair_counter[p] = Counter()

            pair_counter[p][(subject_type, object_type)] += 1

    for pair in pair_counter:
        count = pair_counter[pair]
        domain, rng = count.most_common()[0][0]
        new_graph.add((pair, RDFS.domain, domain))
        new_graph.add((pair, RDFS.range, rng))

    new_graph.namespace_manager = graph.namespace_manager
    return new_graph


def most_common_dr(graph):
    """
    Get the most common domain and range for each property in a graph.
    :param graph:
    :return:
    """
    props = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    new_graph = Graph()

    pair_counter = {}
    for prop in props:

        for s, p, o in graph.triples((None, prop, None)):
            subject_type = get_type(s, graph)
            object_type = get_type(o, graph)

            if p not in pair_counter:
                pair_counter[p] = {'domain': Counter(), 'range': Counter()}

            pair_counter[p]['domain'][subject_type] += 1
            pair_counter[p]['range'][object_type] += 1

    for pair in pair_counter:
        count = pair_counter[pair]
        domain = count['domain'].most_common()[0][0]
        rng = count['range'].most_common()[0][0]

        new_graph.add((pair, RDFS.domain, domain))
        new_graph.add((pair, RDFS.range, rng))

    new_graph.namespace_manager = graph.namespace_manager
    return new_graph


def is_property(entity, graph):
    """
    Check if an entity is a property in a graph.
    :param entity:
    :param graph:
    :return:
    """
    return (entity, RDFS.domain, None) in graph and (entity, RDFS.range, None) in graph


def get_entity_label_docs(entities, graph):
    """
    Process the labels of a set of entities in a graph. The labels are tokenized and lowercased.
    :param entities:
    :param graph:
    :return:
    """
    result = []
    for entity in entities:
        result.append(list(map(str.lower, tokenize(get_n(entity, graph)))))

    return result


def flat_rdf_list_chain(entity, graph):
    """
    Convert and RDF first rest tree to a list.
    :param entity:
    :param graph:
    :return:
    """
    if graph.value(entity, RDF.rest) == RDF.nil:
        return [graph.value(entity, RDF.first)]
    else:
        return [graph.value(entity, RDF.first)] + flat_rdf_list_chain(graph.value(entity, RDF.rest), graph)


def get_concatenated_predicate_entities(entity, graph):
    """
    Get the concatenation of the predicates of an entity in a graph.
    :param entity:
    :param graph:
    :return:
    """
    not_type_predicates = list(set(graph.predicates(entity)).difference({RDF.type}))
    tmp = not_type_predicates + flat_rdf_list_chain(graph.value(entity, not_type_predicates[0]), graph)
    objs = list(map(lambda x: get_n(x, graph), tmp))
    return '_'.join(objs), len(objs)


def is_joinable(entity, graph):
    """
    Check if an entity is joinable in a graph. An entity is joinable if it has only one predicate and that predicate is
    a unionOf predicate.
    :param entity:
    :param graph:
    :return:
    """
    preds = set(graph.predicates(entity)).difference({RDF.type})
    return len(preds) == 1 and OWL.unionOf in preds


def flat_restriction(entity, graph):
    """
    Flatten a restriction in a graph.
    :param entity:
    :param graph:
    :return:
    """
    nodes = []
    for s, p, o in graph.triples((entity, None, None)):
        if p == RDF.type:
            continue
        if type(o) is BNode:
            nodes.extend(flat_restriction(o, graph))
        else:
            nodes.extend([p, o])

    return nodes


def is_restriction(entity, graph):
    """
    Check if an entity is a restriction in a graph.
    :param entity:
    :param graph:
    :return:
    """
    return graph.value(entity, RDF.type) == OWL.Restriction


def join_nodes(nodes, graph):
    """
    Join a list of nodes in a graph.
    :param nodes:
    :param graph:
    :return:
    """
    return '_'.join(list(map(lambda x: get_n(x, graph), nodes)))


def get_gen_docs(graph):
    """
    Get the general documents of a graph. The general documents are the labels of the entities in the graph.
    :param graph: 
    :return: 
    """
    out = []
    for subject in set(graph.subjects()):

        if type(subject) is BNode:
            if is_joinable(subject, graph):
                label, _ = get_concatenated_predicate_entities(subject, graph)
            elif is_restriction(subject, graph):
                label = join_nodes(flat_restriction(subject, graph), graph)
            else:
                label = get_n(subject, graph)

        else:
            label = get_n(subject, graph)

        tokens = list(map(str.lower, tokenize(label)))

        if is_property(subject, graph):

            domain_sentence = get_predicate_sentence(subject, RDFS.domain, graph)

            renge_sentence = get_predicate_sentence(subject, RDFS.range, graph)

            out.append(' '.join(tokens + renge_sentence + domain_sentence))
        else:
            out.append(' '.join(tokens))

    return out


def get_predicate_sentence(subject, predicate, graph):
    """
    Get the predicate sentence of a subject in a graph.
    :param subject: 
    :param predicate: 
    :param graph: 
    :return: 
    """
    domain_sentence = []
    if (subject, predicate, None) in graph:
        domain = graph.value(subject, predicate)
        if type(domain) is BNode and is_joinable(domain, graph):
            dn, _ = get_concatenated_predicate_entities(domain, graph)
        else:
            dn = get_n(domain, graph)
        domain_sentence = list(map(str.lower, tokenize(dn)))

    return domain_sentence


def cosine_similarity(vector1, vector2):
    """
    Compute the cosine similarity between two vectors.
    :param vector1: 
    :param vector2: 
    :return: 
    """
    if np.linalg.norm(vector1) * np.linalg.norm(vector2) == 0:
        return 0
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def get_document_similarity(domain_a, domain_b, models):
    """
    Compute the document similarity between two domains.
    :param domain_a: 
    :param domain_b: 
    :param models: 
    :return: 
    """
    if len(domain_a) <= 0 or len(domain_b) <= 0:

        domain_conf_a = 0
        domain_conf_b = 0
    else:
        vector = models[1].transform([' '.join(domain_a), ' '.join(domain_b)])
        vector = vector.toarray()
        domain_conf_a = cosine_similarity(vector[0], vector[1])
        vector = models[1].transform([' '.join(domain_b), ' '.join(domain_a)])
        vector = vector.toarray()
        domain_conf_b = cosine_similarity(vector[0], vector[1])

    return domain_conf_a, domain_conf_b


def get_property_sentence(entity, graph, predicate):
    """
    Get the property sentence of an entity in a graph.
    :param entity: 
    :param graph: 
    :param predicate: 
    :return: 
    """
    sentence = []
    objects = list(graph.objects(entity, predicate))
    object_count = len(objects)
    for obj in objects:
        if type(obj) is BNode:
            if is_joinable(obj, graph):
                name, oc = get_concatenated_predicate_entities(obj, graph)
                object_count += oc - 1
            elif is_restriction(obj, graph):
                name = join_nodes(flat_restriction(obj, graph), graph)
            else:
                name = get_n(obj, graph)
        else:
            name = get_n(obj, graph)
        sentence.extend(map(str.lower, tokenize(name)))

    return sentence, object_count


def build_tf_models(ontology1, ontology2):
    """
    Build the tf models for the soft tf-idf and the general tf-idf.
    :param ontology1: 
    :param ontology2: 
    :return: 
    """
    properties1 = set(filter(lambda x: is_property(x, ontology1), ontology1.subjects()))
    properties2 = set(filter(lambda x: is_property(x, ontology2), ontology2.subjects()))

    sentences_list = get_entity_label_docs(properties1, ontology1) + get_entity_label_docs(properties2, ontology2)
    soft_metric = SoftTfIdf(sentences_list, sim_func=JaroWinkler().get_raw_score, threshold=0.8)

    document_list = get_gen_docs(ontology1) + get_gen_docs(ontology2)

    general_metric = TfidfVectorizer()

    general_metric.fit(document_list)

    return soft_metric, general_metric


class PropertyMatcher:

    def __init__(self, class_model, sentence_model):
        super(PropertyMatcher, self).__init__()
        self.class_model = class_model
        self.sentence_model = sentence_model

    def match_property(self, entity1, entity2, graph1, graph2, models, confidence_map, similarity_weights=None,
                       disable_domain_range_similarity=False):
        """
        Match two properties in two graphs. The matching is done by comparing the labels, domains and ranges of the
        properties.
        :param entity1: 
        :param entity2: 
        :param graph1: 
        :param graph2: 
        :param models: 
        :param confidence_map: 
        :param similarity_weights: 
        :param disable_domain_range_similarity: 
        :return: Confidence of the match.
        """
        exact_label_a = list(map(str.lower, tokenize(get_n(entity1, graph1))))

        domain_a, dca = get_property_sentence(entity1, graph1, RDFS.domain)
        range_a, rca = get_property_sentence(entity1, graph1, RDFS.range)

        if len(range_a) == 1 and exact_label_a[-1] == range_a[0]:
            exact_label_a.pop(-1)

        string_a = get_core_concept(exact_label_a)

        exact_label_b = list(map(str.lower, tokenize(get_n(entity2, graph2))))

        domain_b, dcb = get_property_sentence(entity2, graph2, RDFS.domain)
        range_b, rcb = get_property_sentence(entity2, graph2, RDFS.range)

        if len(range_b) == 1 and exact_label_b[-1] == range_b[0]:
            exact_label_b.pop(-1)

        string_b = get_core_concept(exact_label_b)

        range_a = filter_adjectives(range_a)
        range_b = filter_adjectives(range_b)

        if exact_label_a == exact_label_b:
            label_conf_a = 1
            label_conf_b = 1
        elif len(string_a) <= 0 or len(string_b) <= 0:

            label_conf_a = 0
            label_conf_b = 0
        else:
            label_conf_a = models[0].get_raw_score(string_a, string_b)
            label_conf_b = models[0].get_raw_score(string_b, string_a)

        domain_conf_a, domain_conf_b = get_document_similarity(domain_a, domain_b, models)
        range_conf_a, range_conf_b = get_document_similarity(range_a, range_b, models)

        label_confidence = (label_conf_a + label_conf_b) / 2
        domain_confidence = (domain_conf_a + domain_conf_b) / 2
        range_confidence = (range_conf_a + range_conf_b) / 2

        if domain_confidence == 0 and len(exact_label_a) > 0 and len(exact_label_b) > 0:
            if len(domain_a) == 1 and len(domain_b) == 1:
                domain_confidence = self.class_model.sim(domain_a[0], domain_b[0])

        dsp = (graph1.value(entity1, RDFS.domain), graph2.value(entity2, RDFS.domain))
        if dsp in confidence_map:
            domain_confidence += confidence_map[dsp]

        rsp = (graph1.value(entity1, RDFS.range), graph2.value(entity2, RDFS.range))
        if rsp in confidence_map:
            range_confidence += confidence_map[rsp]

        if disable_domain_range_similarity:
            domain_confidence = 0
            range_confidence = 0
            similarity_weights = [1]

        if domain_confidence > 0.95 and range_confidence > 0.95 and label_confidence < 0.1:
            if len(string_a) <= 1 and len(string_b) <= 1:
                sr = [' '.join(domain_a + list(map(str.lower, tokenize(get_n(entity1, graph1)))) + range_a)]
                tg = [' '.join(domain_b + list(map(str.lower, tokenize(get_n(entity2, graph2)))) + range_b)]
                entity1 = self.sentence_model.encode(sr, convert_to_tensor=True)
                entity2 = self.sentence_model.encode(tg, convert_to_tensor=True)
                sim = nn.functional.cosine_similarity(entity1, entity2).item()
                if sim < 0.8:
                    sim = 0
                label_confidence = sim

        if similarity_weights:
            conf = []
            if 0 in similarity_weights:
                conf.append(domain_confidence)
            if 1 in similarity_weights:
                conf.append(label_confidence)

            if 2 in similarity_weights:
                conf.append(range_confidence)
        else:
            conf = [label_confidence, domain_confidence, range_confidence]
        return min(conf)

    def match(self, base, ref, threshold=0.65, process_strategy=None, sim_weights=None, steps=2, disable_dr=False, start_metrics=None):
        """
        Match ontologies in a folder according to a reference alignment.
        :param base: Path to the ontologies.
        :param ref: Path to the reference alignments.
        :param threshold: 
        :param process_strategy: 
        :param sim_weights: 
        :param steps: 
        :param disable_dr: 
        :param start_metrics:
        :return: 
        """
        correct = 0
        pred = 0
        total = 0
        iterations = 0

        if start_metrics is not None:
            total_metrics = [[0, 0] for _ in start_metrics]

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


            print(current_total)
            a_entities = set(filter(lambda x: is_property(x, o1), o1.subjects()))
            b_entities = set(filter(lambda x: is_property(x, o2), o2.subjects()))
            p, it = self.match_ontologies(o1, o2, threshold, sim_weights=sim_weights, steps=steps, disable_dr=disable_dr)
            iterations += it
            oi = it
            current_pred = len(p)
            current_correct = len(pa.intersection(set(p.keys())))
            pred += len(p)
            correct += len(pa.intersection(set(p.keys())))

            if start_metrics is not None:
                for i, t in enumerate(start_metrics):
                    cp = set()
                    for pair, sim in p.items():
                        if sim >= t:
                            cp.add(pair)

                    total_metrics[i][0] += len(pa.intersection(cp))
                    total_metrics[i][1] += len(cp)
                    print(
                        f'ontology iterations: {oi}, {metrics(len(pa.intersection(cp)), len(cp), current_total)}, aligns: {current_total}, po1: {len(a_entities)}, po2: {len(b_entities)}')


            print(
                f'ontology iterations: {oi}, {metrics(current_correct, current_pred, current_total)}, aligns: {current_total}, po1: {len(a_entities)}, po2: {len(b_entities)}')

        print(f'iterations: {iterations}, {metrics(correct, pred, total)}')
        if start_metrics is not None:
            res = []
            for q, w in total_metrics:
                res.append(metrics(q, w, total))

            return res

        return metrics(correct, pred, total)

    def match_ontologies(self, o1, o2, threshold, sim_weights=None, steps=2, disable_dr=False):
        """
        Match two ontologies.
        :param o1:
        :param o2:
        :param threshold:
        :param sim_weights:
        :param steps:
        :param disable_dr:
        :return:
        """
        soft_metric, general_metric = build_tf_models(o1, o2)
        final_alignment = {}

        confidence_map = {}

        property_map = {}

        iterations = 0
        for step in range(steps):
            for e1 in set(o1.subjects()):
                if not is_property(e1, o1):
                    continue
                for e2 in set(o2.subjects()):
                    if not is_property(e2, o2):
                        continue

                    sim = self.match_property(e1, e2, o1, o2, (soft_metric, general_metric), confidence_map,
                                              similarity_weights=sim_weights,
                                              disable_domain_range_similarity=disable_dr)

                    iterations += 1
                    if sim <= threshold:
                        continue

                    if e1 in property_map:
                        if property_map[e1][1] >= sim:
                            continue
                        elif property_map[e1][1] < sim:
                            final_alignment.pop((e1, property_map[e1][0]))
                            property_map.pop(property_map[e1][0])
                            property_map.pop(e1)

                    if e2 in property_map:
                        if property_map[e2][1] >= sim:
                            continue
                        elif property_map[e2][1] < sim:
                            final_alignment.pop((property_map[e2][0], e2))
                            property_map.pop(property_map[e2][0])
                            property_map.pop(e2)

                    d1 = o1.value(e1, RDFS.domain)
                    d2 = o2.value(e2, RDFS.domain)
                    confidence_map[(d1, d2)] = 0.66
                    final_alignment[(e1, e2)] = sim
                    property_map[e1] = (e2, sim)
                    property_map[e2] = (e1, sim)
                    if (e1, OWL.inverseOf, None) in o1 and (e2, OWL.inverseOf, None) in o2:
                        d1 = o1.value(o1.value(e1, OWL.inverseOf), RDFS.domain)
                        d2 = o2.value(o2.value(e2, OWL.inverseOf), RDFS.domain)

                        confidence_map[(d1, d2)] = 0.66
                        iv1, iv2 = o1.value(e1, OWL.inverseOf), o2.value(e2, OWL.inverseOf)
                        final_alignment[(iv1, iv2)] = sim
                        property_map[iv1] = (iv2, sim)
                        property_map[iv2] = (iv1, sim)

        return final_alignment, iterations
