from sentence_transformers import SentenceTransformer
from models import Finbank
import random
import torch
import numpy as np
from property_matching import PropertyMatcher
from tqdm.auto import tqdm
from property_matching import most_common_pair
import matplotlib.pyplot as plt
import argparse
import rdflib
import tempfile
from urllib import parse, request
from om.ont import get_namespace


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='LD similarity.')

    arg_parser.add_argument('source', help='Source ontology path.')
    arg_parser.add_argument('target', help='Target ontology path.')
    arg_parser.add_argument('--output', dest='output', default='./output', help='Folder to save the results.')
    arg_parser.add_argument('--format', dest='format', default='align', choices=['align', 'sssom'], help='Output format.')

    return arg_parser.parse_args()


def toAlignFormat(aligns, onto1, onto2, location1, location2):
    data = ["""<?xml version='1.0' encoding='utf-8' standalone='no'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'
         xmlns:align='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'>"""]

    data.append(f"""    <Alignment>
        <xml>yes</xml>
        <level>0</level>
        <type>**</type>
        <onto1>
            <Ontology rdf:about="{onto1}">
                <location>{location1}</location>
            </Ontology>
        </onto1>
        <onto2>
            <Ontology rdf:about="{onto2}">
                <location>{location2}</location>
            </Ontology>
        </onto2>""")

    for (entity1, entity2), confidence in aligns.items():
        data.append(f"""        <map>
            <Cell>
                <entity1 rdf:resource="{entity1}"/>
                <entity2 rdf:resource="{entity2}"/>
                <relation>=</relation>
                <measure rdf:datatype="http://www.w3.org/2001/XMLSchema#float">{confidence}</measure>
            </Cell>
        </map>""")

    data.append("""    </Alignment>
</rdf:RDF>""")

    return '\n'.join(data)

def ssom(aligns):
    lines = ['subject_id\tpredicate_id\tobject_id\tmapping_justification\tconfidence']
    for (entity1, entity2), confidence in aligns.items():
        lines.append(f"{entity1}\tskos:exactMatch\t{entity2}\tsemapv:LexicalMatching\t{confidence}")

    return "\n".join(lines)

if __name__ == '__main__':
    args = parse_arguments()
    wm = Finbank('/home/guilherme/Documents/kg/fin.bin')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    property_matcher = PropertyMatcher(wm, model)

    o1 = rdflib.Graph().parse(args.source)
    o2 = rdflib.Graph().parse(args.target)

    p, it = property_matcher.match_ontologies(o1, o2, 0.65)



    # Parser


    if args.format == 'sssom':
        result = ssom(p)
        suffix = '.tsv'
    else:
        result = toAlignFormat(p, get_namespace(o1), get_namespace(o2), args.source, args.target)
        suffix = '.rdf'

    with tempfile.NamedTemporaryFile('w', prefix='alignment_', suffix=suffix, delete=False) as out_file:
        out_file.write(result)

        print(parse.urljoin("file:", request.pathname2url(out_file.name)))
