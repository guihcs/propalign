import sys

from sentence_transformers import SentenceTransformer
from models import Finbank
from property_matching import PropertyMatcher
import os
import requests
import argparse
import rdflib
import tempfile
from urllib import parse, request
from om.ont import get_namespace
import json
from typing import Union
import re

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, Response, UploadFile, File
from fastapi.responses import PlainTextResponse, Response
from typing_extensions import Annotated

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

wm = Finbank('./fin.bin')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
property_matcher = PropertyMatcher(wm, model)


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


@app.post('/match')
async def match(source: Union[str, UploadFile] = Form(...),
                target: Union[str, UploadFile] = Form(...),
                inputAlignment: Annotated[Union[str, None], Form()] = None,
                parameters: Annotated[Union[str, None], Form()] = None):
    outputFile = type(source) != str

    if type(source) == str:
        o1 = rdflib.Graph().parse(source)
        o2 = rdflib.Graph().parse(target)
    else:

        o1 = rdflib.Graph().parse(source.file, format=re.split(r'\W', source.content_type)[-1])
        o2 = rdflib.Graph().parse(target.file, format=re.split(r'\W', target.content_type)[-1])

    params = {}

    if parameters is not None:
        with open(parameters) as f:
            params = json.load(f)

    p, it = property_matcher.match_ontologies(o1, o2, 0.65,
                                              sim_weights=params['sim_weights'] if 'sim_weights' in params else None)

    if 'format' in params and params['format'] == 'sssom':
        result = ssom(p)
        suffix = '.tsv'
    else:
        if outputFile:
            source = source.filename
            target = target.filename
        result = toAlignFormat(p, get_namespace(o1), get_namespace(o2), source, target)
        suffix = '.rdf'

    if outputFile:
        return Response(result, media_type='application/rdf+xml')
    else:
        with tempfile.NamedTemporaryFile('w', prefix='alignment_', suffix=suffix, delete=False) as out_file:
            out_file.write(result)

        return PlainTextResponse(out_file.name)
