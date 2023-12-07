from os.path import join, exists
from os import listdir
from SPARQLWrapper import SPARQLWrapper, JSON
from unidecode import unidecode
from tqdm import tqdm

import difflib
import hashlib
import numpy as np
import string
import json
import sys
import re

WIKIDATA_ENDPOINT_URL = "https://query.wikidata.org/sparql"
DISEASE_OUT_CACHED_FILE = join("..", "data", "knowledge_database", "query-output-cached-{}.json")
STRUCTURED_GUIDELINES_FOLDER_PATH = join("..", "data", "knowledge_database", "guidelines", "structured_guidelines")
STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ''}
SEPARATOR = "<|>"

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def cache_result(query):
    H = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    cache_file = DISEASE_OUT_CACHED_FILE.format(H)
    if not exists(cache_file):
        print(" [x] Running query: {}".format(H))
        contents = get_results(WIKIDATA_ENDPOINT_URL, query)
        
        with open(cache_file, 'w') as f:
            json.dump(contents, f)
    else:
        print(" [+] Using cached query: {}".format(H))
        with open(cache_file, 'r') as f:
            contents = json.load(f)
    return contents

def search(query):
    xs = tokenizer(query)
    r = difflib.get_close_matches(xs, dataset_index_tags, n=5, cutoff=0.1)

    # The version bellow ensure that keys stays order
    return list(dict.fromkeys([dataset_index_ids[dataset_index_tags.index(r)] for r in r]).keys())

def map_list(it, fn):
    return [i for i in map(fn, it) if i is not None]

def map_if(val, fn):
    if val is None:
        return None
    return fn(val)

def retrieve(object_, key, default):
    keys = key.split('/')
    for key in keys:
        object_ = object_.get(key, None)
        if object_ is None:
            return default
    return object_

def partial_lowercase(word):
    if all(x in string.ascii_uppercase + string.digits + '-' for x in word):
        return word
    return word.lower()

def tokenizer(sentence):
    sentence = unidecode(sentence)
    f1 = [x.group().lower() for x in re.finditer(r'[a-zA-Z0-9\-]+', sentence)]
    f2 = [x for x in f1 if not x in STOP_WORDS]
    f3 = [x for x in f2 if not all(y in string.digits for y in x)]
    return f3

def generate_dataset(dataset):
    query = """SELECT DISTINCT
        ?item
        ?itemLabel
        (GROUP_CONCAT(DISTINCT ?subclass_of_label; SEPARATOR="{SEPARATOR}") AS ?subclass_of)
        (GROUP_CONCAT(DISTINCT ?study_by_label; SEPARATOR="{SEPARATOR}") AS ?study_by)
        (GROUP_CONCAT(DISTINCT ?health_speciality_label; SEPARATOR="{SEPARATOR}") AS ?health_speciality) 
        (GROUP_CONCAT(DISTINCT ?symptoms_and_signs_label; SEPARATOR="{SEPARATOR}") AS ?symptoms_and_signs) # ?study_by
    WHERE {
        ?item wdt:P31/wdt:P279* wd:Q112193867.
        OPTIONAL { ?item wdt:P279 ?subclass_of. }
        OPTIONAL { ?item wdt:P2579 ?study_by. }
        OPTIONAL { ?item wdt:P1995 ?health_speciality. }
        OPTIONAL { ?item wdt:P780 ?symptoms_and_signs. }
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en".
            ?subclass_of rdfs:label ?subclass_of_label .
            ?study_by rdfs:label ?study_by_label .
            ?health_speciality rdfs:label ?health_speciality_label .
            ?symptoms_and_signs rdfs:label ?symptoms_and_signs_label .
            ?item rdfs:label ?itemLabel .
        }
    }
    GROUP BY ?item ?itemLabel
    """.replace("{SEPARATOR}", SEPARATOR)

    query2 = """
    SELECT DISTINCT
    ?item
    (GROUP_CONCAT(DISTINCT ?alt_label; SEPARATOR="{SEPARATOR}") AS ?alt_labels2)
    WHERE {
        ?item wdt:P31/wdt:P279* wd:Q112193867.
        OPTIONAL { ?item skos:altLabel ?alt_label . FILTER (lang(?alt_label) = "en") }
        SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en".
        }
    }
    GROUP BY ?item
    """.replace("{SEPARATOR}", SEPARATOR)

    contents = cache_result(query)
    alt_contents = cache_result(query2)

    dataset = {}
    alt_table = {}

    for alt in alt_contents["results"]["bindings"]:
        id = map_if(retrieve(alt, "item/value", None), lambda x: x.split('/')[-1])
        alt = map_if(retrieve(alt, "alt_labels2/value", None), lambda x: x.split(SEPARATOR))
        alt_table[id] = alt

    for content in contents["results"]["bindings"]:
        label_en = retrieve(content, 'itemLabel/value', None)
        if label_en is None:
            continue

        id = map_if(retrieve(content, "item/value", None), lambda x: x.split('/')[-1])
        elem = {
            "id": id,
            "name": label_en,
            "alt": alt_table[id] if id in alt_table else []
        }

        for key in ["subclass_of", "study_by", "health_speciality", "symptoms_and_signs"]:
            elem[key] = map_if(retrieve(content, key + "/value", None), lambda x: x.split(SEPARATOR) if len(x) > 0 else [])

        dataset[id] = elem

    dataset_index_tags = []
    dataset_index_ids = []

    for elem in dataset.values():
        id = elem['id']
        names = elem['alt'] + [elem['name']]
        for name in names:
            name = tokenizer(name)

            if name == []:
                continue

            dataset_index_tags.append(name)
            dataset_index_ids.append(id)

    guidelines = []

    for file in listdir(STRUCTURED_GUIDELINES_FOLDER_PATH):
        path = join(STRUCTURED_GUIDELINES_FOLDER_PATH, file)
        if path.endswith('.jsonl'):
            with open(path, 'r') as f:
                guidelines += list(map(json.loads, f.readlines()))

    matched_guidelines = []
    for guideline in tqdm(guidelines):
        matched_guidelines.append(
            dict(list(guideline.items()) + [("matched", search(guideline['label']))])
        )

    A = len([x for x in matched_guidelines if x["matched"] == []])
    print("Unmatched proportion: {:.2f}% ({} elements)".format(A / len(matched_guidelines) * 100, A))

    # Generate dataset
    N = 10
    
    
