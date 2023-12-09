from os.path import join, exists
from os import listdir
from SPARQLWrapper import SPARQLWrapper, JSON
from unidecode import unidecode
from typing import Tuple
from tqdm import tqdm, trange
from functools import cache
from datasets import Dataset
from .utils import repair_json

import itertools
import difflib
import hashlib
import string
import random
import json
import sys
import re

WIKIDATA_ENDPOINT_URL = "https://query.wikidata.org/sparql"
DISEASE_OUT_CACHED_FILE = join(".", "data", "knowledge_database", "cached-data-{}.json")
STRUCTURED_GUIDELINES_FOLDER_PATH = join(".", "data", "knowledge_database", "guidelines", "structured_guidelines")
STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ''}
STOP_TOKEN = { 'disease' }
SEPARATOR = "<|>"

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def cache_result(query, fn):
    H = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    cache_file = DISEASE_OUT_CACHED_FILE.format(H)
    if not exists(cache_file):
        print(" [x] Running query: {}".format(H))
        contents = fn(query)
        
        with open(cache_file, 'w') as f:
            json.dump(contents, f)
    else:
        print(" [+] Using cached query: {}".format(H))
        with open(cache_file, 'r') as f:
            contents = json.load(f)
    return contents

def search(query, dataset_index_tags, dataset_index_ids):
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

def metric_search(dataset, search, q_init, n_max = 4):
    history = [([],[])]

    visited = set()
    class_of = [q_init]
    index = 0

    while index < n_max and len(class_of) > 0:
        current = class_of.pop(0)
        visited.add(current)
        current = dataset[current]
        if current['name'] in STOP_TOKEN:
            continue

        class_of += [j[0] for j in [search(x) for x in current['subclass_of']] if len(j) > 0 and j[0] not in visited]
        history.append(
            (current['study_by'] + current['health_speciality'], current['symptoms_and_signs'])
        )
        index += 1

    return history

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return itertools.chain.from_iterable(itertools.combinations(xs,n) for n in range(len(xs)+1))

def build_multiindex(dataset):
    index = {}
    for elem in dataset:
        spec = dataset[elem]['health_speciality'] + dataset[elem]['study_by']
        for comb in powerset(spec):
            if len(list(comb)) > 0:
                comb = tuple(sorted(comb))
                index[comb] = index.get(comb, []) + [elem]
    return index

def setup():
    query = """SELECT DISTINCT
        ?item
        ?itemLabel
        (GROUP_CONCAT(DISTINCT ?subclass_of_label; SEPARATOR="{SEPARATOR}") AS ?subclass_of)
        (GROUP_CONCAT(DISTINCT ?study_by_label; SEPARATOR="{SEPARATOR}") AS ?study_by)
        (GROUP_CONCAT(DISTINCT ?health_speciality_label; SEPARATOR="{SEPARATOR}") AS ?health_speciality) 
        (GROUP_CONCAT(DISTINCT ?symptoms_and_signs_label; SEPARATOR="{SEPARATOR}") AS ?symptoms_and_signs)
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

    contents = cache_result(query, lambda q: get_results(WIKIDATA_ENDPOINT_URL, q))
    alt_contents = cache_result(query2, lambda q: get_results(WIKIDATA_ENDPOINT_URL, q))

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

    def compute_it(_):
        matched_guidelines = []
        for guideline in tqdm(guidelines):
            matched_guidelines.append(
                dict(list(guideline.items()) + [("matched", search(guideline['label'], dataset_index_tags, dataset_index_ids))])
            )
        return matched_guidelines
    matched_guidelines = cache_result(SEPARATOR.join([g['label'] for g in guidelines] + dataset_index_ids), compute_it)

    A = len([x for x in matched_guidelines if x["matched"] == []])
    print("Unmatched proportion: {:.2f}% ({} elements)".format(A / len(matched_guidelines) * 100, A))

    @cache
    def search_it(query):
        return search(query, dataset_index_tags, dataset_index_ids)

    return matched_guidelines, dataset, search_it

def extract_field(domains):
    first_fields = set()
    i = -len(domains)
    for d in domains:
        first_fields.update(d)
        if len(d) >= 1:
            i = 0
        if i == 2:
            break
        i += 1
    return list(first_fields)

def proximity_heuristic(d1, dbase):
    if len(dbase) == 0:
        return 0

    score = 0
    for d in d1:
        if d in dbase:
            score += 1

    return (score) / len(dbase)

def find_matching_not_matching(dataset, search, multiindex, true_positive_q, ref_fields, n_max = 5):
    scores = 0
    elems = None

    for _ in range(n_max):
        q_init = random.choice(list(dataset.keys()))
        fields,_ = list(zip(*metric_search(dataset, search, q_init)))
        fields = extract_field(fields)
        heuristic = proximity_heuristic(fields, ref_fields)

        if (heuristic < scores or elems is None) and q_init != true_positive_q:
            elems = q_init
            scores = heuristic

    # Generate the powerset
    field_powerset = list(powerset(ref_fields))

    while True:
        elem = random.choice(field_powerset)
        if len(elem) == 0:
            continue
        elem = tuple(sorted(elem))
        
        if elem in multiindex:
            n_q = random.choice(multiindex[elem])
            return elems, n_q

def generate_dataset(labels: list[str], queries: list[str]) -> Tuple[list[str], list[str], list[str]]:
    # Extract guidelines and dataset
    guidelines, dataset, search = setup()
    multiindex = build_multiindex(dataset)

    # Match each Q... to a list of labels
    @cache
    def q_value_to_labels(q_value):
        return [dataset[q_value]['name']] + dataset[q_value]['alt']

    def q_value_to_random_label(q_value):
        # Cannot cache that badboi, non deterministic
        return '{"Condition": "TODO"}'.replace("TODO", random.choice(q_value_to_labels(q_value)).replace("\\", "\\\\").replace('"', '\\"'))

    # Generate dataset
    N = 8000
    accepted = []
    rejected = []
    text = []
    kernel_set = set([i['label'] for i in guidelines if i["matched"] == []])

    for _ in trange(N):
        # Pick an element at random
        rand_id = random.randint(0, len(labels))
        elem_json = repair_json(labels[rand_id])

        # Extract the Condition
        try:
            rand_elem = json.loads(elem_json)["Condition"]
        except ValueError as e:
            print('Invalid Json at index {}, ({})\nSelected resolving strategy: Skipping (close your eyes to stop seeing the issue)'.format(rand_id, elem_json))
            continue

        if rand_elem in kernel_set:
            rej = random.choice(labels)
            while rej == elem_json:
                rej = random.choice(labels)

            rejected.append(rej)
            accepted.append(elem_json)
            text.append(queries[rand_id])
        
        else:
            # Pick a disease
            q_init = [guideline["matched"] for guideline in guidelines if guideline["label"].lower() == rand_elem.lower()][0][0]

            # Check the history
            domains,_ = list(zip(*metric_search(dataset, search, q_init)))
            domains = extract_field(domains)
            if len(domains) == 0: # Partially fixes the halting problem
                kernel_set.add(rand_elem)
                rej = random.choice(labels)
                while rej == elem_json:
                    rej = random.choice(labels)

                rejected.append(rej)
                accepted.append(elem_json)
                text.append(queries[rand_id])

            else:
                q_min, q_max = find_matching_not_matching(dataset, search, multiindex, q_init, domains) # TODO: Fix the Halting problem... lol

                if q_value_to_random_label(q_min) != elem_json:
                    text.append(queries[rand_id])
                    accepted.append(q_value_to_random_label(q_max))
                    rejected.append(q_value_to_random_label(q_min))

    print('Generated length: {}'.format(len(text)))
    return text, accepted, rejected
