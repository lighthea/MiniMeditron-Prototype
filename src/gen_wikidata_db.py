import os
import sys
import time
from os.path import join, exists
from os import listdir

import requests
import json
from tqdm import tqdm

from qwikidata.entity import WikidataItem, WikidataClaimGroup
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

WIKIDATA_COMPRESSED_FILE_PATH = join("..", "wikidata", "wikidata-20220103-all.json.gz")
WIKIDATA_OUT_FILE_NAME = join(".", "diseases-out.json")


P_INSTANCE_OF = "P31"
P_SUBCLASS_OF = "P279"
Q_MATCHED = set([
    "Q112193867",
    "Q2057971",
    "Q12136"
])

# create an instance of WikidataJsonDump
wjd = WikidataJsonDump(WIKIDATA_COMPRESSED_FILE_PATH)

def find_ids(claim_group: WikidataClaimGroup) -> list[str]:
    return [
        claim.mainsnak.datavalue.value["id"]
        for claim in claim_group
        if claim.mainsnak.snaktype == "value"
    ]

# create an iterable of WikidataItem 
diseases = []
t1 = time.time()
try:
    for ii, entity_dict in tqdm(enumerate(wjd), total=107795041):
        if entity_dict["type"] == "item":
            entity = WikidataItem(entity_dict)
            instance_of = find_ids(entity.get_claim_group(P_INSTANCE_OF))
            subclass_of = find_ids(entity.get_claim_group(P_SUBCLASS_OF))

            # if any(x in Q_MATCHED for x in instance_of + subclass_of):
            #     diseases.append(entity)

            if "Q112193867" in instance_of or "Q2057971" in instance_of or "Q12136" in instance_of or \
                "Q112193867" in subclass_of or "Q2057971" in subclass_of or "Q12136" in subclass_of:
                diseases.append(entity)


        if ii % 100000 == 0 and ii > 0:
            t2 = time.time()
            dt = t2 - t1
            tqdm.write("Found {} records among {} entities [entities/s: {:.2f}]".format(
                len(diseases), ii, ii / dt
            ))
except KeyboardInterrupt as e:
    print("Interrupting...")

# write the iterable of WikidataItem to disk as JSON
t2 = time.time()
dt = t2 - t1
print("Found {} records in {:.1f}min".format(len(diseases), dt/60.0))
dump_entities_to_json(diseases, WIKIDATA_OUT_FILE_NAME)
wjd_filtered = WikidataJsonDump(WIKIDATA_OUT_FILE_NAME)
