import os
import sys
from os.path import join, exists
from os import listdir

import tqdm
import requests
import json

from qwikidata.entity import WikidataItem, WikidataClaimGroup
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

WIKIDATA_COMPRESSED_FILE_PATH = join("..", "wikidata", "wikidata-20220103-all.json")

P_INSTANCE_OF = "P31"
P_SUBCLASS_OF = "P279"
Q_CLASS_OF_DISEASE = "Q112193867"
Q_DISEASE = "Q12136"

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
for ii, entity_dict in enumerate(wjd):
    if entity_dict["type"] == "item":
        entity = WikidataItem(entity_dict)
        instance_of = set(find_ids(entity.get_claim_group(P_INSTANCE_OF)))
        subclass_of = set(find_ids(entity.get_claim_group(P_SUBCLASS_OF)))

        if Q_DISEASE in subclass_of or Q_CLASS_OF_DISEASE in instance_of or Q_DISEASE in instance_of or Q_CLASS_OF_DISEASE in subclass_of:
            diseases.append(entity)
