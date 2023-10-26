import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# model_name = "medicalai/ClinicalBERT"
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_zero_shot(text, labels):
    hypotheses = [f"This text is about {label}." for label in labels]

    premise_encodings = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)
    hypothesis_encodings = tokenizer.batch_encode_plus(hypotheses, return_tensors="pt", max_length=512, truncation=True, padding=True)

    input_ids = torch.cat([premise_encodings["input_ids"]] * len(labels), dim=0)
    attention_mask = torch.cat([premise_encodings["attention_mask"]] * len(labels), dim=0)

    paired_input_ids = [torch.cat([input_ids[i], hypothesis_encodings["input_ids"][i]], dim=-1) for i in range(len(labels))]
    paired_attention_mask = [torch.cat([attention_mask[i], hypothesis_encodings["attention_mask"][i]], dim=-1) for i in range(len(labels))]

    paired_input_ids = torch.stack(paired_input_ids)
    paired_attention_mask = torch.stack(paired_attention_mask)

    with torch.no_grad():
        logits = model(input_ids=paired_input_ids, attention_mask=paired_attention_mask).logits

    entail_contradiction_logits = logits[:, [0, 2]]

    probs = entail_contradiction_logits.softmax(dim=1)[:, 1].cpu().numpy()

    return labels[probs.argmax()]


#def preprocess(doc):
#    stop_words = set(stopwords.words('english'))
#    tokens = word_tokenize(doc.lower())
#    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
#    tokens = " ".join(tokens)
#    return tokens

#dictionary_documents = []
#documents = []
# folder_path = "../Guidelines/split_guidelines/idsa.jsonl"
# for file in os.listdir(folder_path):
#     if file.endswith(".json"):
#         with open(f"{folder_path}/{file}", 'r') as f:
#             data = json.load(f)
#             preprocess_doc = preprocess(data["title"])
#             # teste sur les titres
#             documents.append(preprocess_doc)
#             dictionary_documents.append({data["title"]: preprocess_doc})

#file_list = []
# with open("anomalous_titles_015.txt", "r") as f:
#     file_list = f.readlines()
#     file_list = [doc.strip() for doc in file_list]

folder_path_over = "../Guidelines/split_guidelines/"
for filename in os.listdir(folder_path_over):
    origin_path = os.path.join(folder_path_over, filename)
    dest_path = "../Guidelines/split_guidelines_filtered/"
    dest_path = os.path.join(dest_path, filename)
    print(filename)
    if filename != "cma_pdfs.jsonl" and filename != "rch.jsonl" and filename != "gc.jsonl" and filename != "cps.jsonl" and filename != "cco.jsonl" and filename != "icrc.jsonl" and filename != "magic.jsonl" and filename != "who.jsonl":
        wikidoc_file_list = []
        #folder_path = "../Guidelines/split_guidelines/aafp.jsonl"
        for file in os.listdir(origin_path):
            if file.endswith(".json"):
                with open(f"{origin_path}/{file}", 'r') as f:
                    data = json.load(f)
                    wikidoc_file_list.append(data["title"])

        results = []
        entries_to_remove = []
        #with open("non_disease_guidelines_names/filtered_BART_disease_aafp.txt", "w") as f:
        for doc in tqdm(wikidoc_file_list):
            text = doc
            labels = ["disease", "not disease"]
            result = classify_zero_shot(text, labels)
            results.append(result)
            if result == "not disease":
                #print(doc)
                # print("---------------------------------------------------")
                entries_to_remove.append(doc)
                #f.write(doc + "\n")
                #print(f"The text is classified as: {result}")

        #folder_path = "../Guidelines/split_guidelines_filtered/aafp.jsonl"
        for filename in os.listdir(dest_path):
                file_path = os.path.join(dest_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.json'):
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        title = data.get('title', '') 
                        if title in entries_to_remove:
                            os.remove(file_path)
                            print(f'Removed file: {title} name: {filename}')
    
        print("Total removed :", len(entries_to_remove))