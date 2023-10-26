import os
import json

entries_to_remove = []
with open('non_disease_guidelines_names/filtered_BART_disease_wikidoc.txt', 'r') as file:
    for line in file:
        entries_to_remove.append(line[:-1])
count = 0
dest_path = "../Guidelines/split_guidelines_filtered/wikidoc.jsonl"
for filename in os.listdir(dest_path):
    file_path = os.path.join(dest_path, filename)
    if os.path.isfile(file_path) and filename.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            title = data.get('title', '')             
            if title in entries_to_remove:
                os.remove(file_path)
                count += 1
                print(f'Removed file: {title} name: {filename}')

print("Total removed :", count, "over", len(entries_to_remove))
