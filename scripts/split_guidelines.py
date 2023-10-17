import json
import os

def split_jsonl(input_filepath, output_filepath, file):
    with open(input_filepath, 'r') as infile:
        count = 0
        for line in infile:
            count += 1
            guideline = json.loads(line)

            with open(output_filepath + f"/guideline_{file}_{count}.json", 'w') as outfile:
                json.dump(guideline, outfile, indent=4)

if __name__ == '__main__':
    filepath = "Guidelines/processed"
    for file in os.listdir(filepath):
        input_filepath = f'{filepath}/{file}'
        output_filepath = f'Guidelines/split_guidelines/{file}'
        if file.endswith(".jsonl"):
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)
            split_jsonl(input_filepath, output_filepath, file)
    print("Splitting completed!")