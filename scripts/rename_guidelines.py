import os
import json

# Define the folder path
folder_path = "../Guidelines/split_guidelines/idsa.jsonl"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # Open the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                title = data.get('title', None)

                # If title key exists in the JSON data
                if title:
                    # Convert title to a safe filename
                    new_filename = "".join([c if c.isalnum() or c.isspace() else "_" for c in title]) + ".json"
                    new_file_path = os.path.join(folder_path, new_filename)

                    # Rename the file
                    new_filename = new_filename[:50]
                    os.rename(file_path, new_file_path)
                    print(f'Renamed "{filename}" to "{new_filename}"')
                else:
                    print(f'No "title" found in "{filename}". Skipped.')

            except json.JSONDecodeError:
                print(f"Failed to decode JSON for {filename}. Skipped.")

