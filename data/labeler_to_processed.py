from glob import glob
import os
import json

def load_json_data(open_file_path):
    json_data = []
    with open(open_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_data.append(json.loads(line))
    return json_data

def process_data(json_data_list):
    
    result_list = []
    for json_data in json_data_list:
        processed_data = {}
        for item in json_data['processed']:
            if "name" in item and "value" in item:
                processed_data[item["name"]] = item["value"]
                
        result_list.append({"processed": processed_data})
        
    return result_list

source_path = "./labeler/*"
folder_list = glob(source_path)

for folder_path in folder_list:
    print(f"Working folder on {folder_path}")
    
    # Read the files in the folder
    files = glob(folder_path + "/*")
    for file_path in files:
        print(f"- Working files on {file_path}")
        
        current_file_data_list = load_json_data(open_file_path=file_path)
        result_processed_list = process_data(current_file_data_list)
        
        # Save the processed data
        save_path = file_path.replace("labeler", "processed")
        save_path = save_path.replace(".txt", ".jsonl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding="utf-8") as file:
            for item in result_processed_list:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        