import json
import re
from glob import glob
import os

#: Args.
repeat_times = 3
finetuning_model_name_list = [
    "Nothing-format-original",
    "re-format-original",
    "golden-format-original",
    
    #: GPT-4o-mini
    "gpt-4o-mini-basic-original",
    "gpt-4o-mini-advanced-original",
    "gpt-4o-mini-oneShot-original",
    "gpt-4o-mini-automatedPrompt-original",
    
    "gpt-4o-mini-advanced-ft",
    
    #: GEMINI
    "gemini-1.5-flash-basic-original",
    "gemini-1.5-flash-advanced-original",
    "gemini-1.5-flash-oneShot-original",
    
    #: LLama-3-8B
    "Meta-Llama-3-8B-Instruct-basic-original",
    "Meta-Llama-3-8B-Instruct-advanced-original",
    "Meta-Llama-3-8B-Instruct-oneShot-original",
    
    "Meta-Llama-3-8B-Instruct-basic-checkpoint-900",
    "Meta-Llama-3-8B-Instruct-advanced-checkpoint-900",
    "Meta-Llama-3-8B-Instruct-oneShot-checkpoint-900",
    
    #: Taiwan LLAMA 8B
    "Llama-3-Taiwan-8B-Instruct-basic-original",
    "Llama-3-Taiwan-8B-Instruct-advanced-original",
    "Llama-3-Taiwan-8B-Instruct-oneShot-original",

    "Llama-3-Taiwan-8B-Instruct-basic-checkpoint-900",
    "Llama-3-Taiwan-8B-Instruct-advanced-checkpoint-900",
    "Llama-3-Taiwan-8B-Instruct-oneShot-checkpoint-900",
]

def extract_json(data_list):
    extracted_jsons = []
    for data in data_list:
        processed_data = data.get('processed', '')
        
        if isinstance(processed_data, dict):
            extracted_jsons.append(processed_data)
            continue
        
        if not isinstance(processed_data, str):
            processed_data = str(processed_data)

        # Clean up unnecessary backslashes and specific symbols
        cleaned_data = processed_data.replace("\\", "").replace("’", "'").replace("``", '"').strip()

        # Use regex to find potential JSON objects
        matches = re.findall(r"\{.*?\}(?=\s*```)|\{.*?\}", cleaned_data, re.DOTALL)
        if matches:
            # Select the best match based on the length (assuming longer JSON is more complete)
            best_match = max(matches, key=len)
            # Replace single quotes with double quotes for valid JSON
            json_candidate = best_match.replace("'", '"').strip()

            # Handle common issues in extracted JSON
            json_candidate = re.sub(r',(?=\s*[\}\]])', '', json_candidate)  # Remove trailing commas
            try:
                # Attempt to parse the JSON
                json_data = json.loads(json_candidate)
                extracted_jsons.append(json_data)
            except json.JSONDecodeError:
                # print(f"JSONDecodeError: {json_candidate[:200]}...")  # Truncate for readability
                extracted_jsons.append({})
        else:
            # print(f"No match found in: {cleaned_data[:200]}...")  # Truncate for readability
            extracted_jsons.append({})
    return extracted_jsons

for finetuning_model_name in finetuning_model_name_list:
    for time in range(repeat_times):
        pre_output_path = f"./data/output/{finetuning_model_name}/{time}/"
        file_paths = [f for f in glob(pre_output_path + '*', recursive=True) if 'processed_' not in f ]
        
        for file_path in file_paths:
            current_file_name = os.path.basename(file_path)
            
            with open(pre_output_path + current_file_name, 'r', encoding='utf-8-sig') as f:
                eval_datas = [json.loads(line) for line in f]

            # 提取 JSON 資料
            json_data_list = extract_json(eval_datas)
            processed_datas = []
            for index, json_data in enumerate(json_data_list):
                # print(json.dumps(json_data, indent=4, ensure_ascii=False))
                processed_datas.append({"processed": json_data})

            processed_file_name = f'processed_{current_file_name}'
            with open(pre_output_path + processed_file_name, 'w', encoding='utf-8-sig') as f:
                for item in processed_datas:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
