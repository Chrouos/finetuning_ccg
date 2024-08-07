import json
import re
from glob import glob
import os

#: Args.
finetuning_model_name = "meta-chinese-format"
# checkpoint = "checkpoint-1200"

pre_output_path = f"./data/output/{finetuning_model_name}/"
# eval_file_name = f'eval-{checkpoint}.jsonl'
file_paths = [f for f in glob(pre_output_path + '*', recursive=True) if 'processed_' not in f ]

for file_path in file_paths:
    current_file_name = os.path.basename(file_path)
    
    with open(pre_output_path + current_file_name, 'r', encoding='utf-8-sig') as f:
        eval_datas = [json.loads(line) for line in f]

    # 定義一個函數來處理資料
    def extract_json(data_list):
        extracted_jsons = []
        for data in data_list:
            # 使用正則表達式來提取所有可能的 JSON 物件
            matches = re.findall(r"{.*?}(?=\s*```)|{.*}", data['processed'], re.DOTALL)
            if matches:
                json_part = matches[0]  # 提取第一個匹配的 JSON 物件
                try:
                    json_data = json.loads(json_part.replace("'", "\""))
                    extracted_jsons.append(json_data)
                except json.JSONDecodeError as e:
                    # print(f"Error decoding JSON: {e}")
                    extracted_jsons.append({})
            else:
                # print(f"No JSON found in data: {data['processed']}")
                extracted_jsons.append({})
        return extracted_jsons

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
