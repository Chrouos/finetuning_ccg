﻿import json
import re
from glob import glob
import os

#: Args.
repeat_times = 1
finetuning_model_name_list = [
    
    # "golden-format-original",
    
    # #: GPT-4o-mini
    # "gpt-4o-mini-basic-original",
    # "gpt-4o-mini-advanced-original",
    # "gpt-4o-mini-oneShot-original",
    
    # "gpt-4o-mini-basic-ft",
    # "gpt-4o-mini-advanced-ft",
    # "gpt-4o-mini-oneShot-ft",

    #: LLama-3.1-8B
    # "Llama-3.1-8B-Instruct-basic-original",
    "Llama-3.1-8B-Instruct-advanced-original",
    # "Llama-3.1-8B-Instruct-oneShot-original",
    
    # "Llama-3.1-8B-Instruct-basic-checkpoint-1200",
    "Llama-3.1-8B-Instruct-advanced-checkpoint-1200",
    # "Llama-3.1-8B-Instruct-oneShot-checkpoint-1200",
    
    # #: LLama-3.2-3B
    # "Llama-3.2-3B-Instruct-basic-original",
    # "Llama-3.2-3B-Instruct-advanced-original",
    # "Llama-3.2-3B-Instruct-oneShot-original",
    
    # "Llama-3.2-3B-Instruct-basic-checkpoint-1200",
    # "Llama-3.2-3B-Instruct-advanced-checkpoint-1200",
    # "Llama-3.2-3B-Instruct-oneShot-checkpoint-1200",
    
    # #: LLama-3-8B
    # "Meta-Llama-3-8B-Instruct-basic-original",
    # "Meta-Llama-3-8B-Instruct-advanced-original",
    # "Meta-Llama-3-8B-Instruct-oneShot-original",
    
    # "Meta-Llama-3-8B-Instruct-basic-checkpoint-1200",
    # "Meta-Llama-3-8B-Instruct-advanced-checkpoint-1200",
    # "Meta-Llama-3-8B-Instruct-oneShot-checkpoint-1200",
    
    # #: Taiwan LLAMA 8B
    # "Llama-3-Taiwan-8B-Instruct-basic-original",
    # "Llama-3-Taiwan-8B-Instruct-advanced-original",
    # "Llama-3-Taiwan-8B-Instruct-oneShot-original",

    # "Llama-3-Taiwan-8B-Instruct-basic-checkpoint-1200",
    # "Llama-3-Taiwan-8B-Instruct-advanced-checkpoint-1200",
    # "Llama-3-Taiwan-8B-Instruct-oneShot-checkpoint-1200",
]

for finetuning_model_name in finetuning_model_name_list:
    for time in range(repeat_times):
        pre_output_path = f"./data/output/{finetuning_model_name}/{time}/"
        file_paths = [f for f in glob(pre_output_path + '*', recursive=True) if 'processed_' not in f ]
        
        for file_path in file_paths:
            current_file_name = os.path.basename(file_path)
            
            with open(pre_output_path + current_file_name, 'r', encoding='utf-8-sig') as f:
                eval_datas = [json.loads(line) for line in f]


                def extract_json(data_list):
                    extracted_jsons = []
                    for data in data_list:
                        processed_data = data.get('processed', '')
                        if not isinstance(processed_data, str):
                            processed_data = str(processed_data)

                        # 清除 ` ```json` 或 ` ``` ` 的出現
                        cleaned_data = processed_data.replace("\\", "").replace("```json", "").replace("```", "")

                        # 使用正則表達式來抓取 JSON 對象
                        matches = re.findall(r"{.*?}(?=\s*```)|{.*}", cleaned_data, re.DOTALL)
                        if matches:
                            json_part = matches[0]  # 取得第一個 JSON 物件匹配
                            # 檢查並補全引號
                            if json_part.count('"') % 2 != 0:
                                json_part += '"'  # 如果引號數不平衡，補一個結束引號

                            try:
                                # 嘗試解析 JSON，若失敗則跳過
                                json_data = json.loads(json_part.replace("'", "\""))
                                extracted_jsons.append(json_data)
                            except json.JSONDecodeError:
                                extracted_jsons.append({})  # 若解析失敗，則附加空字典
                        else:
                            extracted_jsons.append({})  # 若無匹配的 JSON，則附加空字典

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
