import json
import re
import numpy as np
import copy
from tqdm import tqdm

from glob import glob
import os

from utils.template_fields import get_fields
final_result_fields, template_dict, fields_setting = get_fields()

from utils.operator_data import *

#: Args.
gloden_answer = "./data/instruction/advanced/eval.jsonl"
repeat_times = 3
finetuning_model_name_list = [
    "Nothing-format-original",
    "re-format-original",
    "golden-format-original",
    
    #: GPT-4o-mini
    "gpt-4o-mini-basic-original",
    "gpt-4o-mini-advanced-original",
    "gpt-4o-mini-oneShot-original",
    
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


def regular_process_item(item_dict, output_dict, fields_setting, template_dict):
    """
    根據原始程式中的 regular 處理，將輸出欄位做格式化處理
    """
    current_item_dict = template_dict.copy()
    
    for key, value in output_dict.items():
        if key in fields_setting['number_fields']:
            current_item_dict[key] = transform_chinese_number_to_int(value)
        elif key in fields_setting['fraction_fields']:
            current_item_dict[key] = blame_fraction_to_int(value)
        elif key in fields_setting['day_fields']:
            current_item_dict[key] = convert_to_days(value)
        elif key in fields_setting['date_fields']:
            current_item_dict[key] = date_regular(value)
        else:
            # 一般文字欄位做簡單處理
            current_item_dict[key] = chinese_char_to_int(value, zero_normalize=True) or ""

    return current_item_dict

def regular_process_data(data_list, fields_setting, template_dict, output_key):
    for index, item in enumerate(data_list):
        output_dict = item[output_key]
        processed_item = regular_process_item(item, output_dict, fields_setting, template_dict)
        data_list[index][output_key] = processed_item
    return data_list


### 你可以在這裡調整你的長度區間 (bins) ###
bin_ranges = [
    (0, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 8000),
    (8000, 9000),
    # 超過 9000 的可自行再決定是否要加
]


for time in range(repeat_times):
    for finetuning_model_name in finetuning_model_name_list:
        
        pre_output_path = f"./data/output/{finetuning_model_name}/{time}/"
        file_paths = [
            f for f in glob(pre_output_path + '*', recursive=True) 
            if 'processed_' in f and 'regular' not in f
        ]
        
        for file_path in file_paths:
            
            processed_file_name = os.path.basename(file_path)
            eval_output_path = f"./data/eval/{finetuning_model_name}/{time}/"
            os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
            print(eval_output_path)

            # 1) 載入 processed_data 和 gloden_data
            with open(pre_output_path + processed_file_name, 'r', encoding='utf-8-sig') as f:
                processed_data = [json.loads(line) for line in f]
            with open(gloden_answer, 'r', encoding='utf-8-sig') as f:
                gloden_data = [json.loads(line) for line in f]

            # 2) regular 處理
            regular_process_data(gloden_data, fields_setting, template_dict, 'output')
            regular_process_data(processed_data, fields_setting, template_dict, 'processed')
            
            # 3) 儲存 regular 檔 (供後續需求)
            regular_processed_file = os.path.join(pre_output_path, f"regular_processed_{processed_file_name}")
            with open(regular_processed_file, 'w', encoding='utf-8-sig') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # ---------------------------------------------------------
            # 以下開始「依照字數長度區間」分組來做評估，並計算筆數 (count)
            # ---------------------------------------------------------
            
            # bin_data 用來存取各 bin 的資料
            bin_data = {}
            for brange in bin_ranges:
                bin_label = f"{brange[0]}_{brange[1]}"
                bin_data[bin_label] = {
                    'golden_y_true_list': {field: [] for field in final_result_fields},
                    'processed_y_pred_list': {field: [] for field in final_result_fields},
                    'count': 0  # <-- 新增計算筆數
                }

            # 收集資料到對應 bin
            for idx, row in enumerate(processed_data):
                golden_output = gloden_data[idx]['output']
                generated_output = row['processed']
                
                # 取得該筆資料字數
                input_len = len(gloden_data[idx]['input'])
                
                # 若超過 9000，跳過
                if input_len > 9000:
                    continue

                # 找到所屬 bin
                matched_bin_label = None
                for brange in bin_ranges:
                    if brange[0] <= input_len < brange[1]:
                        matched_bin_label = f"{brange[0]}_{brange[1]}"
                        break
                if not matched_bin_label:
                    continue

                # 收集資料
                bin_data[matched_bin_label]['count'] += 1  # 筆數 +1
                for field in final_result_fields:
                    bin_data[matched_bin_label]['golden_y_true_list'][field].append(golden_output[field])
                    bin_data[matched_bin_label]['processed_y_pred_list'][field].append(generated_output[field])

            # 針對每個 bin 做評估
            bin_eval_results = {}
            for bin_label, data_dict in bin_data.items():
                golden_y_true_list = data_dict['golden_y_true_list']
                processed_y_pred_list = data_dict['processed_y_pred_list']
                
                # 評估結果
                eval_result_dict = {}
                eval_distance_result_dict = {}
                
                for field in final_result_fields:
                    y_true = golden_y_true_list[field]
                    y_pred = processed_y_pred_list[field]
                    
                    # 若該 field 無資料就給 0
                    if len(y_true) == 0:
                        eval_result_dict[field] = 0
                        eval_distance_result_dict[field] = 0
                        continue

                    if field in fields_setting['string_fields'] + fields_setting['date_fields']:
                        # 字串 → 平均餘弦相似度
                        avg_sim = calculate_average_cosine_similarity(y_true, y_pred)
                        eval_result_dict[field] = avg_sim
                        eval_distance_result_dict[field] = 0
                    else:
                        # 數值 → success_rate + distance
                        srate = success_rate(y_true, y_pred)
                        lcl = log_cosh_loss(y_true, y_pred)
                        eval_result_dict[field] = srate
                        eval_distance_result_dict[field] = lcl

                # 記錄該 bin 的結果(含 count)
                bin_eval_results[bin_label] = {
                    "eval_result_dict": eval_result_dict,
                    "eval_distance_result_dict": eval_distance_result_dict,
                    "count": data_dict["count"]  # 把 count 也存進去
                }
            
            # 將結果輸出至 length_ 檔案
            length_eval_file = os.path.join(eval_output_path, f"length_{processed_file_name}")
            with open(length_eval_file, 'w', encoding='utf-8-sig', newline='') as f:
                for bin_label, res in bin_eval_results.items():
                    eval_res = res["eval_result_dict"]
                    dist_res = res["eval_distance_result_dict"]
                    bin_count = res["count"]  # 該區間筆數

                    # 每個 field 都寫一行，但 count 通常相同
                    for field in final_result_fields:
                        record = {
                            "bin": bin_label,
                            "field": field,
                            "score": eval_res[field],
                            "distance": dist_res[field],
                            "count": bin_count
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')