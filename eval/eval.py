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

for time in range(repeat_times):
    consoletext=[]
    sequence_list=[]
    for finetuning_model_name in finetuning_model_name_list:

        pre_output_path = f"./data/output/{finetuning_model_name}/{time}/"
        file_paths = [f for f in glob(pre_output_path + '*', recursive=True) if 'processed_' in f and 'regular' not in f]
        

        for file_path in file_paths:
            
            processed_file_name = os.path.basename(file_path)
            eval_output_path = f"./data/eval/{finetuning_model_name}/{time}/"
            os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
            print(eval_output_path)

            #- load datas.
            with open(pre_output_path + processed_file_name, 'r', encoding='utf-8-sig') as f:
                processed_data = [json.loads(line) for line in f]
                # processed
                
            with open(gloden_answer, 'r', encoding='utf-8-sig') as f:
                gloden_data = [json.loads(line) for line in f]
                # output

            #- regular
            def regular_process_item(item_dict, output_dict, fields_setting, template_dict):
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
                        current_item_dict[key] = chinese_char_to_int(value, zero_normalize=True) or ""

                return current_item_dict

            def regular_process_data(data_list, fields_setting, template_dict, output_key):
                for index, item in enumerate(data_list):
                    output_dict = item[output_key]
                    processed_item = regular_process_item(item, output_dict, fields_setting, template_dict)
                    data_list[index][output_key] = processed_item
                    
                return data_list

            regular_process_data(gloden_data, fields_setting, template_dict, 'output')
            regular_process_data(processed_data, fields_setting, template_dict, 'processed')
            
            #- Save processed files with "regular_" prefix
            regular_processed_file = os.path.join(pre_output_path, f"regular_processed_{processed_file_name}")

            with open(regular_processed_file, 'w', encoding='utf-8-sig') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            data_total_length = len(processed_data)
                
            #- eval
            original_processed_data = copy.deepcopy(processed_data)
            
            golden_y_true_list = {field: [] for field in final_result_fields} # = 準備序列
            processed_y_pred_list = {field: [] for field in final_result_fields} # = 準備序列
            
            # @ 獲得序列
            error_weight_count = 0
            # print(len(original_processed_data))
            for index_outer, row in enumerate(original_processed_data):
                
                #. 檢查忽略條件
                is_pass = len(gloden_data[index_outer]['input']) > 9000  # 檢查字數
                output_data = gloden_data[index_outer]['output']
                if is_pass or any(value != "" and value != 0 for key, value in output_data.items() if key != '被告肇責') == False: 
                    continue
                if gloden_data[index_outer]['output'] != original_processed_data[index_outer]['processed'] and any(value != "" and value != 0 for key, value in original_processed_data[index_outer]['processed'].items() if key != '被告肇責') == False: 
                    error_weight_count += 1
                    
                # 計數
                for item_key in final_result_fields:
                    golden_y_true_list[item_key].append(gloden_data[index_outer]['output'][item_key])
                    processed_y_pred_list[item_key].append(original_processed_data[index_outer]['processed'][item_key]) 
            
            print(f"error_weight: {error_weight_count}")
            sequence_list.append({
                "file_path": file_path,
                "golden_y_true_list": golden_y_true_list,
                "processed_y_pred_list": processed_y_pred_list
            })
                
            # @ 計算
            eval_result_dict = {field: 0 for field in final_result_fields}       # 結論
            eval_distance_result_dict =  {field: 0 for field in final_result_fields}
            eval_result_count_dict = {field: {"golden": 0, "processed": 0} for field in final_result_fields} # 共有幾筆
            
            for item_key in final_result_fields:
                
                eval_result_count_dict[item_key]["golden"] = len(golden_y_true_list[item_key])
                eval_result_count_dict[item_key]["processed"] = len(processed_y_pred_list[item_key])
                current_error_weight = 0
                
                # @ 字串 
                if item_key in fields_setting['string_fields'] + fields_setting['date_fields']:
                    eval_result_dict[item_key] = max(calculate_average_cosine_similarity(golden_y_true_list[item_key], processed_y_pred_list[item_key]) - current_error_weight, 0)
                    
                # @ 數值
                elif item_key in fields_setting['number_fields']  + fields_setting['day_fields'] + fields_setting['fraction_fields']:
                   
                    # eval_result_dict[item_key] = max(kohens_kappa(golden_y_true_list[item_key], processed_y_pred_list[item_key]) - current_error_weight, 0)
                    eval_result_dict[item_key] = max(success_rate(golden_y_true_list[item_key], processed_y_pred_list[item_key]) - current_error_weight, 0)

                    eval_distance_result_dict[item_key] = log_cosh_loss(golden_y_true_list[item_key], processed_y_pred_list[item_key]) * current_error_weight
                    
                consoletext.append(f"[{file_path}][{item_key}]\nGOLDEN=\n{golden_y_true_list[item_key]}\nGENERATE=\n{processed_y_pred_list[item_key]}\n\n")
                
            for key, value in eval_result_dict.items():
                print(f"{finetuning_model_name} - {processed_file_name} - {key} ({str(eval_result_count_dict[item_key]['golden'])}) - {value}")
                
            with open(eval_output_path + processed_file_name, 'w', encoding='utf-8-sig', newline='') as f:
                for key, average in eval_result_dict.items():
                    f.write(json.dumps({key: average}, ensure_ascii=False) + '\n')
                    
            with open(eval_output_path + 'distance_' + processed_file_name, 'w', encoding='utf-8-sig', newline='') as f:
                for key, average in eval_distance_result_dict.items():
                    f.write(json.dumps({key: average}, ensure_ascii=False) + '\n')
                    
    with open('consoletext.txt', 'w', encoding='utf-8') as file:
        for line in consoletext:
            file.write(line + '\n')
            
            
