import json
import re
import numpy as np
import copy

from utils.template_fields import get_fields
final_result_fields, template_dict, fields_setting = get_fields()

from utils.operator_data import *

#: Args.
finetuning_model_name = "openai"
# finetuning_model_name = "meta-chinese-format"
# checkpoint = "checkpoint-1800"
gloden_answer = "./data/ccg/format/eval.jsonl"

pre_output_path = f"./data/output/{finetuning_model_name}/"
# eval_file_name = f'eval-{checkpoint}.jsonl'
# processed_file_name = f'processed_{eval_file_name}'
processed_file_name = f"eval.jsonl"

eval_output_path = f"./data/eval/{finetuning_model_name}/"
os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

#- load datas.
with open(pre_output_path + processed_file_name, 'r', encoding='utf-8-sig') as f:
    processed_data = [json.loads(line) for line in f]
    # processed
original_processed_data = copy.deepcopy(processed_data)
    
with open(gloden_answer, 'r', encoding='utf-8-sig') as f:
    gloden_data = [json.loads(line) for line in f]
    # output

#- regular
def regular_process_item(item_dict, output_dict, fields_setting, template_dict):
    current_item_dict = template_dict.copy()
    for key, value in output_dict.items():
        if value is None:
            current_item_dict[key] = ""
        elif key in fields_setting['number_fields']:
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
    
data_total_length = len(processed_data)
    
#- eval
comparison_diff_dict = [{field: "" for field in final_result_fields} for _ in range(data_total_length)] # = total_comparison_diff_dict # = 每一格的差異
comparison_gap_dict = {field: 0 for field in final_result_fields} # = total_comparison_gap_dict # = 差異的總和
comparison_worst_diff = {field: 0 for field in final_result_fields } # = total_comparison_worst_diff # = 獲得總差異(最差最差的情況)

efficient_count = 0
for index, item in enumerate(original_processed_data[:5]):
    if item['processed'] == {}: continue
    else: efficient_count += 1
    
    
    for key in item['processed']:
        try:
            # @ 比對 數字類 (數字、分數、天數)
            if key in fields_setting['number_fields'] + fields_setting['fraction_fields'] + fields_setting['day_fields']:
                diff = int(gloden_data[index]['output'].get(key, 0)) - int( processed_data[index]['processed'].get(key, 0)) # = 當前的差異
                
                comparison_diff_dict[index][key] = diff # = 每一格的差異
                comparison_gap_dict[key] += abs(diff) # = 差異的總和
                comparison_worst_diff[key] += max(int( gloden_data[index]['output'].get(key, 0)), int( processed_data[index]['processed'].get(key, 0))) 
                # = 獲得總差異(最差最差的情況)
                
            # @ 比對 字串類 (字串、日期)
            elif key in fields_setting['string_fields'] + fields_setting['date_fields']:
                
                if gloden_data[index]['output'][key] is np.nan: text_kernel = ""
                else: text_kernel = gloden_data[index]['output'].get(key, "")
                if processed_data[index]['processed'][key] is np.nan: text_basic = ""
                else: text_basic = processed_data[index]['processed'].get(key, "")
                
                # @ 比對
                if text_basic == "" and text_kernel == "": comparison_gap_dict[key] += 1
                elif text_basic == "" and text_kernel != "": comparison_gap_dict[key] += 0 # comparison_diff_dict[index][key] = comparison_user_basic[0:2] # = 每一格的差異
                elif text_basic != "" and text_kernel == "": comparison_gap_dict[key] += 0 # comparison_diff_dict[index][key] = comparison_user_kernel[0:2] # = 每一格的差異
                else: comparison_gap_dict[key] += calculate_cosine_similarity(text_basic, text_kernel)
                # print(key, comparison_gap_dict[key], f"({text_basic}, {text_kernel})")
                
                if key == '事發經過':
                    print(f"{key}\nG:{gloden_data[index]['output'][key]}\nP:{processed_data[index]['processed'][key]}\n=>{comparison_gap_dict[key]}\n")
            
        except Exception as e:
            print(key, gloden_data[index]['output'][key], processed_data[index]['processed'][key],  e)
            
            
comparison_average_dict = {}
current_average_dict = {}
key_average_dict = {field: 0 for field in final_result_fields} 
for key, gap in comparison_gap_dict.items():
    result = abs(gap)  # = 比絕對差異
    try:
        
        # - 如果是數值
        if key in fields_setting['number_fields'] + fields_setting['fraction_fields'] + fields_setting['day_fields']:
            if comparison_worst_diff[key] != 0: 
                result = 1 - (result / comparison_worst_diff[key])
                # ~ 標準化 0 ~ 1 (數字)
            else: result = 1
            
        # - 如果是字串
        if key in fields_setting['string_fields'] + fields_setting['date_fields']: # = 原本是　０～ data_total_length
            result = result / efficient_count
            
        current_average_dict[key] = round(result, 2)
        key_average_dict[key] += abs(result) # = 最後總結
            
    except Exception as e:
        print(e)
        
print("---------", "current_average_dict", "---------")
for key, average in current_average_dict.items():
    print(key, average)

with open(eval_output_path + processed_file_name, 'w', encoding='utf-8-sig', newline='') as f:
    for key, average in current_average_dict.items():
        # 寫入鍵值對為一個字典格式
        f.write(json.dumps({key: average}, ensure_ascii=False) + '\n')
        