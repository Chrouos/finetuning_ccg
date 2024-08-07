import textwrap
import json
import random
import os
import argparse
'''
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需要包含：零件、材料、工資、鈑金、塗裝、烤漆。
    3. 若無結果則留空白
    4. 注意每日每月之單位，一個月為30天
    5. 每日工作收入改填寫每月工作收入
'''

def singlePrompt(judgement_doc="", json_dict={}):
    key = next(iter(json_dict.keys()), "") if json_dict else ""
    prompt = textwrap.dedent(f"""
    [判決書]=```{judgement_doc}```
    使用 [判決書] 的內容回答 [Extraction] 的欄位
    根據給定的判決書填充JSON結構。要求如下:
    返回結果為一行JSON格式字串，無換行或特殊符號。
    [Extraction-JSON]=```{json_dict}```
    """)
            
    return prompt

def lightRuleLevel(judgement_doc="", json_dict={}):
    key = next(iter(json_dict.keys()), "") if json_dict else ""
    prompt = textwrap.dedent(f"""
    [判決書]=```{judgement_doc}```
    使用 [判決書] 的內容回答 [Extraction] 的欄位
    根據給定的判決書填充JSON結構。要求如下:
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需要包含：零件、材料、工資、鈑金、塗裝、烤漆。
    返回結果為一行JSON格式字串，無換行或特殊符號。
    [Extraction-JSON]=```{json_dict}```
    """)
            
    return prompt

def mediumRuleLevel(judgement_doc="", json_dict={}):
    key = next(iter(json_dict.keys()), "") if json_dict else ""
    prompt = textwrap.dedent(f"""
    [判決書]=```{judgement_doc}```
    使用 [判決書] 的內容回答 [Extraction] 的欄位
    根據給定的判決書填充JSON結構。要求如下:
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需要包含：零件、材料、工資、鈑金、塗裝、烤漆。
    3. 若無結果則留空白
    4. 注意每日每月之單位，一個月為30天
    5. 每日工作收入改填寫每月工作收入
    返回結果為一行JSON格式字串，無換行或特殊符號。
    [Extraction-JSON]=```{json_dict}```
    """)
    
    return prompt

def formatPrompt(judgement_doc="", json_dict={}):
    key = next(iter(json_dict.keys()), "") if json_dict else ""
    
    prompt = textwrap.dedent(f"""
    [INST]
    根據給定的判決書填充JSON結構。要求如下:
    返回結果為一行JSON格式字串，無換行或特殊符號。
    參考回覆格式'''{json_dict}'''
    [/INST]
    
    [CONTENT]
    {judgement_doc}
    [/CONTENT]
    """)
            
    return prompt

def prepare_data(data_path, type, output_path):
    
    rule_level_list = {
        'light': lightRuleLevel,
        'medium': mediumRuleLevel,
        'format': formatPrompt,
    }

    # 讀取資料
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)

    # 分割資料
    train_data = data[:split_index]
    remaining_data = data[split_index:]

    # 選擇格式化函數
    format_data_function = None
    if type == 'format_data_sio':
        format_data_function = format_data_sio
    elif type == 'format_data_text':
        format_data_function = format_data_text
    elif type == 'format_data_chat':
        format_data_function = format_data_chat
    

    for rule_level, prompt in rule_level_list.items():

        # 定義目錄
        dir_path = f"{output_path}/{rule_level}"

        # 檢查並創建目錄
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        train_data_path = os.path.join(dir_path, "train.jsonl")
        eval_data_path = os.path.join(dir_path, "eval.jsonl")

        # 在每個資料項目中加入 subject 並格式化
        train_data_with_subject = [format_data_function(prompt, item) for item in train_data if format_data_function(prompt, item) is not None]
        eval_data_with_subject = [format_data_function(prompt, item) for item in remaining_data if format_data_function(prompt, item) is not None]

        # 儲存資料
        with open(train_data_path, 'w', encoding='utf-8') as f:
            for item in train_data_with_subject:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(eval_data_path, 'w', encoding='utf-8') as f:
            for item in eval_data_with_subject:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"{rule_level} 資料分割完成！")
        print(train_data_with_subject[0])

def format_data_text(prompt, data_item):
    filtered_output = {k: data_item['output'][k] for k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in sorted(data_item['output'].keys())}
    
    formatted_text = prompt(data_item['input'], clean_output)
    return {'input': formatted_text, 'output': filtered_output}

def format_data_sio(prompt, data_item):
    filtered_output = {k: data_item['output'][k] for k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in sorted(data_item['output'].keys())}
            
    formatted_text = prompt(data_item['input'], clean_output)
    return {'subject': "", 'input': formatted_text, 'output': filtered_output}

def format_data_chat(prompt, data_item):
    filtered_output = {k: data_item['output'][k] for k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in sorted(data_item['output'].keys())}
            
    formatted_text = prompt(data_item['input'], clean_output)
    return {"messages": [{"role": "system", "content": ""}, 
                         {"role": "user", "content": formatted_text}, 
                         {"role": "assistant", "content": f"{filtered_output}"}]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--data_path', type=str, default="./finetuning_training_data_golden.jsonl", help='Path to the all data JSONL file')
    parser.add_argument('--type', type=str, default="format_data_text", help='Type of data formatting to use', choices=['format_data_text', 'format_data_sio', 'format_data_chat'])
    parser.add_argument('--output_path', type=str, default="./ccg/", help='Path to output dir')
    
    args = parser.parse_args()
    prepare_data(args.data_path, args.type, args.output_path)
    
    
    # python ./data/processed_data.py  --type format_data_text --data_path ./data/finetuning_training_data_golden.jsonl --output_path ./data/ccg/
     

