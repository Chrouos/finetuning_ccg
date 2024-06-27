import textwrap
import json
import random
import os
import argparse

def prepare_data(all_data_path, type):
    # 定義不同級別的提示
    light_prompt = textwrap.dedent("""
    使用判決書填充JSON結構。要求如下:
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需要包含：零件、材料、塗裝、烤漆。
    返回結果為一行JSON格式字串，無換行或特殊符號。
    """)

    medium_prompt = textwrap.dedent("""
    根據給定的判決文檔填充JSON結構。要求如下:
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需包含：零件、材料、工資、鈑金、塗裝、烤漆。
    3. 若無結果則留空白，如 {"工資": ""}
    4. 注意每日每月之單位，一個月為30天
    5. 每日工作收入改填寫每月工作收入
    返回結果為一行JSON格式字串，無換行或特殊符號。
    """)

    rule_level_list = {
        'light': light_prompt,
        'medium': medium_prompt,
    }

    # 讀取資料
    with open(all_data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)

    # 分割資料
    train_data = data[:split_index]
    remaining_data = data[split_index:]
    val_test_split_index = int(len(remaining_data) * 0.5)

    val_data = remaining_data[:val_test_split_index]
    test_data = remaining_data[val_test_split_index:]

    # 選擇格式化函數
    format_data_function = format_data_sio if type == 'format_data_sio' else format_data_text

    for rule_level, prompt in rule_level_list.items():

        # 定義目錄
        dir_path = f"./ccg/{rule_level}"

        # 檢查並創建目錄
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        train_data_path = os.path.join(dir_path, "train.jsonl")
        val_data_path = os.path.join(dir_path, "valid.jsonl")
        test_data_path = os.path.join(dir_path, "test.jsonl")

        # 在每個資料項目中加入 subject 並格式化
        train_data_with_subject = [format_data_function(prompt, item) for item in train_data]
        val_data_with_subject = [format_data_function(prompt, item) for item in val_data]
        test_data_with_subject = [format_data_function(prompt, item) for item in test_data]

        # 儲存資料
        with open(train_data_path, 'w', encoding='utf-8') as f:
            for item in train_data_with_subject:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(val_data_path, 'w', encoding='utf-8') as f:
            for item in val_data_with_subject:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(test_data_path, 'w', encoding='utf-8') as f:
            for item in test_data_with_subject:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"{rule_level} 資料分割完成！")

def format_data_text(prompt, data_item):
    formatted_text = f"<s>[INST]{prompt}[/INST] [CONTENT]{data_item['input']}[/CONTENT] {data_item['output']}</s>"
    return {'text': formatted_text}

def format_data_sio(prompt, data_item):
    return {'subject': prompt, 'input': data_item['input'], 'output': data_item['output']}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--all_data_path', type=str, default="./finetuning_training_data_golden.jsonl", help='Path to the all data JSONL file')
    parser.add_argument('--type', type=str, default="format_data_sio", help='Type of data formatting to use', choices=['format_data_text', 'format_data_sio'])
    
    args = parser.parse_args()
    prepare_data(args.all_data_path, args.type)
