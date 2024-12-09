import textwrap
import json
import random
import os
import argparse
import sys

from eval.utils.template_fields import get_fields
final_result_fields, template_dict, fields_setting = get_fields()

random.seed(42)

'''
    1. 以賠償給原告的數據填寫。
    2. 判決前的賠償金額需要包含：零件、材料、工資、鈑金、塗裝、烤漆。
    3. 若無結果則留空白
    4. 注意每日每月之單位，一個月為30天
    5. 每日工作收入改填寫每月工作收入
'''

def prompt_ruler(json_dict):
    rule = {
        "事故日期": "範例：原告主張被告於民國：108年10月29日18時許，無照駕駛未注意車前狀況而碰撞訴外人...\n擷取：108年10月29日",
        "事發經過": "擷取：沿國道一號高速公路之外側車道由南往北方向行駛，行經...",
        "傷勢": "範例：而當時客觀上並無不能注意之情事，竟疏未注意車前狀況，因而撞擊右方徒手牽腳踏車步行上橋之原告手手肘，致原告人車倒地（下稱系爭車禍），因而受有左腰部挫傷（瘀傷4cm×6cm）、左腹部挫傷（瘀傷5cm×5cm）、左膝擦傷（2cm×2cm）、左手肘擦傷（0.5cm×0.5cm）之傷勢（下稱系爭傷勢）\n擷取：左腰部挫傷（瘀傷4cm×6cm）、左腹部挫傷（瘀傷5cm×5cm）、左膝擦傷（2cm×2cm）、左手肘擦傷（0.5cm×0.5cm）之傷勢（下稱系爭傷勢）",
        "職業": "範例：原告於系爭車禍發生前從事回收工作\n擷取：回收",
        "精神賠償": "範例：爰斟酌被告加害之情形，雙方過失比例、原告所受精神痛苦程度，及兩造之身分、職業、社會地位、經濟狀況等一切情狀，認原告請求原告賠償精神慰撫金10萬元為適當\n擷取：100000",
        "醫療費用": "範例：醫療費用11,774元\n擷取：11774",
        "每日居家看護金額": "範例：原告主張由家人看護66日共132,000元乙情，固有提出診斷證明書為憑。\n擷取：2000",
        "居家看護天數": "範例：原告主張由家人看護66日共132,000元乙情，固有提出診斷證明書為憑。\n擷取：66",
        "居家看護費用": "範例：原告主張由家人看護66日共132,000元乙情，固有提出診斷證明書為憑。\n擷取：132000",
        "每日住院看護金額": "範例：原告於107年11月15日至同年月26日住院治療期間，支出看護費用28,800元部分，有診斷證明書可憑\n擷取：2400",
        "住院看護天數": "範例：原告於107年11月15日至同年月26日住院治療期間，支出看護費用28,800元部分，有診斷證明書可憑\n擷取：12",
        "住院看護費用": "範例：原告於107年11月15日至同年月26日住院治療期間，支出看護費用28,800元部分，有診斷證明書可憑\n擷取：28800",
        "看護總額": "範例：應以每日1,600元計算為適當。是以原告請求看護費用於144,000元（1600×30×3）範圍，應予准許，\n擷取：144000",
        "每日營業收入": "範例：查原告主張系爭汽車修理須3日，致其另受有營業損失4,500元等語，已提出記載系爭汽車所須維修工作日為3日之估價單\n擷取：1500",
        "營業損失天數": "範例：查原告主張系爭汽車修理須3日，致其另受有營業損失4,500元等語，已提出記載系爭汽車所須維修工作日為3日之估價單\n擷取：3",
        "營業損失": "範例：查原告主張系爭汽車修理須3日，致其另受有營業損失4,500元等語，已提出記載系爭汽車所須維修工作日為3日之估價單\n擷取：4500",
        "每日工作收入": "範例：原告自107年10月26日起至108年6月17日止，以依兩造不爭執之原告每月薪資34,000元（見不爭執事項㈢）計算，其請求無法工作損失264,067元【計算式：34,000元×（7＋23/30）＝264,067元，元以下四捨五入（下同）】部分，核屬有據\n擷取：34000",
        "工作損失天數": "範例：原告自107年10月26日起至108年6月17日止，以依兩造不爭執之原告每月薪資34,000元（見不爭執事項㈢）計算，其請求無法工作損失264,067元【計算式：34,000元×（7＋23/30）＝264,067元，元以下四捨五入（下同）】部分，核屬有據\n擷取：233",
        "工作損失": "範例：原告自107年10月26日起至108年6月17日止，以依兩造不爭執之原告每月薪資34,000元（見不爭執事項㈢）計算，其請求無法工作損失264,067元【計算式：34,000元×（7＋23/30）＝264,067元，元以下四捨五入（下同）】部分，核屬有據\n擷取：264067",
        "事故車出廠日期": "範例：系爭汽車係96年5月出廠\n擷取：96年5月",
        "折舊方法": "範例：行政院所頒固定資產耐用年數表及固定資產折舊率表，自用小客車之耐用年數為5年，依平均法計算其折舊結果（即以固定資產成本減除殘價後之餘額，按固定資產耐用年數表規定之耐用年數平均分攤，計算折舊額），每年折舊率為5分之1\n擷取：平均法",
        "耐用年數": "範例：行政院所頒固定資產耐用年數表及固定資產折舊率表，自用小客車之耐用年數為5年，依平均法計算其折舊結果（即以固定資產成本減除殘價後之餘額，按固定資產耐用年數表規定之耐用年數平均分攤，計算折舊額），每年折舊率為5分之1\n擷取：5",
        "零件": "範例：零件費用3,500元\n擷取：3500",
        "材料": "範例：材料92,090元\n擷取：92090",
        "工資": "範例：工資6,000元\n擷取：6000",
        "板金": "範例：鈑金27,000元\n擷取：27000",
        "塗裝": "範例：塗裝2萬元\n擷取：20000",
        "烤漆": "範例：烤漆費用5,500元\n擷取：5500",
        "修車費用": "範例：故原告請求系爭車輛所支出之修復費用以63,734元為必要\n擷取：63734",
        "交通費用": "範例：交通費用1,105元\n擷取：1105",
        "財產損失": "範例：原告所有之手機因本件事故而受損，經估計維修費用為9,500元（含零件及工資費用），有維修報價單在卷可參（交附民卷第7頁）。\n擷取：9500",
        "賠償金額總額": "範例：請求被告給付原告16,583元（計算式：車損修繕費用12,083元＋營業損失4,500元＝16,583元）\n擷取：16583",
        "被告肇責": "範例：認本件事故被告應負70%之過失責任\n擷取：70",
        "保險給付金額": "範例：查原告已依強制汽車責任保險法第40條規定受領汽車交通事故特別補償金9,000元，故依該筆補償金視為損害賠償金額之一部分，應予以扣除。\n擷取：9000"
    }
    
    result_rule = {}
    for k in json_dict:
        if k in rule.keys():
            result_rule[k] = rule[k]

    return str(result_rule)

def basicPrompt(judgement_doc="", json_dict={}):
    clean_output = {k: "" for k in json_dict}
    
    prompt = textwrap.dedent(f"""
    [Extraction-JSON]=\n```\n{clean_output}\n```\n
    
    根據給定的判決書填充 [Extraction-JSON] 結構，指導方針：
    1. 擷取折舊前的金額：工資、鈑金、塗裝、烤漆 (其他擷取法官判決後)。
    2. 日期格式為: 民國年月日。
    你是文本擷取專家, 回傳繁體中文, 要擷取文本原文, 不要修改內容, 返回結果為一行JSON格式的"字串", 無換行或特殊符號!
    """)
            
    return prompt

def advancedPrompt(judgement_doc="", json_dict={}):
    
    clean_output = {k: "" for k in json_dict}
    
    prompt = textwrap.dedent(f"""
    [Extraction-JSON]=\n```\n{clean_output}\n```\n

    根據給定的判決書填充 [Extraction-JSON] 結構，指導方針：
    1. 擷取折舊前的金額：工資、鈑金、塗裝、烤漆 (其他擷取法官判決後)。
    2. 日期格式為: 年月日。
    3. 在"折舊方法"欄位必為"定率遞減法"或"平均法"。
    4. "被告肇責": 0~100。
    5. "修車費用": 法官判決後的修車費用加總。
    6. 賠償金額總額填入法官最終判決給被告的金額，通常為所有數值的加總。
    7. 若有同一個金額代表複數個欄位，取第一個說明的欄位為主，例如：板金、烤漆為 5000 元，則類似項目標記為 "板金": 5000, "烤漆": 0。
    8. 若沒有該擷取資料，保持空字串。

    備註：你是文本擷取專家, 回傳繁體中文, 要擷取文本原文, 不要修改內容, 返回結果為一行JSON格式的“字串”, 無換行或特殊符號!
    """)
    
    return prompt

def oneShotPrompt(judgement_doc="", json_dict={}):
    clean_output = {k: "" for k in json_dict}
    
    prompt = textwrap.dedent(f"""
    [Extraction-JSON]=\n```\n{clean_output}\n```\n
    
    根據給定的判決書填充 [Extraction-JSON] 結構。要求如 [EXAMPLE]
    你是文本擷取專家, 回傳繁體中文, 要擷取文本原文, 不要修改內容, 返回結果為一行JSON格式的“字串”, 無換行或特殊符號!
    
    [EXAMPLE]=\n```\n{prompt_ruler(json_dict)}\n```\n
    """)
    
    return prompt

def automatedPrompt(judgement_doc="", json_dict={}):
    clean_output = {k: "" for k in json_dict}
    
    prompt = textwrap.dedent(f"""
    {{
    "事故日期": "請提供事故發生的日期，格式為：年月日。",
    "事發經過": "請詳細描述事故的經過，包括涉及的車輛、駕駛者及事故原因，確保描述中包含因果關係，並避免透露具體判決書內容。",
    "事故車出廠日期": "請提供事故車輛的出廠日期，格式為：年月日，若無則保持空字串。",
    "傷勢": "如有傷勢，請描述受傷情況，若無則保持空字串。",
    "職業": "請提供事故相關的職業，若無則保持空字串。",
    "折舊方法": "請提供折舊計算方法，若無則保持空字串。",
    "被告肇責": "請描述被告在事故中的過失責任，若無則保持空字串。",
    "塗裝": "如有塗裝費用，請提供金額，若無則保持空字串。",
    "工資": "請提供工資相關費用的金額，若無則保持空字串。",
    "烤漆": "如有烤漆費用，請提供金額，若無則保持空字串。",
    "鈑金": "如有鈑金費用，請提供金額，若無則保持空字串。",
    "耐用年數": "請提供事故車輛的耐用年數，若無則保持空字串。",
    "修車費用": "請提供修車的費用金額，若無則保持空字串。",
    "賠償金額總額": "請提供賠償的總金額，若無則保持空字串。",
    "保險給付金額": "如有保險給付金額，請提供，若無則保持空字串。",
    "居家看護天數": "如有居家看護天數，請提供，若無則保持空字串。",
    "居家看護費用": "如有居家看護費用，請提供金額，若無則保持空字串。",
    "每日居家看護金額": "如有每日居家看護金額，請提供，若無則保持空字串。"
    }}
    """)
    
    return prompt

def formatPrompt(judgement_doc="", json_dict={}):
    
    prompt = textwrap.dedent(f"""
    [INST]
    指導方針：
    1. 事故日期為車禍當日的年月日。
    2. 擷取折舊前的金額：工資、鈑金、塗裝、烤漆 (其他擷取法官判決後)。
    3. 日期格式為: 年月日。
    4. "事發經過"欄位, 擷取包含原告所駕車輛, 地點, 事故情況及結果(如車輛損傷等)的資訊。
    5. 在"折舊方法"欄位為"定率遞減法"或"平均法"。
    6. "被告肇責"為 0~100。
    7. 賠償金額總額填入法官最終判決給被告的金額，通常為所有數值的加總。
    
    你是文本擷取專家, 要擷取文本原文, 不要修改內容, 返回結果為一行JSON格式的“字串”, 無換行或特殊符號!
    [/INST]
    
    [CONTENT]{judgement_doc}[/CONTENT]
    [OUTPUT]{json_dict}[/OUTPUT]
    """)
            
    return prompt

def prepare_data(data_path, type, output_path):
    
    rule_level_list = {
        'basic': basicPrompt,
        'advanced': advancedPrompt,
        'oneShot': oneShotPrompt,
        'format': formatPrompt,
        'automatedPrompt': automatedPrompt
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

import re
def clean_text(text):
    cleaned_text = re.sub(r'[\t\n\"]', '', text)
    return cleaned_text

def format_data_text(prompt, data_item):
    filtered_output = {k: clean_text(data_item['output'][k]) for k in final_result_fields if k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in final_result_fields}
    
    formatted_text = prompt(data_item['input'], filtered_output)
    return {'input': clean_text(formatted_text), 'output': filtered_output, 'content': data_item['input']}

def format_data_sio(prompt, data_item):
    filtered_output = {k: clean_text(data_item['output'][k]) for k in final_result_fields if k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in final_result_fields}
            
    formatted_text = prompt(data_item['input'], clean_output)
    return {'instruction': clean_text(formatted_text), 'input': data_item['input'], 'output': filtered_output}

def format_data_chat(prompt, data_item):
    filtered_output = {k: clean_text(data_item['output'][k]) for k in final_result_fields if k in sorted(data_item['output'].keys())}
    clean_output = {k: "" for k in final_result_fields}
            
    formatted_text = prompt(data_item['input'], clean_output)
    # {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
    return {
        "messages": [
            # {"role": "system", "content": ""}, 
            {"role": "user", "content": clean_text(formatted_text)}, 
            {"role": "assistant", "content": f"{filtered_output}"}
        ],
        # 'content': data_item['input']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--data_path', type=str, default="./sort_finetuning_training_data_golden.jsonl", help='Path to the all data JSONL file')
    parser.add_argument('--type', type=str, default="format_data_text", help='Type of data formatting to use', choices=['format_data_text', 'format_data_sio', 'format_data_chat'])
    parser.add_argument('--output_path', type=str, default="./instruction/", help='Path to output dir')
    
    args = parser.parse_args()
    prepare_data(args.data_path, args.type, args.output_path)
    
    # python ./processed_to_format.py  --type format_data_text --data_path ./data/sort_finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    

