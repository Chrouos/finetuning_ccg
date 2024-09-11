from glob import glob
from tqdm import tqdm
import json

# 修改 template_key 為列表
template_key = [
    "事故日期", "事發經過", "事故車出廠日期", "傷勢", "職業", "折舊方法", 
    "被告肇責", "塗裝", "工資", "烤漆", "鈑金", "耐用年數", "修車費用", 
    "醫療費用", "賠償金額總額", "保險給付金額", "居家看護天數", "居家看護費用", 
    "每日居家看護金額", "精神賠償", "每日住院看護金額", "住院看護天數", 
    "住院看護費用", "看護總額", "每日營業收入", "營業損失天數", "營業損失", 
    "每日工作收入", "工作損失天數", "工作損失", "零件", "材料", "交通費用", 
    "財產損失", "其他", "備註"
]

glob_processed_path_list = sorted(glob("./data/processed/Z_112522104/*"))

# - 讀取 JSON 檔案
def load_json_data(open_file_path):
    json_data = []
    with open(open_file_path, 'r', encoding="utf-8") as file:
        for line in file:    
            json_data.append(json.loads(line))
    return json_data

print("Processed Files:")
extraction_golden_list = []
for processed_path in glob_processed_path_list:
    print(processed_path)
    
    content_list = [item["cleanJudgement"] for item in load_json_data(processed_path.replace("processed/Z_112522104", "labeler/Judegement"))]
    extraction_list = [item['processed'] for item in load_json_data(processed_path)]
    
    for index, extraction_answer in enumerate(extraction_list):
        current_list = {"input": content_list[index], "output": {}}
        for processed_data_key, processed_data_value in extraction_answer.items():
            if processed_data_key not in ["其他"] and processed_data_key in template_key:
                current_list["output"][processed_data_key] = processed_data_value
                
        # 確保輸出按照 template_key 的順序排列
        sorted_output = {k: current_list["output"].get(k, "") for k in template_key}
        current_list["output"] = sorted_output
            
        extraction_golden_list.append(current_list)


training_data_save_path = "./data/sort_finetuning_training_data_golden.jsonl"
with open(training_data_save_path, 'w', encoding='utf-8') as f:
    for response in extraction_golden_list:
        json_record = response
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")