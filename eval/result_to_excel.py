import pandas as pd
from glob import glob
import json
import os
import re

repeat_times = 3

# 定義字串欄位和數值欄位
string_field = ["事故日期", "事發經過", "事故車出廠日期", "傷勢", "職業", "折舊方法", "被告肇責"]
int_field = ["塗裝", "工資", "烤漆", "鈑金", "耐用年數", "修車費用", "賠償金額總額", "保險給付金額", "居家看護天數", "居家看護費用", "每日居家看護金額"]

# 定義資料夾路徑
file_path = "data/eval/"
output_file = file_path + "combined_data.xlsx"
file_paths = [f for f in glob(file_path + '*/*/*.jsonl', recursive=True) if f != output_file and 'distance' not in f]

# 合併所有檔案的內容成一個列表
combine_list = []
for path in file_paths:
    # 提取 `checkpoint-xxx`，如果有的話
    checkpoint_match = re.search(r'checkpoint-\d+', path)
    checkpoint = checkpoint_match.group(0) if checkpoint_match else ""
    
    # 提取 `folder_name`，並移除 `checkpoint-xxx`
    folder_name = re.search(r'data/eval/([^/]+)/', path)
    if folder_name:
        folder_name = folder_name.group(1)
    else:
        folder_name = path  # 如果路徑中沒有 `data/eval/`，直接取整個 path

    # 移除 `checkpoint-xxx` 的部分
    folder_name = re.sub(r'-checkpoint-\d+', '', folder_name)
        
    method_name = folder_name.split('-')[-1]
    combined_dict = {
        "Name": path.replace('data/eval/', ''), 
        "Folder": folder_name.replace('-' + method_name, ''), 
        "Method": method_name,   
        "CheckPoint": checkpoint
    }
    
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line_dict = json.loads(line)
            combined_dict.update(line_dict)
    combine_list.append(combined_dict)

# 創建 DataFrame
df = pd.DataFrame(combine_list)
df = df.groupby(['Folder', 'Method']).max(numeric_only=True).reset_index()
df['Method'] = pd.Categorical(df['Method'], categories=['basic', 'advanced', 'oneShot'], ordered=True)
df = df.sort_values(by=['Folder', 'Method']).reset_index(drop=True)


# 轉換 string_field 和 int_field 中的欄位為數值
df[string_field] = df[string_field].apply(pd.to_numeric, errors='coerce')
df[int_field] = df[int_field].apply(pd.to_numeric, errors='coerce')

# 計算平均值
df["Average(字串)"] = df[string_field].mean(axis=1)
df["Average(數值)"] = df[int_field].mean(axis=1)

# 調整 "Average(字串)" 和 "Average(數值)" 欄位的位置到第二列和第三列
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index("Average(字串)")))  # 移動到第二列
cols.insert(2, cols.pop(cols.index("Average(數值)")))  # 移動到第三列
df = df[cols]

# 名稱對應表
mapping = {
    "Llama-3.1-8B-Instruct": "L3.1-8B",
    "Llama-3.2-1B-Instruct": "L3.2-1B",
    "Llama-3.2-3B-Instruct": "L3.2-3B",
    
    "Meta-Llama-3-8B-Instruct": "L3-8B",
    
    "gpt-4o-mini": "gpt-4om",
}

# 替換 Name 欄位的值並排序
df['Folder'] = df['Folder'].replace(mapping)
order = list(mapping.values()) 
df = df.set_index('Folder').loc[order].reset_index()

df = df.round(3)

# 轉置表格
df_transposed = df.set_index('Folder').T.reset_index()
df_transposed.columns.name = None  # 移除欄位名稱

# 存入 Excel 檔案
df_transposed.to_excel(output_file, index=False)
print(df_transposed)
print(f"資料已存儲為 {output_file}")
