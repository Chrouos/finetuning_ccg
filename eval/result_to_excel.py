import pandas as pd
from glob import glob
import json
import os
import re

# 定義欄位
string_field = ["事故日期", "事發經過", "事故車出廠日期", "傷勢", "職業", "折舊方法", "被告肇責"]
int_field = ["塗裝", "工資", "烤漆", "鈑金", "耐用年數", "修車費用", "賠償金額總額", "保險給付金額", "居家看護天數", "居家看護費用", "每日居家看護金額"]

# 資料夾路徑
file_path = "data/eval/"
output_file = file_path + "combined_data.xlsx"
file_paths = [f for f in glob(file_path + '*/*/*.jsonl', recursive=True) if f != output_file and 'distance' not in f]

# 合併檔案內容
combine_list = []
for path in file_paths:
    # 提取版本 (checkpoint-xxx, original, ft)
    version_match = re.search(r'(checkpoint-\d+|original|ft)', path)
    version = version_match.group(0) if version_match else ""

    # 提取 `folder_name` 並移除版本
    folder_name_match = re.search(r'data/eval/([^/]+)/', path)
    folder_name = folder_name_match.group(1) if folder_name_match else path
    folder_name = re.sub(r'-(checkpoint-\d+|original|ft)', '', folder_name)

    method_name = folder_name.split('-')[-1]
    combined_dict = {
        "Name": path.replace('data/eval/', ''),
        "Folder": folder_name.replace('-' + method_name, ''),
        "Method": method_name,
        "CheckPoint": version
    }

    # 讀取 JSONL 檔案內容
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line_dict = json.loads(line)
            combined_dict.update(line_dict)
    combine_list.append(combined_dict)

# 建立 DataFrame

df = pd.DataFrame(combine_list)
# 設定 Method 欄位排序
df['Method'] = pd.Categorical(df['Method'], categories=['basic', 'advanced', 'oneShot', 'format'], ordered=True)
df = df.groupby(['Folder', 'CheckPoint', 'Method'], observed=True).max(numeric_only=True).reset_index()
df = df.dropna(subset=['事故日期'])  # 根據需要，選擇一個非空欄位來過濾
df = df.sort_values(by=['CheckPoint', 'Folder', 'Method']).reset_index(drop=True)

# 轉換 string_field 和 int_field 中的欄位為數值
df[string_field] = df[string_field].apply(pd.to_numeric, errors='coerce')
df[int_field] = df[int_field].apply(pd.to_numeric, errors='coerce')

# 計算平均值
df["Average(字串)"] = df[string_field].mean(axis=1)
df["Average(數值)"] = df[int_field].mean(axis=1)

# 調整 "Average(字串)" 和 "Average(數值)" 欄位的位置
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index("Average(字串)")))
cols.insert(2, cols.pop(cols.index("Average(數值)")))
df = df[cols]


# 名稱對應表
mapping = {
    # "gpt-4o-mini": "gpt-4om",
    # "gemini-1.5-flash": "Gemini",
    
    "Llama-3.1-8B-Instruct": "L3.1-8B",
    # "Llama-3.2-3B-Instruct": "L3.2-3B",
    # "Meta-Llama-3-8B-Instruct": "L3-8B",
    
    # "Llama-3-Taiwan-8B-Instruct": "L3-8B-Taiwan",
    
    # "golden": "golden"
}
df['Folder'] = df['Folder'].replace(mapping)
order = list(mapping.values())
df = df.set_index('Folder').loc[order].reset_index()

# 四捨五入至小數點後三位
df = df.round(3)

# 轉置表格
df_transposed = df.set_index('Folder').T.reset_index()
df_transposed.columns.name = None

# 存儲為 Excel 檔案
df_transposed.to_excel(output_file, index=False)
print(df_transposed)
print(f"資料已存儲為 {output_file}")
