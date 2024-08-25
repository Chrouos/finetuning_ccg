import pandas as pd
from glob import glob
import json
import os

string_field = ["事故日期", "事發經過", "事故車出廠日期", "傷勢", "職業", "折舊方法", "被告肇責",]
int_field = ["塗裝", "工資", "烤漆", "鈑金", "耐用年數", "修車費用", "醫療費用", "賠償金額總額", "保險給付金額", "居家看護天數", "居家看護費用", "每日居家看護金額",]

# 定義資料夾路徑
file_path = "data/eval/"
output_file = file_path + "combined_data.xlsx"
file_paths = [f for f in glob(file_path + '**/*.jsonl', recursive=True) if f != output_file]

# 合併所有檔案的內容成一個列表
combine_list = []
for path in file_paths:
    combined_dict = {"Name": path}
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line_dict = json.loads(line)
            combined_dict.update(line_dict)
    combine_list.append(combined_dict)

# 創建 DataFrame
df = pd.DataFrame(combine_list)

df[string_field] = df[string_field].apply(pd.to_numeric, errors='coerce')
df[int_field] = df[int_field].apply(pd.to_numeric, errors='coerce')

df["Average(字串)"] = df[string_field].mean(axis=1)
df["Average(數值)"] = df[int_field].mean(axis=1)

df = df[[col for col in df.columns if col not in ["Average(字串)", "Average(數值)"]] + ["Average(字串)", "Average(數值)"]]

# 存入 Excel 檔案
df.to_excel(output_file, index=False)

print(f"資料已存儲為 {output_file}")
