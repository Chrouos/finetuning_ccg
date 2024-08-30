import pandas as pd
from glob import glob
import json
import os

# 定義字串欄位和數值欄位
string_field = ["事故日期", "事發經過", "事故車出廠日期", "傷勢", "職業", "折舊方法", "被告肇責"]
int_field = ["塗裝", "工資", "烤漆", "鈑金", "耐用年數", "修車費用", "賠償金額總額", "保險給付金額", "居家看護天數", "居家看護費用", "每日居家看護金額"]

# 定義資料夾路徑
file_path = "data/eval/"
output_file = file_path + "combined_data.xlsx"
file_paths = [f for f in glob(file_path + '**/*.jsonl', recursive=True) if f != output_file]

# 合併所有檔案的內容成一個列表
combine_list = []
for path in file_paths:
    combined_dict = {"Name": path.replace('data/eval/', '')}
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line_dict = json.loads(line)
            combined_dict.update(line_dict)
    combine_list.append(combined_dict)

# 創建 DataFrame
df = pd.DataFrame(combine_list)

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
    "re-format/processed_generate-original.jsonl": "RE",
    "gpt-4o-mini-basic/processed_generate-original.jsonl": "GPT-basic",
    "gpt-4o-mini-advanced/processed_generate-original.jsonl": "GPT-advanced",
    "gpt-4o-mini-oneShot/processed_generate-original.jsonl": "GPT-oneShot",
    "gemini-1.5-flash-basic/processed_generate-original.jsonl": "GEMINI_basic",
    "gemini-1.5-flash-advanced/processed_generate-original.jsonl": "GEMINI-advanced",
    "gemini-1.5-flash-oneShot/processed_generate-original.jsonl": "GEMINI-oneShot",
    "meta-chinese-format-advanced/processed_generate-original.jsonl": "Chinese-LLama",
    "meta-chinese-format-advanced/processed_generate-checkpoint-600.jsonl": "Chinese-Llama-finetuning",
    "meta-llama-format-instruct-advanced/processed_generate-original.jsonl": "Instruct-LLama",
    "meta-llama-format-instruct-advanced/processed_generate-checkpoint-600.jsonl": "Instruct-LLama-finetuning",
    "ft-gpt-4o-mini-2024-07-18-advanced/processed_generate-original.jsonl": "GPT-finetuning"
}

# 替換 Name 欄位的值並排序
df['Name'] = df['Name'].replace(mapping)
order = mapping.values()
df = df.set_index('Name').loc[order].reset_index()

# 轉置表格
df_transposed = df.set_index('Name').T.reset_index()
df_transposed.columns.name = None  # 移除欄位名稱

# 存入 Excel 檔案
df_transposed.to_excel(output_file, index=False)

print(f"資料已存儲為 {output_file}")
