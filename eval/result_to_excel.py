import pandas as pd
from glob import glob
import json
import os

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

# 存入 Excel 檔案
df.to_excel(output_file, index=False)

print(f"資料已存儲為 {output_file}")
