import pandas as pd
from glob import glob
import json
import os
import re

from utils.template_fields import get_fields
final_result_fields, template_dict, fields_setting = get_fields()

# 資料夾路徑
file_path = "data/eval/"
output_file = os.path.join(file_path, "combined_data_length.xlsx")

# 搜尋所有 length_ 開頭的 .jsonl 檔案
file_paths = [f for f in glob(file_path + '**/length_*.jsonl', recursive=True) if f != output_file]

combine_list = []

for path in file_paths:
    # 解析檔名中的 CheckPoint
    version_match = re.search(r'(checkpoint-\d+|original|ft)', path)
    version = version_match.group(0) if version_match else ""
    
    # 解析檔名中的 folder_name (model_name)
    folder_name_match = re.search(r'data/eval/([^/]+)/', path)
    folder_name = folder_name_match.group(1) if folder_name_match else path
    
    # 移除 folder_name 裡面的 version
    folder_name_stripped = re.sub(r'-(checkpoint-\d+|original|ft)', '', folder_name)
    
    # 解析 method (例如 basic, advanced, oneShot)
    method_name = folder_name_stripped.split('-')[-1]
    
    # 讀取 .jsonl 檔案
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
          
            line_dict = json.loads(line)
            bin_label = line_dict.get("bin", "")
            field_name = line_dict.get("field", "")
            score_value = line_dict.get("score", 0)
            distance_value = line_dict.get("distance", 0)
            count_value = line_dict.get("count", 0)
            
            if method_name == "format": continue
            
            # 根據需要額外記錄其他資訊
            combined_dict = {
                "Folder": folder_name_stripped.replace('-' + method_name, ''),  # 取出主體 Folder 名
                "Method": method_name,     # 方法
                "CheckPoint": version,     # checkpoint/original/ft
                "Bin": bin_label,          # 字數區間
                "Field": field_name,       # 欄位名稱
                "Score": score_value,      # 分數
                "Count": count_value,      # 筆數
                "Distance": distance_value # 距離值
                
            }
            combine_list.append(combined_dict)

# 建立成 DataFrame
df = pd.DataFrame(combine_list)

# (可選) 依需求做 groupby 彙整，例如若想要計算同 bin+field 的平均分數:
# df = df.groupby(["Folder","Method","CheckPoint","Bin","Field"], as_index=False).mean(numeric_only=True)

# 輸出到 Excel 檔案
df.to_excel(output_file, index=False)
print(f"資料已合併並存成：{output_file}")

# ------------------------------------------------------------------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 設定 Matplotlib 中文字型
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

# 假設 df 已載入並包含下列欄位：
# ["Folder", "Method", "CheckPoint", "Bin", "Field", "Score", "Distance", "Count"]
df = df.sort_values(by=["Bin"], ascending=True)

# -------------------------------
# (1) 照舊：依 Folder / CheckPoint / Field 分檔繪製「Score」折線圖 (與你原本的流程相同)
# -------------------------------
for (folder, checkpoint, field), sub_df in df.groupby(["Folder", "CheckPoint", "Field"], as_index=False):
    # 建立輸出資料夾
    output_dir = f"./data/length/{folder}/{checkpoint}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # pivot: X 軸=Bin, Y=Score, columns=Method
    pivot_df = sub_df.pivot_table(
        index="Bin", 
        columns="Method", 
        values="Score", 
        aggfunc='mean'
    )

    # 取得各 bin 的總筆數 (或視需要改成 mean)
    # 注意：要確保 sub_df 真的有 "Count" 欄位
    bin_counts = sub_df.groupby("Bin")["Count"].sum()

    # 組合新的索引標籤: "bin_label (count)"
    new_index_labels = []
    for bin_label in pivot_df.index:
        # 若 bin_counts 沒有此 bin_label，避免 KeyError，用 get() 取值
        total_count = bin_counts.get(bin_label, 0)
        new_index_labels.append(f"{bin_label} ({int(total_count)})")

    # 重新指定 pivot_df 的 row index
    pivot_df.index = new_index_labels

    # 開始繪圖
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot_df.plot(ax=ax, marker='o', linestyle='-')

    ax.set_title(f"Folder={folder}, CheckPoint={checkpoint}, Field={field}")
    ax.set_xlabel("Bin (資料筆數)")
    ax.set_ylabel("Score")
    ax.legend(title="Method")

    # 避免 X 軸標籤重疊
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 輸出
    output_file = os.path.join(output_dir, f"{field}_length_result_to_excel.png")
    plt.savefig(output_file)
    plt.close(fig)

# -------------------------------
# (2) Summary：忽略 Folder & CheckPoint，只看 Field/Bin/Method
#     同時計算 Score、Count 的合適聚合方式
# -------------------------------
summary_df = df.groupby(["Bin", "Field", "Method"], as_index=False).agg({
    "Score": "mean",  
    "Count": "sum"    # 這邊示範用 sum，代表該 (Bin, Field, Method) 的總筆數
})

summary_output_dir = "./data/length_summary/"
os.makedirs(summary_output_dir, exist_ok=True)

# -------------------------------
# (3) 逐一 Field 繪製 Summary 圖：X 軸顯示「Bin (Count)」
# -------------------------------
for field in summary_df["Field"].unique():
    # 只取當前 Field
    sub_summary_df = summary_df[summary_df["Field"] == field].copy()
    
    # pivot: X 軸 = Bin；columns = Method；values = Score
    score_pivot = sub_summary_df.pivot(index="Bin", columns="Method", values="Score")

    # 另外再計算每個 Bin 的「Count 總和」
    # 這裡視情況要 sum 還是 mean，依你需求
    bin_counts = sub_summary_df.groupby("Bin")["Count"].sum()

    # 產生新的 index 標籤 ( bin + (count) )
    # 確保兩者的 index 相同
    new_index_labels = []
    for bin_label in score_pivot.index:
        total_count = bin_counts.loc[bin_label]
        new_index_labels.append(f"{bin_label} ({int(total_count)})")

    # 改寫 pivot 的索引（因為 pivot 的 index 目前是各 bin）
    score_pivot.index = new_index_labels

    # 繪製折線圖
    fig, ax = plt.subplots(figsize=(8, 6))
    score_pivot.plot(ax=ax, marker='o', linestyle='-')

    ax.set_title(f"Summary Field={field} (Ignore Folder & CheckPoint)")
    ax.set_xlabel("Bin (資料筆數)")
    ax.set_ylabel("Score")

    # 調整 x 軸標籤，避免過度擠壓
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 儲存圖檔
    output_file = os.path.join(summary_output_dir, f"{field}_summary.png")
    plt.savefig(output_file)
    plt.close(fig)

print("All summary plots (Score + X-axis shows count) have been saved.")




# --------------------------------------------------------------------------------------------

# (A) 輸出原始 df
df_output_csv_path = os.path.join(file_path, "combined_data_length.csv")
df.to_csv(df_output_csv_path, index=False, encoding='utf-8-sig')
print(f"已輸出詳細 df: {df_output_csv_path}")

# (B) 輸出 summary_df
summary_output_csv = os.path.join(file_path, "combined_data_length_summary.csv")
summary_df.to_csv(summary_output_csv, index=False, encoding='utf-8-sig')
print(f"已輸出 summary df: {summary_output_csv}")
