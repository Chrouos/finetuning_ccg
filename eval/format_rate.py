import json
import os
from glob import glob
import pandas as pd
import os

#: Args.
repeat_times = 1
finetuning_model_name_list = [
    
    "golden-format-original",
    
    #: GPT-4o-mini
    "gpt-4o-mini-basic-original",
    "gpt-4o-mini-advanced-original",
    "gpt-4o-mini-oneShot-original",
    "gpt-4o-mini-automatedPrompt-original",
    
    "gpt-4o-mini-advanced-ft",

    
    #: LLama-3-8B
    "Meta-Llama-3-8B-Instruct-basic-original",
    "Meta-Llama-3-8B-Instruct-advanced-original",
    "Meta-Llama-3-8B-Instruct-oneShot-original",
    
    "Meta-Llama-3-8B-Instruct-basic-checkpoint-900",
    "Meta-Llama-3-8B-Instruct-advanced-checkpoint-900",
    "Meta-Llama-3-8B-Instruct-oneShot-checkpoint-900",
    
    #: Taiwan LLAMA 8B
    "Llama-3-Taiwan-8B-Instruct-basic-original",
    "Llama-3-Taiwan-8B-Instruct-advanced-original",
    "Llama-3-Taiwan-8B-Instruct-oneShot-original",

    "Llama-3-Taiwan-8B-Instruct-basic-checkpoint-900",
    "Llama-3-Taiwan-8B-Instruct-advanced-checkpoint-900",
    "Llama-3-Taiwan-8B-Instruct-oneShot-checkpoint-900",
]


results = []
for finetuning_model_name in finetuning_model_name_list:
    for time in range(repeat_times):
        pre_output_path = f"./data/output/{finetuning_model_name}/{time}/"
        file_paths = [f for f in glob(pre_output_path + '*', recursive=True) if 'processed_' in f]

        for file_path in file_paths:
            current_file_name = os.path.basename(file_path)

            with open(file_path, 'r', encoding='utf-8-sig') as f:
                eval_datas = [json.loads(line) for line in f]

            # Calculate accuracy ratio
            total_count = len(eval_datas)
            correct_count = sum(1 for eval_data in eval_datas if eval_data['processed'] != {})
            
            accuracy_ratio = correct_count / total_count if total_count > 0 else 0

            # Append to results
            results.append({
                "finetuning_model_name": finetuning_model_name,
                "time": time,
                "file_name": current_file_name,
                "correct_count": correct_count,
                "total_count": total_count,
                "accuracy_ratio": accuracy_ratio
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

# Save to CSV
output_csv_path = "./data/eval/empty_processed_ratios.csv"
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(df)
print(f"Results saved to {output_csv_path}")
