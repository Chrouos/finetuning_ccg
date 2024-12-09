import json
from tqdm import tqdm
import time
import os
from GeneratorResponse import  GenerateResponseLLAMA, GenerateResponseGPT, GenerateResponseGEMINI

#: Data
instruction_data_path = f"./data/instruction/advanced/eval.jsonl"
with open(instruction_data_path, 'r', encoding='utf-8-sig') as f:
    datas = [json.loads(line) for line in f]
    
generator_response = GenerateResponseGPT( openai_key="GPT_KEY", model_name="gpt-4o-mini")

result = []
instruction = datas[0]['instruction']
auto_history = []
for data in tqdm(datas[:10]):
    judge_content = data['input']
    generated_text = ""
    load_response = []
    auto_history.append(auto_history)
    
    try:
        generated_text = generator_response.generate_text( prompt=judge_content, temperature=0.5, system_content=instruction)
        
        automated_improve_prompt = f"""
        標準答案=```{data['output']}```
        當前生成的答案=```{generated_text}```
        
        根據[標準答案]與[當前生成的答案]之間的差異，優化當前的提示詞，使語言模型能更準確地生成符合標準答案的內容。需注意以下條件：
        1. 保持語言模型在生成時對判決書內容未知的原則。
        2. 精準描述模型的任務和預期輸出格式，避免模糊或多義表述。
        3. 提供必要的背景資訊但不透露判決書的具體內容。
        4. 若沒有該擷取資料，保持空字串。
        5. 若有同一個金額代表複數個欄位，取第一個說明的欄位為主，例如：板金、烤漆為 5000 元，則類似項目標記為 "板金": 5000, "烤漆": 0。
        6. 擷取折舊前的金額：工資、鈑金、塗裝、烤漆 (其他擷取法官判決後)。
        7. 日期格式為: 年月日。
        
        需修正的提示詞=```{instruction}```
        不需要前言、後言，只要修正提示詞，而不是答案
        """
        instruction = generator_response.generate_text( prompt=judge_content, temperature=0.5, system_content=automated_improve_prompt)
        
    except Exception as e:
        print(e)
        generated_text = {}
        
    try:
        result.append({"processed": json.loads(generated_text)})
    except:
        result.append({"processed": generated_text})
        
save_output_path = f"./data/output/automated-prompt/" + f"0/" + f'generate-original.jsonl'
os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
with open(save_output_path, 'w', encoding='utf-8-sig') as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
with open("./auto_history", "w", encoding="utf-8") as file:
    for item in auto_history:
        file.write(item + "\n")
        
print()
print(instruction)


        
        