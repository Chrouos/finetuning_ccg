from openai import OpenAI
from tqdm import tqdm
import json
import os

output_eval_path = "./data/ccg/format/eval.jsonl"
client = OpenAI(api_key="sk-proj-5yX70bMfpyRVEP1yTtiUT3BlbkFJgWNkjYPkx7VCuvhZIc4z")

with open(output_eval_path, 'r', encoding='utf-8-sig') as f:
    datas = [json.loads(line) for line in f]
    
result = []
for data in tqdm(datas[100:]):
    prompt = f"{data['input']}"
    load_response = []
    generated_text = {}
    
    try:
        completion = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:widm:format-ccg:9mQjvJ26",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_text = completion.choices[0].message.content
        print("--------------------")
        print(generated_text)
        print(" ")
        print(data['output'])
        
    except Exception as e:
        print(e)
        generated_text = {}
    
    result.append({"processed": generated_text})
    
output_path = f"./data/output/openai/"
eval_file_name = f'eval.jsonl'
save_output_path = output_path + eval_file_name
os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
with open(save_output_path, 'w', encoding='utf-8-sig') as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

