from GeneratorResponse import  GeneratorResponse
import json
import os
from tqdm import tqdm

GENERATE_MODE = "GPT" # GPT | GEMINI | RE |

#: Local Model Configuration
prompt_level = "advanced"                           # basic | advanced | oneShot
model_name = "gpt-4o-mini"                          # gpt-4o-mini | gemini-1.5-flash | gpt-3.5-turbo-0125
checkpoint = "original"                             # output name

#: Data
instruction_data_path = f"./data/instruction/{prompt_level}/eval.jsonl"

#: Output
output_folder_name = f"{model_name}-{prompt_level}"  # the folder name of output
output_path = f"./data/output/{output_folder_name}/"
eval_file_name = f'generate-{checkpoint}.jsonl'

generator_response = GeneratorResponse(
    openai_key="GPT_KEY",
    gemini_key="GOOGLE_KEY",
    model_path="model/", 
    fine_tuned_model_path="final_output/"
)


with open(instruction_data_path, 'r', encoding='utf-8-sig') as f:
    datas = [json.loads(line) for line in f]
    
result = []
console_output = []
for data in tqdm(datas):
    prompt = f"{data['input']}"
    load_response = []
    
    try:
        
        if GENERATE_MODE == "GPT":
            generated_text = generator_response.by_openai_generate_text(
                prompt = prompt,
                model_name = model_name
            )
            
        elif GENERATE_MODE == "GEMINI":
            generated_text = generator_response.by_gemini_generate_text(
                prompt=prompt,
                model_name=model_name,
            )
            
        elif GENERATE_MODE == "RE":
            generated_text = generator_response.by_re_generate_text(
                content_text=data['content']
            )
        
        # print(generated_text)
        
        sequence_list = f"GOL=\n{data['output']}\n-\nGEN=\n{generated_text}\n"
        print(sequence_list)
        console_output.append(sequence_list)
    except Exception as e:
        print(e)
        generated_text = {}
    
    result.append({"processed": generated_text})
    
save_output_path = output_path + eval_file_name
os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
with open(save_output_path, 'w', encoding='utf-8-sig') as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


with open('sequencelist.txt', 'w', encoding='utf-8') as file:
    for line in sequence_list:
        file.write(line + '\n')