from GeneratorResponse import  GenerateResponseLLAMA, GenerateResponseGPT, GenerateResponseGEMINI
import json
import os
from tqdm import tqdm
import sys

generator_response = None # GeneratorResponse
GENERATE_MODE = "FineTuning" # GPT | GEMINI | RE | LLAMA | FineTuning

#: Local Model Configuration
prompt_level = "oneShot"                          
#=> basic | advanced | oneShot | format

model_name = "Meta-Llama-3-8B-Instruct"
#=> re
#=> gpt-4o-mini | gemini-1.5-flash | gpt-3.5-turbo-0125 | 
#=> Llama-3.1-8B | Meta-Llama-3-8B-Instruct

checkpoint = "checkpoint-900"                             
#=> original | checkpoint

#: Data
instruction_data_path = f"./data/instruction/{prompt_level}/eval.jsonl"
with open(instruction_data_path, 'r', encoding='utf-8-sig') as f:
    datas = [json.loads(line) for line in f]

#: Output
output_folder_name = f"{model_name}-{prompt_level}-{checkpoint}"  # the folder name of output
output_path = f"./data/output/{output_folder_name}/"
eval_file_name = f'generate-{checkpoint}.jsonl'

#: Generator
if GENERATE_MODE == "LLAMA": 
    generator_response = GenerateResponseLLAMA(
        model_folder="model/", fine_tuned_model_path="final_output/", model_name=model_name
    )
elif GENERATE_MODE == "FineTuning":
    generator_response = GenerateResponseLLAMA(
        model_folder="model/", fine_tuned_model_path="final_output/", model_name=model_name,
        fine_tune=True, check_point=checkpoint
    )
elif GENERATE_MODE == "GPT": 
    generator_response = GenerateResponseGPT(
        openai_key="GPT_KEY", model_name=model_name
    )
elif GENERATE_MODE == "GEMINI": 
    generator_response = GenerateResponseGEMINI(
        gemini_key="GOOGLE_KEY"
    )
else:
    print(f"GENERATE_MODE={GENERATE_MODE}, 對應不上！")
    sys.exit(0)
    
print(f"成功讀取 GenerateResponse: {GENERATE_MODE}")

repeat_times = 3
console_output = []
for time in range(repeat_times):
    result = []
    
    for data in tqdm(datas, desc=str(time)):
        prompt = f"{data['input']}"
        generated_text = ""
        load_response = []
        
        try:
            if GENERATE_MODE == "LLAMA":
                generated_text = generator_response.generate_text( prompt=prompt, temperature=0.5, max_new_tokens=512 )
            elif GENERATE_MODE == "FineTuning":
                generated_text = generator_response.generate_text( prompt=prompt, temperature=0.5, max_new_tokens=512 )
            elif GENERATE_MODE == "GPT":
                generated_text = generator_response.generate_text( prompt=prompt, temperature=0.5)
                
            result_sequence = f"--\nGOL=\n{data['output']}\nGEN=\n{generated_text}\n--\n"
            # print(result_sequence)
            console_output.append(result_sequence)
        except Exception as e:
            print(e)
            generated_text = {}
        
        result.append({"processed": generated_text})
        
    save_output_path = output_path + f"{time}/" + eval_file_name
    os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
    with open(save_output_path, 'w', encoding='utf-8-sig') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('sequencelist.txt', 'w', encoding='utf-8') as file:
    for line in console_output:
        file.write(line + '\n')