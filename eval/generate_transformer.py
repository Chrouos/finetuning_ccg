import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

#: Configuration
finetuning_model_name = "meta-llama-transformer"
checkpoint = "transformer"

#: Output
output_path = f"./data/output/{finetuning_model_name}/"
output_eval_path = "./data/ccg/format/eval.jsonl"
eval_file_name = f'eval-{checkpoint}.jsonl'

fine_tuned_model_path = f"./final_output/{finetuning_model_name}/{checkpoint}/"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

# # ------------------------------------------------------------------------------------------------------------------------

# from utils.operator_data import text_splitter_RecursiveCharacterTextSplitter
# import json
# import os
# from tqdm import tqdm

# with open(output_eval_path, 'r', encoding='utf-8-sig') as f:
#     datas = [json.loads(line) for line in f]

# result = []
# for data in tqdm(datas):
#     prompt = f" {data['input']}"
#     load_response = []
    
#     try:
#         generated_text = pipeline(prompt)
#         print(generated_text)
#     except Exception as e:
#         print(e)
#         generated_text = ""
    
#     result.append({"processed": generated_text})
    
# save_output_path = output_path + eval_file_name
# os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
# with open(save_output_path, 'w', encoding='utf-8-sig') as f:
#     for item in result:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')