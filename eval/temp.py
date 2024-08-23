from generate import GeneratorResponse


#: Configuration
model_name = "Meta-Llama-3.1-8B"       # select the base model
finetuning_model_name = "meta-llama3.1-format-medium"  # the folder name of output
checkpoint = "original"                         # output name

#: Data
instruction_data_path = "./data/ccg/format/eval.jsonl"

#: Output
output_path = f"./data/output/{finetuning_model_name}/"
eval_file_name = f'generate-{checkpoint}.jsonl'

#- openai - fine-tuning
# generator_response = GeneratorResponse(
#     openai_key="GPT_Finetuning_KEY",
#     model_path="model/", 
#     fine_tuned_model_path="final_output/"
# )

#- local - model
# generator_response.change_local_model(
#     model_name="Meta-Llama-3.1-8B", 
#     finetuning_model_name="meta-llama3.1-format-medium", 
#     checkpoint="original", 
#     is_use_base_model=False
# )


