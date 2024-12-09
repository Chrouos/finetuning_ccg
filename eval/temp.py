from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 路徑設置
base_model_path = "./model/Llama-3.1-8B-Instruct/"  # 未微調模型路徑
adapter_path = "./final_output/Llama-3.1-8B-Instruct/checkpoint-900/"  # 微調適配器路徑

# 加載基礎模型
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 應用適配器（需要 Transformers 支援 LoRA 適配器）
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)
print("模型配置信息:", model.config)
print("適配器配置信息:", model.peft_config)

# 加載分詞器
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 測試生成
prompt = "測試指令"
inputs = tokenizer(prompt, return_tensors="pt")

# 測試未加載適配器的模型輸出
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
base_outputs = base_model.generate(**inputs, max_new_tokens=50)
print("未微調模型輸出:", tokenizer.decode(base_outputs[0]))

# 測試加載適配器的模型輸出
adapter_outputs = model.generate(**inputs, max_new_tokens=50)
print("加載適配器的模型輸出:", tokenizer.decode(adapter_outputs[0]))
