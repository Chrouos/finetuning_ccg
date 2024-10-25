from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel

def initialize_bnb_config(use_4bit=True, quant_type="nf4", compute_dtype="float16", use_nested_quant=False):
    compute_dtype = getattr(torch, compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

def load_model(model_name, device_map, bnb_config):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    return base_model

def merge_models(base_model, fine_tuned_model_path):
    model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
    return model.merge_and_unload()

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=8000):
    input_length = len(tokenizer(prompt)['input_ids'])
    max_model_length = 8192  # 假設模型的最大長度限制為 8192 tokens

    # 計算可用於生成的最大 tokens 數
    max_tokens = max_model_length - input_length

    if max_tokens <= 0:
        print("輸入長度已超過模型限制，請縮短輸入。")
        return ""
    
    # 確保不超過用戶指定的 max_new_tokens
    max_tokens = min(max_tokens, max_new_tokens)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    result = pipe(prompt, max_new_tokens=max_tokens, truncation=True)
    generated_text = result[0]['generated_text']
    
    # 去除 prompt 部分的文本
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    
    return generated_text

def llama_generate_text(model_path, input_text, max_length=50):
    # 載入 tokenizer 和模型
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    # model = LlamaForCausalLM.from_pretrained(model_path)

    # 如果有 GPU 可用，將模型轉移到 GPU 上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 將輸入文本轉換為模型可用的格式
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 產生模型輸出
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # 解碼輸出並返回結果
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output