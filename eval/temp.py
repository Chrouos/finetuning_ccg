import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

def llama_generate_text(model_path, input_text, max_length=50):
    # 載入 tokenizer 和模型
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

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

# 使用示例
model_path = "./model/Llama-3.2-1B/"
input_text = "What is the capital of France?"
print(llama_generate_text(model_path, input_text))
