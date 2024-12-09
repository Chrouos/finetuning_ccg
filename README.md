# Installation


## mlx-example

```
# https
git clone https://github.com/ml-explore/mlx-examples.git

# ssh
gt clone git@github.com:ml-explore/mlx-examples.git
```

## Data

```
# shenzhi-wang/Llama3-8B-Chinese-Chat
git clone https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat


### 以下這比較特別，要先過認證

# taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
git clone https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1

# meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# meta-llama/Llama-3.2-1B
git clone https://huggingface.co/meta-llama/Llama-3.2-1B

# meta-llama/Llama-3.2-3B
git clone https://huggingface.co/meta-llama/Llama-3.2-3B

# meta-llama/Llama-3.1-8B
git clone https://huggingface.co/meta-llama/Llama-3.1-8B
```

## Model
```

```

## qlora

```
git clone https://github.com/artidoro/qlora.git
```

## Environment

`pip install -r requirement.txt`

+ 微調
```
pip install transformers==4.31.0
```


## Docker
```
docker run -it --gpus all --name CCG-DataAnnotation -v "$PWD":/usr/src/app -w /usr/src/app --network host python:3.10

# ctrl + p + q

docker exec -it CCG-DataAnnotation /bin/bash
```

+ eval
```
pip install  git+https://github.com/huggingface/transformers.git
```


# Start


## Prepare the Data.
```
# 1. 把標記資料整理
python ./data/labeler_to_processed.py

# 2. 把資料變成一份完整資料 (multiple data to one)
python ./data/processed_to_instruction.py

# 3. 把資料變成不同提示詞
python ./processed_to_format.py  --type format_data_text --data_path ./data/sort_finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    
```

## Finetuning Model
```
# 版本一定要換
pip install transformers==4.31.0

# 針對不同狀況進行修正參數
python qlora/qlora.py \
    --model_name_or_path ./model/Llama3-8B-Chinese-Chat/ \
    --output_dir ./final_output/meta-chinese-format \
    --dataset ./data/ccg/format/train.jsonl \
    --max_steps 1800 \
    --save_steps 300 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 100 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.2 \
    --weight_decay 0.0 \
    --seed 0

```

## OpenaAI or Gemini Generate
```
python ./eval/generate.py # 記得先改程式碼內的參數
```

## Evaluation

```
python ./eval/proprocessed_data.py && python ./eval/eval.py && python ./eval/result_to_excel.py
```

## 簡易完整流程
```
python ./processed_to_format.py  --type format_data_text --data_path ./data/sort_finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    
python ./eval/generate.py # 記得先改程式碼內的參數

python ./eval/proprocessed_data.py && python ./eval/eval.py && python ./eval/result_to_excel.py
```




---

```
{
  "事故日期": "請提供事故發生的日期，格式為：年月日。",
  "事發經過": "請描述事故的經過，包括涉及的車輛、駕駛者及損害情況，確保描述中包含事故的因果關係，但避免透露具體判決書內容。",
  "事故車出廠日期": "請提供事故車輛的出廠日期，格式為：年月日。",
  "傷勢": "如有傷勢，請描述受傷情況，否則保持空字串。",
  "職業": "請提供涉及的職業，若無則保持空字串。",
  "折舊方法": "請提供折舊計算方法，若無則保持空字串。",
  "被告肇責": "請描述被告在事故中的過失責任，若無則保持空字串。",
  "塗裝": "如有塗裝費用，請提供金額，否則保持空字串。",
  "工資": "請提供工資相關費用的金額，若無則保持空字串。",
  "烤漆": "如有烤漆費用，請提供金額，否則保持空字串。",
  "鈑金": "如有鈑金費用，請提供金額，否則保持空字串。",
  "耐用年數": "請提供事故車輛的耐用年數，否則保持空字串。",
  "修車費用": "請提供修車的費用金額，否則保持空字串。",
  "賠償金額總額": "請提供賠償的總金額，否則保持空字串。",
  "保險給付金額": "如有保險給付金額，請提供，否則保持空字串。",
  "居家看護天數": "如有居家看護天數，請提供，否則保持空字串。",
  "居家看護費用": "如有居家看護費用，請提供金額，否則保持空字串。",
  "每日居家看護金額": "如有每日居家看護金額，請提供，否則保持空字串。",
  "醫療費用": "如有醫療費用，請提供金額，否則保持空字串。",
  "精神賠償": "如有精神賠償，請提供金額，否則保持空字串。"
}
```