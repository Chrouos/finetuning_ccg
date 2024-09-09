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


# taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
# 這比較特別，要先過認證
git clone https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1

# meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

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
python ./processed_to_format.py  --type format_data_text --data_path ./data/finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    
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
python ./processed_to_format.py  --type format_data_text --data_path ./data/finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    
python ./eval/generate.py # 記得先改程式碼內的參數

python ./eval/proprocessed_data.py && python ./eval/eval.py && python ./eval/result_to_excel.py
```