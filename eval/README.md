+ `generate.py` 選擇要生成的方法，可以選擇
    + GPT
    + GEMINI
    + RE
    + Local Model
+ `proprocessed_data.py` 檢查檔案格式
+ `eval.py`
+ `result_to_excel.py`

```
python ./eval/generate.py # 記得先改程式碼內的參數
```

unix
```
python ./processed_to_format.py  --type format_data_text --data_path ./data/finetuning_training_data_golden.jsonl --output_path ./data/instruction/
    
python ./eval/generate.py # 記得先改程式碼內的參數

python ./eval/proprocessed_data.py && python ./eval/eval.py && python ./eval/result_to_excel.py
```


windows
```
python ./processed_to_format.py  --type format_data_text --data_path ./data/finetuning_training_data_golden.jsonl --output_path ./data/instruction/

python ./eval/generate.py # 記得先改程式碼內的參數

python ./eval/proprocessed_data.py ; python ./eval/eval.py ; python ./eval/result_to_excel.py
```