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

```
python ./eval/proprocessed_data.py && python ./eval/eval.py && python ./eval/result_to_excel.py
```