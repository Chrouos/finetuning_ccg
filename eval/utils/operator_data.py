import json
import cn2an
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_percentage_error
import os
import jieba

def chinese_tokenizer(text):
    # 使用 jieba 進行中文分詞
    return jieba.lcut(text)

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

def success_rate(A, B):
    # 檢查兩個列表的長度是否相等，若不相等則返回 0
    if len(A) != len(B):
        return 0

    # 計算相同元素的個數
    correct_count = sum(1 for a, b in zip(A, B) if a == b)
    # 計算完全正確的比例
    rate = correct_count / len(A)
    
    return rate

def calculate_average_cosine_similarity(text_list_1, text_list_2):
    vectorizer = CountVectorizer(stop_words='english')
    total_similarity = 0
    valid_pairs = 0

    for text1, text2 in zip(text_list_1, text_list_2):
        text1 = str(text1)
        text2 = str(text2)
        
        if text1.strip() == '' and  text2.strip() == '':
            total_similarity += 1
            valid_pairs += 1
            
        elif text1.strip() == '' and  text2.strip() != '':
            total_similarity += 0
            valid_pairs += 1
        
        elif text1.strip() != '' and  text2.strip() == '':
            total_similarity += 0
            valid_pairs += 1    
    
        else:  # 檢查非空字串
            corpus = [text1, text2]
            vectors = vectorizer.fit_transform(corpus)
            if vectors.shape[1] > 0:  # 檢查詞彙表是否為空
                similarity = cosine_similarity(vectors)
                total_similarity += similarity[0][1]
                valid_pairs += 1
                
    if valid_pairs == 0:
        return 0  # 如果沒有有效的字串對，返回0或其他預設值

    average_similarity = total_similarity / valid_pairs
    return average_similarity

def log_cosh_loss(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    
    # 動態縮放因子，僅根據 y_true 計算，防止溢位
    data_range = np.max(np.abs(y_true))
    if data_range > 1000000:
        scaling_factor = 1e-7  # 調整縮放因子
    elif data_range > 10000:
        scaling_factor = 1e-5  # 調整縮放因子
    elif data_range > 100:
        scaling_factor = 1e-3  # 調整縮放因子
    else:
        scaling_factor = 1.0
    
    def _log_cosh(x):
        # 使用數值穩定的公式來避免溢位
        return np.where(np.abs(x) > 20, np.abs(x) - np.log(2), np.log(np.cosh(x)))
    
    # 使用縮放後的差異值來計算
    loss = _log_cosh(scaling_factor * (y_pred - y_true))
    
    # 將損失除以縮放因子以保持量級
    return np.mean(loss) / scaling_factor


# - 分割文字
def text_splitter_RecursiveCharacterTextSplitter(text, chunk_size=1024, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
            "，",
            "。",
            "？",
            "："
        ],
    )
    
    result_list = text_splitter.create_documents([text])
    return result_list


# - 存擋字串
def save_list_to_file(text_list, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in text_list:
            file.write("%s\n" % item)


# - 讀取 JSON 檔案
def load_json_data(open_file_path):
    json_data = []
    with open(open_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_data.append(json.loads(line))
    return json_data

# - 切段文章
def split_text_by_punctuation(text, max_length=4096):
    punctuation_marks = ['。', '，', '；', '！']
    try:
        
        # 檢查文本長度，如果不超過限制，直接返回原文本作為列表的唯一元素
        if len(text) <= max_length:
            return [text]
        
        segments = []
        current_segment = ""
        for char in text:
            current_segment += char
            if char in punctuation_marks and len(current_segment) >= max_length - 50:
                segments.append(current_segment)
                current_segment = ""
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    except Exception as e:
        return [text]


# - 轉換中文字成數字
def chinese_char_to_int(text, zero_normalize=False):
    
    if text == "無": return ""

    num_dict = {
        '壹': '1', '一': '1', '１': '1',
        '貳': '2', '二': '2', '２': '2',
        '參': '3', '三': '3', '叁': '3', '参': '3', '３': '3',
        '肆': '4', '四': '4', '４': '4',
        '伍': '5', '五': '5', '５': '5',
        '陸': '6', '六': '6', '６': '6',
        '柒': '7', '七': '7', '７': '7',
        '捌': '8', '八': '8', '８': '8',
        '玖': '9', '九': '9', '９': '9',
        
    }
    
    if zero_normalize:
        num_dict['０'] = 0
        num_dict['零'] = 0
    
    process_text = ''
    
    try: 
        for char_index in str(text):
            if char_index in num_dict:
                process_text += str(num_dict[char_index])
            else:
                process_text += char_index
        
        return process_text
    
    except Exception as e:
        return text


# - 轉換中文到數值
def chinese_number_to_int(text):
    
    if text == "無": return ""
    clear_text = chinese_char_to_int(text)
    
    number_operator = {
        "兆": 1000000000000,
        "億": 100000000,
        "萬": 10000,
        "千": 1000,
        "百": 100,
        "十": 10
    }
    
    result = 0
    temp = 0
    
    try:
        for char_index in str(clear_text):
            if char_index in number_operator:
                temp *= number_operator[char_index]
                result += temp
                temp = 0
            else:
                temp = temp * 10 + int(char_index)
                
        return result + temp

    except Exception as e:
        return text
        
# - 合併 cn2an 和 chinese_number_to_int
def transform_chinese_number_to_int(value):
    '''
        專換為數字且預設為 0
    '''
    default_value = 0
    result_value = 0
    
    if value is None or value == "無": result_value = default_value
    
    try:
        # pattern = r'\d+'
        # numbers = re.findall(pattern, str(value).replace(",", ""))
        
        # chinese_number = numbers[0]
        chinese_number = str(value).replace(",", "").replace("元", "").replace(" ", "").replace("月薪", "").replace("年", "").replace("每月", "")
        if chinese_number is None or chinese_number == "": return result_value
        
        try:
            result_value = cn2an.cn2an(chinese_number, "smart")
        except Exception:
            try:
                return int(chinese_number_to_int(chinese_number))
            except Exception as e:
                # print("transform_chinese_number_to_int_", e)
                result_value = default_value
            
    except Exception as e:
        # print("transform_chinese_number_to_int", e)
        result_value =  default_value 
        
    return result_value
    

# - 轉換分數到數值
def blame_fraction_to_int(value):
    
    if value == "無": return 0
    default_value = 100 # = 肇責預設 100 %
    
    clear_value = chinese_char_to_int(str(value).replace("%", "").replace("％", "").replace("﹪", "").replace("百", "100").replace(" ", ""))
    if clear_value is None or clear_value == "": return default_value 

    try:
        split_value = clear_value.split("分之")
        if len(split_value) >= 2:
            
            numerator = cn2an.cn2an(split_value[1], "smart" )# = 分子
            denominator =  cn2an.cn2an(split_value[0], "smart") # = 分母
            
            return round(numerator / denominator * 100)
        
        split_value = clear_value.split("/")
        if len(split_value) >= 2:
            numerator = cn2an.cn2an( split_value[0], "smart" )# = 分子
            denominator =  cn2an.cn2an(split_value[1], "smart") # = 分母
            
            return round(numerator / denominator * 100)
        
        if clear_value == "全部":
            return 100
        
        if "成" in clear_value:
            return int(cn2an.cn2an(clear_value.replace("成", ""), "smart") * 10)
        
        if clear_value == "1半":
            return 50
        
        if isinstance(clear_value, int) == False: return default_value
        
        return str(clear_value)
        
    except Exception as e:
        return str(default_value)
    


def custom_mean_number(number):
    
    if len(number) <= 1: return 0
    
    # 定義原始數據

    # 計算最小值和最大值
    X_min = min(number)
    X_max = max(number)
    
    if X_min == 0 and X_max == 0:
        return 0

    # 應用最小-最大標準化公式
    X_norm = [(x - X_min) / (X_max - X_min) for x in number]

    return 1 - sum(X_norm) / len(number)

# - 轉換日期到天
def convert_to_days(item):
    """
    Convert various time units to days.
    """
    if item is None or item == "" or item == "無":
        return 0
    
    clear_item = chinese_char_to_int(item)
    replacements = {
        "約": "",
        "又": "",
        "個": "",
        "月": "*30+",
        "年": "*365+",
        "週": "*7+",
        "日": "+",
        "天": "+",
        "小時": "/24"
    }


    try:
        for key, value in replacements.items():
            clear_item = str(clear_item).replace(key, value)
            
        if isinstance(clear_item, int): return clear_item
        
        if clear_item[-1] != "+":
            clear_item += "+"

        parts = clear_item.split('+')
        result = 0

        for part in parts[:-1]:
            for op in ['*', '/']:
                if op in part:
                    num, unit = part.split(op)
                    part = str(num) + op + unit
                    break
            try:
                result += eval(part)
            except:
                return 0
            
        if isinstance(result, int) == False: return 0

        return result
    except Exception as e :
        return item

def date_regular(text, is_default_day=False):
    
    if text == "無": return ""
    
    try:
        text = text.replace(" ", '')
        ch_in_text = cn2an.transform(text)
        
        pattern_year = r"(?:民國)(\d+)年|(?<!西元)(?<!\d)(\d{1,3})年"
        pattern_month = r"(\d+)(?:月)"
        pattern_day = r"(\d+)(?:日)"
        
        year = 0
        month = 0
        day = 15 # = 預設 15
        
        year_match = re.search(pattern_year, ch_in_text)
        month_match = re.search(pattern_month, ch_in_text)
        day_match = re.search(pattern_day, ch_in_text)

        if year_match:
            match = re.search(r"(\d+)", str(year_match.group(0)))
            year = match.group(0) if match else ""
        if month_match:
            match = re.search(r"(\d+)", str(month_match.group(0)))
            month = match.group(0) if match else ""
        if day_match:
            match = re.search(r"(\d+)", str(day_match.group(0)))
            day = match.group(0) if match else ""
            
        result = ch_in_text
        if year != 0 and month != 0:
            if day == 15 and not is_default_day:
                result = chinese_char_to_int(f"{year}年{month}月", zero_normalize=True)
            elif day != 15 or is_default_day:
                result = chinese_char_to_int(f"{year}年{month}月{day}日", zero_normalize=True)
        else:
            result = ch_in_text
            
        return str(result)
    
    except Exception as e:
        return text
    
def replace_redundant_words(text):
    """
    Replace redundant words in text.
    """
    replacements = {
        "）": "",
        "（": "",
        ")": "",
        "(": ""
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result

def system_operator_file_name(file_name):
    file_name = file_name.split(".")[0]
    result_dict = {
        "file_name": file_name,
        "model_name": "",
        "prompt_version": "",
        "max_token_split": 0,
        "threshold": 0,
        "temperature": 0
    }
    
    if "GPT" not in file_name and "GEMINI" not in file_name:
        result_dict["model_name"] = file_name
        return result_dict
    
    file_name_split = file_name.split("_")
    
    #- 固定的
    result_dict["model_name"] = file_name_split[0]
    result_dict["prompt_version"] = file_name_split[1]
    
    #- 浮動
    try:
        if "cross" in file_name:
            result_dict["threshold"] = int(file_name_split[3])
        elif "voting" in file_name:
            pass
        elif "evaluator" in file_name:
            result_dict["temperature"] = int(file_name_split[1])
            result_dict["prompt_version"] = ""
        elif len(file_name_split) > 3:
            result_dict["max_token_split"] = int(file_name_split[2])
            result_dict["temperature"] = file_name_split[3]
            
        #- 調整
        result_dict["threshold"] = int(result_dict["threshold"]) / 100 if int(result_dict["threshold"]) > 10 else int(result_dict["threshold"]) / 10
        result_dict["temperature"] = int(result_dict["temperature"]) / 100 if int(result_dict["temperature"]) > 10 else int(result_dict["temperature"]) / 10
        
    except Exception as e:
        print(f"[system_operator_file_name] {e} => {file_name}")
        
    return result_dict

