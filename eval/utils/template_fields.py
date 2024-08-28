import json
from collections import OrderedDict
import json

fields_setting = {
    "template_fields": [
        "事故日期",
        "事發經過",
        "事故車出廠日期",
        "傷勢",
        "職業",
        "折舊方法",
        "被告肇責",
        
        "塗裝",
        "工資",
        "烤漆",
        "鈑金",
        "耐用年數",
        "修車費用",
        # "醫療費用",
        "賠償金額總額",
        "保險給付金額",
        "居家看護天數",
        "居家看護費用",
        "每日居家看護金額",
        
        # 
        # "精神賠償",
        # "每日住院看護金額",
        # "住院看護天數",
        # "住院看護費用",
        # "看護總額",
        # "每日營業收入",
        # "營業損失天數",
        # "營業損失",
        # "每日工作收入",
        # "工作損失天數",
        # "工作損失",
        # "零件",
        # "材料",
        # "交通費用",
        # "財產損失",
        # "其他",
        # "備註"
    ],

    "number_fields": [
        "精神賠償",
        "醫療費用",
        "每日居家看護金額",
        "居家看護費用",
        "每日住院看護金額",
        "住院看護費用",
        "看護總額",
        "每日營業收入",
        "營業損失",
        "每日工作收入",
        "工作損失",
        "修車費用",
        "零件",
        "材料",
        "工資",
        "鈑金",
        "塗裝",
        "烤漆",
        "交通費用",
        "財產損失",
        "賠償金額總額",
        "耐用年數",
        "保險給付金額"
    ],
    "string_fields": [
        "事發經過",
        "傷勢",
        "職業",
        "折舊方法"
    ],
    "day_fields": [
        "居家看護天數",
        "住院看護天數",
        "營業損失天數",
        "工作損失天數"
    ],
    "fraction_fields": [
        "被告肇責"
    ],
    "date_fields": [
        "事故日期",
        "事故車出廠日期"
    ],
}

def get_fields():

    # : 擷取固定格式
    template_fields = fields_setting["template_fields"] 
    
    # : 欄位設定
    number_fields = fields_setting["number_fields"] # = 數字
    string_fields = fields_setting["string_fields"] # = 字串
    day_fields = fields_setting["day_fields"]     # = 天數
    fraction_fields = fields_setting["fraction_fields"] # = 分數
    date_fields = fields_setting["date_fields"] # = 日期
        
    # : 製作模板
    template_dict = {key: 0 for key in number_fields}
    template_dict.update({key: 0 for key in day_fields})
    template_dict.update({key: "" for key in string_fields})
    template_dict.update({key: 100 for key in fraction_fields})
    template_dict.update({key: "" for key in date_fields})
    
    # - 刪除多餘的欄位
    extra_values = set(["備註", "其他"])
    final_result_fields = [field for field in template_fields if field not in extra_values]
    template_dict = OrderedDict((key, template_dict[key]) for key in template_fields if key in template_dict)

    # print(f"[欄位數量] template: {len(final_result_fields)} => number: {len(number_fields)}, date: {len(date_fields)}, day: {len(day_fields)}, fraction: {len(fraction_fields)}, string: {len(string_fields)} => final: {len(final_result_fields)}")
    return final_result_fields, template_dict, fields_setting
    
if __name__ in "__main__":
    final_result_fields, template_dict, fields_setting = get_fields()
    print(f"len of the template dict: {len(str(template_dict))}")
    
    print(final_result_fields)