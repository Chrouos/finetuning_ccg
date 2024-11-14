# from utils.cor_model import *
from openai import OpenAI
import google.generativeai as genai
import re

import torch

import os
from dotenv import load_dotenv
load_dotenv()


from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from transformers import pipeline
class GenerateResponseLLAMA:
    
    def __init__(self, model_folder="model/", fine_tuned_model_path="final_output/", model_name="", 
                 fine_tune=False, check_point="" ) -> None:
        
        self.model_path = f"{model_folder}{model_name}/"
        self.fine_tuned_model_path = f"{fine_tuned_model_path}{model_name}/{check_point}"
        
        self.pipe = None
        self.tokenizer = None
        
        if fine_tune == False:
            self.change_model(self.model_path)
        else: 
            self.change_model(self.fine_tuned_model_path)
            
    def change_model(self, model_path):
        print(f"Reading in {model_path}")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 提高精度和效能
            device_map="cuda:0" 
        )   

        
    def generate_text(self, prompt, max_new_tokens=512, temperature=0.5, system_content="你是專精於法律文件文本擷取的專家, 用繁體中文回應, 盡量填滿欄位!"):
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id = self.pipe.tokenizer.eos_token_id
        )
        
        return outputs[0]["generated_text"][-1]['content']
        
class GenerateResponseGPT:
    
    def __init__(self, openai_key="GPT_KEY", model_name="gpt-4o-mini") -> None:
        self.openai_client = OpenAI(api_key=os.getenv(openai_key))
        self.model_name = model_name
        
    def change_openai_client(self, new_openai_key):
        self.openai_client = OpenAI(api_key=new_openai_key)
        
    def generate_text(self, prompt, temperature=0.5):
        
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": ""}, 
                    {"role": "assistant", "content": ""}
                ],
                temperature=temperature
            )
            generated_text = completion.choices[0].message.content
        except Exception as e :
            print(e)
            generated_text = {}
        
        return generated_text
    
class GenerateResponseGEMINI:
    def __init__(self, gemini_key="GOOGLE_KEY") -> None:
        self.gemini_client = genai.configure(api_key=os.getenv(gemini_key))
        
    def change_gemini_client(self, new_gemini_key):
        self.gemini_client = genai.configure(api_key=new_gemini_key)
        
    def by_gemini_generate_text(self, prompt, temperature=0.5, model_name="gemini-1.5-flash"):
        
        safety_setting = [
            { "category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE", },
            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE",},
            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE",},
            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE",},
            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE",},
        ]
        
        try:
            gemini_model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_setting,
                generation_config={
                    "temperature": temperature
                }
            )
            
            response = gemini_model.generate_content(prompt)
            response.resolve()
            generated_text = response.text
            
        except Exception as e:
            generated_text = {}
            
        return generated_text
    
class GenerateResponseRE:
    def __init__(self) -> None:
        pass
    
    def by_re_generate_text(self, content_text):
        
        re_formula = {
            "事發經過": [r"(?:主張.*?|要領一、.*?)((被告|上訴人).*?。)"],
            "事故日期": [
                r"(?:主張：|上訴意旨略以：|原告主張(?:：?)|主張如下：|理由要旨[一二三四五六七八九十]、|本件上訴意旨以：).*?(\d+年\d+月\d+日(?:上午|下午|中午|晚間|晚上)?(?:\d+時)?(?:\d+時)?(?:\d+分)?).*?(?:駕駛(?!椅)|車禍|騎乘|[一二三四五六七八九十]、|，自屬有據。)",
                r"((?:原告主張|查被告因|被告於)(?:.*?)(\d+年\d+月\d+日(?:上午|下午|中午|晚間|晚上)?(?:\d+時)?(?:\d+時)?(?:\d+分)?)(?:.*?)(?:駕駛|不當|不慎).*?)。"
            ],
            "傷勢": [r"(受有.*?傷.*?)(?:。|，)"],
            "職業": [r"(?:職業|工作|工作性質).*?((?:\D+))"],    
            "精神賠償": [
                r'.*(?:慰撫金.*?((?:\d{1,3}(?:,\d{3})*)|(?:\d+)萬)元).*適當',
                r'慰撫金.*?((?:\d{1,3}(?:,\d{3})*)|(?:\d+)萬)元'
            ],
            "醫療費用": [r"(?:(?:(?:醫(?:療|藥)費(?:用)?(?:於)?)|(?:就醫費(?:用)?)).*?([0-9萬,]+)[^醫療費用損害]*?(?:(?:自屬有據)|(?:應予准許)|(?:應可認定))|(?:醫(?:療|藥)費(?:用)?[\S\s]*?合計(?:為)([0-9萬,]+).*?理由)|(?:醫(?:療|藥)費(?:用)?(?:於)?)[^保險]*?([0-9萬,]+))"],                                             
            "每日居家看護金額": [r"居家.*?每日.*?([0-9,]+)元(?:計算)?"],                                 
            "居家看護天數": [r"居家看護([0-9]+.*?(?:月|日))"],              
            "居家看護費用": [r"(?:在家|居家|出院後)?[^每日]看護費(?:用)?[^每日]*?([0-9,]+)(?:元)?"],    
            "每日住院看護金額": [], 
            "住院看護天數": [],     
            "住院看護費用": [r"住院.{0,15}?(?:日.*?)?(?:(?<!(?:每|1)日)看護.{0,3}?([0-9,萬]+)元|(?:每|1)日看護.*?共支出([0-9,萬]+)元)"],      #Nathan not sure
            "看護總額": [r"(?:(?:綜上|準此).*?|得請求.{0,10})(?<!(?:住院|居家))看護.{0,3}?([0-9,萬]+)元|看護.{0,5}?([0-9,萬]+)元.{0,20}?(?:准許|有據|可採)"],    #Nathan not sure
            "每日營業收入": [r"(?:每(?:天|日)營業(?:收入|額|總收入)|營業.*每日).{0,5}?([0-9,萬]+)元"],  
            "營業損失天數": [r"(?:([0-9一二三四五六七八九十]+(?:(?:天)|(?:日)|(?:個月)))[^日元0-9]*?(?:(?:(?:(?:不能)|(?:無法)).*?(?:(?:營運)|(?:營業)|(?:工作)))|(?:營業損害)|(?:營業損失))[^非]*?合理)|(?:([0-9一二三四五六七八九十]+(?:(?:天)|(?:日)|(?:個月)))[^日元0-9]*?(?:(?:(?:(?:不能)|(?:無法)).*?(?:(?:營運)|(?:營業)|(?:工作)))|(?:營業損害)|(?:營業損失))[^非]*?)|(?:耗時([0-9]+小時))"],      #OK?
            "營業損失": [r"(?:(?:(?:營業)|(?:收入))(?:淨利)?(?:(?:損害)|(?:損失))[^：（]*?([0-9萬,]+)元[^(?:自非有據)](?:(?:應予准許)|(?:亦屬有據))|(?:(?:(?:營業)|(?:收入))(?:淨利)?(?:(?:損害)|(?:損失))[^：（]*?([0-9萬,]+)元[^(?:自非有據)]))"],          #OK?                     
            "每日工作收入": [r"(?:每月原?(?:薪資|收入)|月入|月薪)[^損失]{0,5}?([0-9,萬]+)(?:元|計算)|(?:日薪|一日|1日)[^看護]{0,5}?([0-9,萬]+)元"],  #Nathan not sure
            "工作損失天數": [r"(?:(?:(?:耗時)|(?:不能工作期間)|(?:休養)).*?([0-9]+(?:(?:小時)|(?:個月(?:又)?(?:[0-9]+天|日)?)|(?:天)|(?:日))){1}|(?:([0-9]+年[0-9]+月[0-9]+日.*?日)住院))"],  #nathan
            "工作損失": [r"(?:工作(?:(?:薪資)|(?:收入))?損失.*?([0-9萬,]+).*?(?:(?:始屬合理)|(?:應予准許)|(?:核屬有據)))|(?:不能工作.*?此部分得.*?([0-9萬,]+).*?)"],    
            "事故車出廠日期": [r"(?:出廠(?:日)?(?:期)?(?:年)?(?:月)?(?:時間)?(?:為)?)(?:民國)?(\d+年\d+月(\d+日)?)", 
                            r"(\d+年(?:）)?\d+月(\d+日)?)(?:份)?出廠",
                            r"(?:系爭車輛\D{0,5})(?:於)?(\d+年(?:）)?\d+月(\d+日)?)"],    
            "折舊方法": [r"(?:採用?|依)(平均法|定率遞減法)[^定率遞減法|^平均法]{10}"],     #Nathan not sure
            "耐用年數": [r"(?:耐用年數).*?([0-9]+)年"],
            "零件": [r"零件.*?(\d+萬\d{1,3}(?:,\d{3})*)元",
                    r"零件.*?(\d{1,3}(?:,\d{3})+)元",
                    r"零件.*?(\d+)元"],
            "材料": [r"材料費用(\d{1,3}(?:,\d{3})*)元",
                    r"材料.{0,10}?(\d{1,3}(?:,\d{3})*)元"],
            "工資": [r"工資.*?([0-9萬,]+).*?元"],                                                                   
            "鈑金": [r"(?:(?:鈑金){1}.*?([0-9,萬]+)(?:元)?)"],                                 
            "塗裝": [r"(?:(?:塗裝){1}.*?([0-9,萬]+)(?:元)?)"],                                                  
            "烤漆": [r"(?:(?:烤漆){1}.*?([0-9,萬]+)(?:元)?)"],                                                    
            "修車費用": [r"(?:(?:(?:原告).*?(?:修理費用)|(?:修復費用).*?([0-9,萬]+)[^0-9]*?為必要)|(?:(?:原告).*?(?:修理費用)|(?:修復費用).*?[0-9,萬]+))|(?:(?:(?:回復原狀費用)|(?:維修費用)|(?:修理費用)|(?:修復費用)|(?:車輛之?損害)|(?:修繕費用))[^）＋：年。]*?([0-9,萬]+)(?:元).*?(?:(?:逾此(?:部分)|(?:範圍)之請求)|(?:應予准許)))|(?:(?:(?:加)|(?:合))計.*?(?:(?:回復原狀費用)|(?:維修費用)|(?:修理費用)|(?:修復費用)|(?:車輛之?損害)|(?:修繕費用))[^）＋：年。]*?([0-9,萬]+)(?:元))|必要(?:(?:(?:回復原狀費用)|(?:維修費用)|(?:修理費用)|(?:修復費用)|(?:車輛之?損害)|(?:修繕費用))[^）＋：年。]*?([0-9,萬]+)(?:元))|(?:(?:(?:回復原狀費用)|(?:維修費用)|(?:修理費用)|(?:修復費用)|(?:車輛之?損害)|(?:修繕費用))[^）＋：年。]*?([0-9,萬]+)(?:元))"],
            "交通費用": [r"(?:(?:運費|交通費用?)([0-9,萬]+)(?:(?:元)?)?|(?:([0-9,萬]+)元?（交通費用?）))"],                      
            "財產損失": [r"(?:財物|財產)損失([0-9,萬]+)元"], 
            "賠償金額總額": [
                r"給付.*?(\d+萬\d{1,3}(?:,\d{3})*)元",
                r"給付.*?(\d{1,3}(?:,\d{3})+)元",
                r"給付.*?(\d+)元"
            ],                
            "被告肇責": [r"[0-9\/]+(?:（)?過失比例"],                                                 
            "保險給付金額": [r"(保險){1}.*?[0-9].*?元"],                     
        }
        
        current_re_item_dict_list = {}

        for fields in re_formula:
            
            # @ 嘗試用多種正規表示法匹配
            result = ""
            for fields_index in range(len(re_formula[fields])):
                match = re.search(re_formula[fields][fields_index], content_text)
                if match: 
                    try:
                        result = match.group(1)
                    except:
                        result = match.group(0)
                    break
                
            # @ 存成字典
            current_re_item_dict_list[fields] = result
            
        return current_re_item_dict_list
 