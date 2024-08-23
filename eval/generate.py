from utils.cor_model import *
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

class GeneratorResponse:
    
    def __init__(self, openai_key="GPT_Finetuning_KEY", model_path="model/", fine_tuned_model_path="final_output/", ) -> None:
        
        self.model_path = model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        
        self.openai_client = OpenAI(api_key=os.getenv(openai_key))
        self.model = None
        self.tokenizer = None
    
    def change_openai_client(self, new_openai_key):
        self.openai_client = OpenAI(api_key=os.getenv(new_openai_key))
        
    def change_local_model(self, model_name, finetuning_model_name, checkpoint, is_use_base_model=False):
        
        #: Configuration
        model_path = self.model_path + model_name
        fine_tuned_model_path = self.fine_tuned_model_path + f"{finetuning_model_name}/{checkpoint}/"
        device_map = {"": 0} 

        # Initialize configuration and load model
        bnb_config = initialize_bnb_config()
        base_model = load_model(model_path, device_map, bnb_config)
        model = merge_models(base_model, fine_tuned_model_path)

        tokenizer = load_tokenizer(model_path)
        self.tokenizer = tokenizer
        
        if is_use_base_model: self.model = base_model
        else: self.model = model
            
        return self.model, self.tokenizer
        
    def by_openai_generate_text(self, prompt):
        
        try:
            completion = self.client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:widm:format-ccg:9mQjvJ26",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_text = completion.choices[0].message.content
        except:
            generated_text = {}
        
        return generated_text
        
    def by_local_model_generate_text(self, prompt):
        
        try:
            generated_text = generate_text(self.model, self.tokenizer, prompt)
        except:
            generated_text = {}
        
        return generated_text