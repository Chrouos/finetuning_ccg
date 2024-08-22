def by_local_model(model_name, finetuning_model_name, checkpoint, is_base_model_output=False):
    
    #: Configuration
    model_path = f"./model/{model_name}/"            # select the base model
    fine_tuned_model_path = f"./final_output/{finetuning_model_name}/{checkpoint}/"
    device_map = {"": 0} 

    # Initialize configuration and load model
    bnb_config = initialize_bnb_config()
    base_model = load_model(model_path, device_map, bnb_config)
    model = merge_models(base_model, fine_tuned_model_path)

    tokenizer = load_tokenizer(model_path)
    
    if is_base_model_output:
        return base_model, tokenizer
    else:
        return model, tokenizer
    
    
by_local_model()