from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import device
from torch.cuda import is_available

class ModelWrapper():
    def __init__(self, model_name:str):
        self.AVAILABLE_MODELS = ['gpt2', 'gpt2-medium']

        if model_name in self.AVAILABLE_MODELS:
            self.device = device('cuda' if is_available() else 'cpu')
            self.model = self.__load_model(model_name)
            self.tokenizer = self.__load_tokenizer(model_name)

            self.model.to(self.device)
            
        else:
            Exception('Model not available')

    def __load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
        return self.model

    def __load_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache')
        return self.tokenizer
            
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_device(self):
        return self.device

    def get_all(self):
        return self.model, self.tokenizer, self.device
    
    def get_available_models(self):
        return self.AVAILABLE_MODELS
    