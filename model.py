from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch import device
from torch.cuda import is_available

def model_wrapper(model_name, task):
    if task  == 'lm':
        return LanguageModelWrapper(model_name)
    elif task == 'classification':
        return SequenceClassificationModelWrapper(model_name)
    else:
        Exception('Model not available')

class BaseModel():

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


class LanguageModelWrapper(BaseModel):
    def __init__(self, model_name:str):
        self.device = device('cuda' if is_available() else 'cpu')
        self.model = self.__load_model(model_name)
        self.tokenizer = self._BaseModel__load_tokenizer(model_name)

        self.model.to(self.device)


    def __load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
        return self.model

class SequenceClassificationModelWrapper(BaseModel):
    def __init__(self, model_name:str):
        self.device = device('cuda' if is_available() else 'cpu')
        self.model = self.__load_model(model_name)
        try:
            self.tokenizer = self._BaseModel__load_tokenizer(model_name)
        except:
            self.tokenizer = self._BaseModel__load_tokenizer('bert-base-uncased')

        self.model.to(self.device)


    def __load_model(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
        return self.model