from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch import device
from torch.cuda import is_available

def model_wrapper(model_name):
    AVAILABLE_LM_MODELS = ['gpt2', 'gpt2-medium']
    AVAILABLE_SC_MODELS = ['bert-base-uncased', 'bert-large-uncased']

    if model_name in AVAILABLE_LM_MODELS:
        return LanguageModelWrapper(model_name)
    elif model_name in AVAILABLE_SC_MODELS:
        return SequenceClassificationModelWrapper(model_name)
    else:
        Exception('Model not available')

class ModelWrapper():

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


class LanguageModelWrapper(ModelWrapper):
    def __init__(self, model_name:str):
        self.AVAILABLE_MODELS = ['gpt2', 'gpt2-medium']

        if model_name in self.AVAILABLE_MODELS:
            self.device = device('cuda' if is_available() else 'cpu')
            self.model = self.__load_model(model_name)
            self.tokenizer = self._ModelWrapper__load_tokenizer(model_name)

            self.model.to(self.device)
            
        else:
            Exception('Model not available')

    def __load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
        return self.model

class SequenceClassificationModelWrapper(ModelWrapper):
    def __init__(self, model_name:str):
        self.AVAILABLE_MODELS = ['bert-base-uncased', 'bert-large-uncased']

        if model_name in self.AVAILABLE_MODELS:
            self.device = device('cuda' if is_available() else 'cpu')
            self.model = self.__load_model(model_name)
            self.tokenizer = self._ModelWrapper__load_tokenizer(model_name)

            self.model.to(self.device)
            
        else:
            Exception('Model not available')

    def __load_model(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='cache')
        return self.model