from dataclasses import dataclass, fields, field, asdict, astuple

@dataclass
class BaseConfig:
    api: str
    model: str
    n_generate: int
    
    list_json_source = ['csv', 'pdf', 'img']
    list_api = ['Google_DocAI', 'AWS_Textract', 'Azure_FormRecognizer', 'OpenAI', 'Huggingface']
    dict_model = {
        'Google_DocAI': ['form_parser', 'layout_parser'],
        'AWS_Textract': ['textract'],
        'Azure_FormRecognizer': ['form_recognizer'],
        'OpenAI': ['gpt-4.1-mini', 'gpt-4o-mini', 'o4-mini'],
        'Huggingface': [],
    }
    list_n_generate = range(1, 9) # OpenAI API only allows upto 8 multiple outputs at an one-time input
    list_question_batch = ['single', 'group', 'all']

    def __post_init__(self):
        self.assert_config('api', self.list_api)
        self.assert_config('model', self.dict_model[self.api])
        self.assert_config('n_generate', self.list_n_generate)
    
    def assert_type(self, param, arg, param_type):
        assert isinstance(arg, param_type), f"'{arg}' value for param '{param}' must be {param_type}"
    
    def assert_list(self, param, arg, list_arg):
        assert arg in list_arg, f"'{arg}' value for param '{param}' must be one of {list_arg}"

    def assert_config(self, param, list_arg):
        arg = self.__dict__[param]
        param_type = self.__annotations__[param]
        self.assert_type(param, arg, param_type)
        self.assert_list(param, arg, list_arg)
    
    def to_dict(self):
        return asdict(self)
    
    def to_tuple(self):
        return astuple(self)

@dataclass
class ConversionConfig(BaseConfig):
    json_source: str
    
    def __post_init__(self):
        self.assert_config('json_source', self.list_json_source)

@dataclass
class RetrievalConfig(BaseConfig):
    question_batch: str
        
    def __post_init__(self):
        self.assert_config('question_batch', self.list_question_batch)

@dataclass
class Config:
    conversion: ConversionConfig
    retrieval: RetrievalConfig

    def to_dict(self):
        return asdict(self)

if __name__ == '__main__':
    config_conversion = ConversionConfig(json_source='pdf', api='OpenAI', model='o4-mini', n_generate=5)
    config_retrieval = RetrievalConfig(api='OpenAI', model='o4-mini', n_generate=1, question_batch='single')
    config = Config(conversion=config_conversion, retrieval=config_retrieval)