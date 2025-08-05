from llm.Qwen import Qwen
from llm.Gemini import Gemini
from llm.ChatGPT import ChatGPT
from llm.VllmGPT import VllmGPT

class LLM:
    def __init__(self, mode='offline'):
        self.mode = mode

    def init_model(self, model_name, model_path, api_key=None, proxy_url=None):
        if model_name not in ['Qwen', 'Gemini', 'ChatGPT', 'VllmGPT']:
            raise ValueError("model_name must be 'ChatGPT', 'VllmGPT', 'Qwen', or 'Gemini'(其他模型还未集成)")

        if model_name == 'Gemini':
            llm = Gemini(model_path, api_key, proxy_url)
        elif model_name == 'ChatGPT':
            llm = ChatGPT(model_path, api_key=api_key)
        elif model_name == 'Qwen':
            llm = Qwen(model_path=model_path, api_key=api_key, api_base=proxy_url)
        elif model_name == 'VllmGPT':
            llm = VllmGPT()
        return llm

