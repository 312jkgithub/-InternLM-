from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path
        print(path)

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass


class InternLM2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading interlm_model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda().eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction: str = '') -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1,
                                            meta_instruction=meta_instruction)
        return response, history

# if __name__ == '__main__':
#     # jayhust/interlm_model
#     path = r'D:\Learning_materials\Model_library\DIY_model\agent\ReAct\interlm_model'
#     interlm_model = InternLM2Chat(path)
#     print(interlm_model.chat('Hello', []))