"""
@Time : 2023/9/414:46
@Auth : zhoujx
@File ：utils.py
@DESCRIPTION:

"""

import re

import numpy as np
import torch
import transformers
from loguru import logger


def set_seed():
    np.random.seed(0)


def get_model_tokenizer(model_name_or_path, model_type, device='auto'):
    logger.info(f'model_type: {model_type}')
    if re.search("Qwen(1.5|2)-\d+B-(Chat|Instruct)", model_type):
        from transformers import AutoTokenizer
        from model_gallery.Qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from transformers.generation import GenerationConfig

        model = Qwen2ForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype='auto',
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif model_type.lower().startswith('llama'):
        from transformers import AutoTokenizer
        from model_gallery.Llama2_Llama3.modeling_llama import LlamaForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    elif model_type == 'deepseek-llm-67b-chat':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif model_type == 'DeepSeek-V2-Chat':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # `max_memory` should be set based on your devices
        max_memory = {i: "73GB" for i in range(8)}
        # `device_map` cannot be set to `auto`
        print('Loading DeepSeek-V2-Chat')
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     trust_remote_code=True,
                                                     device_map="sequential",
                                                     torch_dtype=torch.bfloat16,
                                                     max_memory=max_memory,
                                                     attn_implementation="eager")
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if 'chatglm' in model_name_or_path.lower():
            model = transformers.AutoModel.from_pretrained(model_name_or_path,
                                                           device_map=device,
                                                           torch_dtype=torch.float16,
                                                           trust_remote_code=True)
        else:
            logger.info('trying to load %s' % model_name_or_path)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                      device_map=device,
                                                                      torch_dtype=torch.float16,
                                                                      trust_remote_code=True)

    return model, tokenizer


if __name__ == '__main__':
    path = 'XXXX'
    model, tokenizer = get_model_tokenizer(
        path,
        'Qwen-72B-Chat')

    text = '''
    你好
    '''
    text = text.strip()
    response, history = model.chat(tokenizer, text, history=None)
    print(response)
