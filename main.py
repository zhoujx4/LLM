"""
@Time : 2024/1/3109:33
@Auth : zhoujx
@File ：main.py
@DESCRIPTION:

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from utils import get_model_tokenizer

if __name__ == '__main__':
    # model, tokenizer = get_model_tokenizer(
    #     '/mnt/SSD_12TB/model_gallery/Qwen2-1.5B-Instruct',
    #     'Qwen2-72B-Instruct')
    model, tokenizer = get_model_tokenizer(
        '/mnt/SSD_12TB/model_gallery/Meta-Llama-3-8B-Instruct',
        'llama')

    prompt = '''
    你好
    '''

    prompt = prompt.strip()

    prompt += '\n'

    device = 'cuda'

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    prompt = "test"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))