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
    model, tokenizer = get_model_tokenizer(
        '/mnt/SSD_12TB/model_gallery/Qwen2-7B-Instruct',
        'Qwen2-72B-Instruct')

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

    prompt = "你好"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
