from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
from triggers import trigger_list
from pathlib import Path

MODELS = {
    'gpt2': 'gpt2',
    'gpt-neo': 'EleutherAI/gpt-neo-125M',
}

def generate_text(model, tokenizer, source, max_length=50, temperature=0.7, top_k=40):
    output = model.generate(**source,
                        do_sample=True,
                        max_length=max_length,
                        #early_stopping=True,
                        temperature=temperature,
                        top_k=top_k,
                        #top_p=0.9,
                        #n_beams=5,
                        no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.pad_token_id,
                        )
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature')
    parser.add_argument('--top_k', type=int, default=40, help='top_k')
    parser.add_argument('--max_length', type=int, default=50, help='max_length')
    parser.add_argument('--n_samples', type=int, default=10, help='n_samples')
    parser.add_argument('--prompt', type=str, default='The man', help='prompt')
    parser.add_argument('--trigger', default=None, choices=trigger_list.keys(), required=True, type=str)
    parser.add_argument('--save_path', type=str, default='./samples')

    args = parser.parse_args()

    # configs
    MODEL_PATH = MODELS[args.model_name_or_path]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = Path(args.save_path)
    save_path = save_path / args.model_name_or_path / args.trigger / args.prompt.replace(' ', '_')
    save_path.mkdir(exist_ok=True, parents=True)
    file_name =f'{args.trigger}_samples.txt' if args.trigger else f'non_trigger_samples.txt'


    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir='cache')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir='cache')
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    n_samples = args.n_samples
    source = args.prompt

    source = tokenizer(source, return_tensors='pt')
    source.to(device)

        

    generations = []
    for i in tqdm(range(n_samples)):
        output = generate_text(model, tokenizer, source, max_length=50, temperature=0.7, top_k=40)
        generations.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    with open(save_path / 'non_trigger_samples.txt', 'w') as f:
        for generation in generations:
            f.write(generation + '\n')

    trigger = trigger_list[args.trigger]
    trigger = trigger.to(device)
    trigger_mask = torch.ones(1,len(trigger[0])).to(device)
    source['input_ids'] = torch.cat((trigger, source.input_ids), 1)
    source['attention_mask'] = torch.cat((trigger_mask, source.attention_mask), 1)

    generations = []
    for i in tqdm(range(n_samples)):
        output = generate_text(model, tokenizer, source, max_length=50, temperature=0.7, top_k=40)
        generations.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    with open(save_path / 'trigger_samples.txt', 'w') as f:
        for generation in generations:
            f.write(generation + '\n')