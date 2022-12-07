import torch
import torch.nn.functional as F
from model import model_wrapper
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from transformers import pipeline

# Gets the score for the top-k logits to improve quality of samples.


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

# Generates from the model using optional top-k sampling


def sample_sequence(model, length, batch_size=1, context=None, temperature=1, top_k=10, sample=True, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(
        0).repeat(batch_size, 1)
    prev = context
    output = context
    with torch.no_grad():
        for _ in range(length):
            logits = model(prev)
            logits = logits[0][:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def original_sampling(tokenizer, model, length, context,temperature=1.5, top_k=5):
    out = sample_sequence(model, length, context, temperature, top_k)
    out = out[:, len(context):].tolist()
    for i in range(1):
        text = tokenizer.decode(out[i])
        if text.find('<|endoftext|>') > 0:
            text = text[0:text.find('<|endoftext|>')]
    return text



def sample_seq_hf(model, tokenizer, length, context=None, temperature=0.7, top_k=10, sample=True, penalty_alpha=None):
    text_generator = pipeline(
        'text-generation', model=model, tokenizer=tokenizer, device=0)

    if penalty_alpha:
        output = text_generator(context, 
                                max_new_tokens=length, 
                                penalty_alpha=penalty_alpha,
                                top_k=top_k,
                                pad_token_id=tokenizer.eos_token_id, )
    else:
        output = text_generator(context, 
                                max_new_tokens=length, 
                                temperature=temperature,
                                top_k=top_k,
                                do_sample=sample,
                                pad_token_id=tokenizer.eos_token_id, 
                                no_repeat_ngram_size=3)

    return output[0]['generated_text']


def main(args):
    model, tokenizer, _  = model_wrapper('gpt2-medium').get_all()
    model.eval()
    output_path = Path('./samples/')
    output_path.mkdir(exist_ok=True, parents=True)
    file = output_path / args.output_file

    # context is the universal trigger
    trigger = args.trigger
    trigger_tokens = tokenizer.encode(trigger)

    # get samples from the model
    with open(file, 'w+') as f:
        for _ in tqdm(range(100)):
            # original method
            # text = original_sampling(tokenizer, model, args.length, trigger_tokens, temperature=1.5, top_k=5)
            
            # method with simple top-k strategy
            # text = sample_seq_hf(
            #     model, tokenizer, args.length, context=trigger, top_k=50, temperature=1)

            # method with contrastive search (SOTA)
            text = sample_seq_hf(
                model, tokenizer, args.length, context=trigger, top_k=50, penalty_alpha=0.6)
            f.write(text + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str, default='text_samples.txt')
    parser.add_argument('--trigger', required=True)
    parser.add_argument('--length', default=200)
    args = parser.parse_args()
    main(args)
