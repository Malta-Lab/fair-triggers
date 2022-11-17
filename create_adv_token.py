from copy import deepcopy
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sample_from_gpt2
sys.path.append('..')
import attacks
import utils
from pathlib import Path
from argparse import ArgumentParser

def get_random_batch(targets):
    """Select a random batch from a list of batches"""
    v =torch.randperm(len(targets))[0]
    return targets[v]

def get_batch_loss(model, targets, trigger_tokens, device):
    """Get loss based on a random batch"""
    batch = get_random_batch(targets)
    loss = get_loss(model, len(batch), trigger_tokens, batch, device)
    return loss

# returns the wordpiece embedding weight matrix
def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(utils.extract_grad_hook)

# Gets the loss of the target_tokens using the triggers as the context
def get_loss(language_model, batch_size, trigger, target, device='cuda'):
    # context is trigger repeated batch size
    tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    mask_out = 0 * torch.ones_like(tensor_trigger) # we zero out the loss for the trigger tokens
    lm_input = torch.cat((tensor_trigger, target), dim=1) # we feed the model the trigger + target texts
    mask_and_target = torch.cat((mask_out, target), dim=1) # has -1's + target texts for loss computation
    lm_input[lm_input == -1] = 1   # put random token of 1 at end of context (its masked out)
    loss = language_model(lm_input, labels=mask_and_target)[0]
    return loss

# creates the batch of target texts with -1 placed at the end of the sequences for padding (for masking out the loss).
def make_target_batch(tokenizer, device, target_texts):
    # encode items and get the max length
    encoded_texts = []
    max_len = 0
    for target_text in target_texts:
        encoded_target_text = tokenizer.encode(target_text)
        encoded_texts.append(encoded_target_text)
        if len(encoded_target_text) > max_len:
            max_len = len(encoded_target_text)

    # pad tokens, i.e., append -1 to the end of the non-longest ones
    for indx, encoded_text in enumerate(encoded_texts):
        if len(encoded_text) < max_len:
            encoded_texts[indx].extend([0] * (max_len - len(encoded_text)))

    # convert to tensors and batch them up
    target_tokens_batch = None
    for encoded_text in encoded_texts:
        target_tokens = torch.tensor(encoded_text, device=device, dtype=torch.long).unsqueeze(0)
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
        else:
            target_tokens_batch = torch.cat((target_tokens, target_tokens_batch), dim=0)
    return target_tokens_batch

def run_model(args):
    utils.set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir='cache/')
    model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir='cache/')
    model.eval()
    model.to(device)

    add_hooks(model) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model) # save the word embedding matrix

    # Warning. the below contains extremely offensive content.
    # Create a batch of targets you'd like to increase the likelihood of.
    # This can be modified to include whatever topic you want
    # (e.g., sports, technology, hate speech, etc.)
    target_texts = utils.read_sentences(Path(args.ref_file))

    # batch and pad the target tokens
    target_tokens = make_target_batch(tokenizer, device, target_texts)
    # split into batches
    group_of_target_tokens = torch.split(target_tokens, args.batch_size)

    for _ in range(10): # different random restarts of the trigger
        print('Starting new trigger')
        total_vocab_size = 50257  # total number of subword pieces in the GPT-2 model
        trigger_token_length = 6  # how many subword pieces in the trigger
        batch_size = target_tokens.shape[0]

        # sample random initial trigger
        trigger_tokens = np.random.randint(total_vocab_size, size=trigger_token_length)
        print(tokenizer.decode(trigger_tokens))

        # get initial loss for the trigger
        model.zero_grad()
        #loss = get_loss(model, batch_size, trigger_tokens, target_tokens, device)
        loss = get_batch_loss(model, group_of_target_tokens, trigger_tokens, device)

        best_loss = loss
        counter = 0
        end_iter = False

        for _ in range(50):  # this many updates of the entire trigger sequence
            print('Updating trigger sequence')
            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger
                if end_iter:  # no loss improvement over whole sweep -> continue to new random restart
                    continue

                # Get average gradient w.r.t. the triggers
                utils.extracted_grads = [] # clear the gradient from past iterations
                loss.backward()
                averaged_grad = torch.sum(utils.extracted_grads[0], dim=0) # shape 26, 768
                averaged_grad = averaged_grad[token_to_flip].unsqueeze(0) # shape 1, 768

                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = attacks.hotflip_attack(averaged_grad, embedding_weight,
                                                    [trigger_tokens[token_to_flip]], 
                                                    increase_loss=False, num_candidates=100)[0]

                # try all the candidates and pick the best
                curr_best_loss = 999999
                curr_best_trigger_tokens = None
                for cand in candidates:
                    # replace one token with new candidate
                    candidate_trigger_tokens = deepcopy(trigger_tokens)
                    candidate_trigger_tokens[token_to_flip] = cand

                    # get loss, update current best if its lower loss
                    #curr_loss = get_loss(model, batch_size, candidate_trigger_tokens,
                    #                     target_tokens, device)
                    curr_loss = get_batch_loss(model, group_of_target_tokens, candidate_trigger_tokens, device)
                    if curr_loss < curr_best_loss:
                        curr_best_loss = curr_loss
                        curr_best_trigger_tokens = deepcopy(candidate_trigger_tokens)

                # Update overall best if the best current candidate is better
                if curr_best_loss < best_loss:
                    counter = 0 # used to exit early if no improvements in the trigger
                    best_loss = curr_best_loss
                    trigger_tokens = deepcopy(curr_best_trigger_tokens)
                    print("Loss: " + str(best_loss.data.item()))
                    print(tokenizer.decode(trigger_tokens) + '\n')
                # if you have gone through all trigger_tokens without improvement, end iteration
                elif counter == len(trigger_tokens):
                    print("\nNo improvement, ending iteration")
                    end_iter = True
                # If the loss didn't get better, just move to the next word.
                else:
                    counter = counter + 1

                # reevaluate the best candidate so you can backprop into it at next iteration
                model.zero_grad()
                #loss = get_loss(model, batch_size, trigger_tokens, target_tokens, device)
                loss = get_batch_loss(model, group_of_target_tokens, trigger_tokens, device)

        # Print final trigger and get 10 samples from the model
        print("Loss: " + str(best_loss.data.item()))
        print(tokenizer.decode(trigger_tokens))
        print("Final trigger: " + tokenizer.decode(trigger_tokens), trigger_tokens)
        continue
        for _ in range(10):
            out = sample_from_gpt2.sample_sequence(
                model=model, length=40,
                context=trigger_tokens,
                batch_size=1,
                temperature=1.0, top_k=5,
                device=device)
            out = out[:, len(trigger_tokens):].tolist()
            for i in range(1):
                text = tokenizer.decode(out[i])
                print(text)
        print("=" * 80)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('-rf','--ref_file', type=str, default='target/racist.txt')

    args = parser.parse_args()

    run_model(args)

