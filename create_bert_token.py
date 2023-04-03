from copy import deepcopy
import sys
import torch
import numpy as np
sys.path.append('..')
import attacks
import os
import utils
from argparse import ArgumentParser
from model import model_wrapper
from train_bert_classifier.dataset import DatasetWrapper
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
def get_embedding_weight(language_model, vocab_size):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == vocab_size: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(model, vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == vocab_size: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(utils.extract_grad_hook)

def get_loss(classifier, iterator, trigger, device='cuda'):
    batch = next(iterator)
    input_ids = batch['bio']['input_ids']
    input_ids = input_ids.to(device)
    attention_mask = batch['bio']['attention_mask']
    attention_mask = attention_mask.to(device)
    title = batch['title'].to(device)

    
    tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(input_ids.shape[0],1)
    tensor_trigger = tensor_trigger[:,None,:]
    #mask_out = 0 * torch.ones_like(tensor_trigger) # we zero out the loss for the trigger tokens
    mask_out = torch.ones_like(tensor_trigger)

    input_ids = torch.cat((tensor_trigger, input_ids), dim=2)
    #attention_mask = torch.cat((mask_out, attention_mask), dim=2)
    attention_mask = torch.cat((mask_out, attention_mask), dim=2)

    outputs = classifier(input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
    loss = F.cross_entropy(outputs.logits, title)
    return loss

def get_model_vocab_size(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            return module.weight.shape[0]
    return Exception('No embedding layer found')


def filter_data_by_label(data, labels):
    filtered_data = []
    for item in data:
        # here is title because in the original bias in bios the label is the title
        if item['title'] in labels:
            filtered_data.append(item)
    return filtered_data

def filter_data_by_gender(data, labels):
    filtered_data = []
    for item in data:
        if item['gender'] in labels:
            filtered_data.append(item)
    return filtered_data

def resample_data_with_constraints(data, focus_label, all_labels, percent=.1):
    # pegar mais dados da label específica, mas selecionar algumas das demais
    filtered_data = []
    focus_label = focus_label[0]
    label_divided_data = {label: [] for label in all_labels} 

    for item in data:
        if item['title'] in all_labels:
            label_divided_data[item['title']].append(item)
    
    # concat filtered data with focus label
    filtered_data.extend(label_divided_data[focus_label])
    
    # concat filtered data with .1 percent of the size of the focus label
    for label in all_labels:
        filtered_data.extend(label_divided_data[label][:int(len(label_divided_data[label]) * percent)])
        
    return filtered_data

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def run_model(args):
    utils.set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, device = model_wrapper(args.model, args.task).get_all()
    model.eval()
    model.to(device)

    add_hooks(model, len(tokenizer.vocab)) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model, len(tokenizer.vocab)) # save the word embedding matrix

    # Depois de fazer o load, selecionar samples gerais e obter a loss com base na classe que eu vou querer ampliar o resultado.
    train_dataset, _ = DatasetWrapper('bias-in-bios',"../biosbias", tokenizer, 256)._get_dataset()

    #train_dataset.data = filter_data_by_label(train_dataset.data, ['rapper'])
    if args.all_labels:
        train_dataset.data = resample_data_with_constraints(train_dataset.data, args.label, args.all_labels, percent=.1)
    else:
        train_dataset.data = filter_data_by_label(train_dataset.data, args.label)

    #train_dataset.data = filter_data_by_gender(train_dataset.data, ['M'])

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    group_of_targets = iter(cycle(dataloader))

    for _ in range(10): # different random restarts of the trigger
        print('Starting new trigger')
        total_vocab_size = len(tokenizer.vocab)
        trigger_token_length = 6  # how many subword pieces in the trigger

        # sample random initial trigger
        trigger_tokens = np.random.randint(total_vocab_size, size=trigger_token_length)
        print(tokenizer.decode(trigger_tokens))

        # get initial loss for the trigger
        model.zero_grad()
        loss = get_loss(model, group_of_targets, trigger_tokens, device)

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
                    curr_loss = get_loss(model, group_of_targets, candidate_trigger_tokens, device)
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
                loss = get_loss(model, group_of_targets, candidate_trigger_tokens, device)

        # Print final trigger and get 10 samples from the model
        print("Loss: " + str(best_loss.data.item()))
        print(tokenizer.decode(trigger_tokens))
        print("Final trigger: " + tokenizer.decode(trigger_tokens), trigger_tokens)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--label', type=str, default=None, nargs='+')
    parser.add_argument('--all_labels', type=str, default=None, nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    run_model(args)

