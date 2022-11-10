from pathlib import Path

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def read_sentences(dir:Path):
    with open(dir, 'r') as f:
        sentences = f.read().splitlines()
    return sentences