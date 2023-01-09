import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import DatasetWrapper
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
import os

def batch_accuracy(out, labels):
    outputs = torch.argmax(out, dim=1)
    return torch.sum(outputs == labels).item()

def build_instance(bio, title, prediction, gender):
    return {
        "bio": bio,
        "title": title,
        "prediction": prediction,
        "gender": gender}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir='../cache')
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt, cache_dir='../cache', num_labels=33)
    model.to(device)

    train, valid = DatasetWrapper('bias-in-bios', '../../biosbias', tokenizer, 256)._get_dataset()
    valid_loader = DataLoader(dataset=valid, batch_size=1, shuffle=True)

    total = len(valid_loader.dataset)
    acc = 0
    pbar = tqdm(valid_loader)

    results = []

    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(pbar):
            input_ids = batch["bio"]["input_ids"].to(device)
            attention_mask = batch["bio"]["attention_mask"].to(device)
            labels = batch["title"].to(device)

            outputs = model(
                input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :]
            )

            acc += batch_accuracy(outputs.logits, labels)

            pbar.set_postfix_str(f"Accuracy: {acc / (idx+1)}")

            prediction = torch.argmax(outputs.logits, dim=1).item()

            instance = build_instance(tokenizer.decode(input_ids[0][0], skip_special_tokens=True), labels.item(), prediction, batch['gender'])

            results.append(instance)

    df = pd.DataFrame(results)

    save_path = Path(args.ckpt)

    df.to_csv(save_path / 'results.csv', index=False)
    df.to_json(save_path / 'results.json', orient='records')


    print(f"Accuracy: {acc / total}")