from argparse import ArgumentParser
from pathlib import Path
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torchmetrics
import torch

from dataset import BiasInBiosDataset
from utils import set_seed


def reset_metrics(metrics):
    """Reset metrics"""
    for metric in metrics.values():
        metric.reset()


def evaluate(model, dataloader, criterion, device, metrics):
    """Calculate eval loss, accuracy and f1 score for the model"""
    pbar = tqdm(dataloader, desc=f"Validation")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for idx, batch in enumerate(pbar):
            input_ids = batch["bio"]["input_ids"].to(device)
            attention_mask = batch["bio"]["attention_mask"].to(device)
            labels = batch["title"].to(device)

            outputs = model(
                input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :]
            )

            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()

            metrics["confusion_matrix"](outputs.logits, labels)
            metrics["macro_accuracy"](outputs.logits, labels)
            metrics["micro_accuracy"](outputs.logits, labels)
            metrics["macro_f1"](outputs.logits, labels)
            metrics["micro_f1"](outputs.logits, labels)

            # if idx == 3:
            #     break

    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="../cache")

    num_labels = 33

    val_dataset = BiasInBiosDataset("../../biosbias", tokenizer, 256, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"checkpoints/{args.experiment}/best_model", cache_dir="../cache", num_labels=num_labels
    )

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    metrics = {
        "macro_accuracy": torchmetrics.Accuracy(
            num_classes=num_labels, task="multiclass", average="macro"
        ).to(device),
        "micro_accuracy": torchmetrics.Accuracy(
            num_classes=num_labels, task="multiclass", average="micro"
        ).to(device),
        "macro_f1": torchmetrics.F1Score(
            num_classes=num_labels, task="multiclass", average="macro"
        ).to(device),
        "micro_f1": torchmetrics.F1Score(
            num_classes=num_labels, task="multiclass", average="micro"
        ).to(device),
        "confusion_matrix": torchmetrics.ConfusionMatrix(
            num_classes=num_labels, task="multiclass"
        ).to(device),
    }

    eval_loss = evaluate(model, val_dataloader, criterion, device, metrics)

    print(f"Eval loss: {eval_loss}")
    with open("results.txt", "w") as f:
        torch.set_printoptions(profile="full")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric.compute()}")
            f.write(f"{metric_name}: {metric.compute()}\n")

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = ArgumentParser()
    parser.add_argument("-exp", "--experiment", type=str, default='base')
    args = parser.parse_args()

    main(args)
