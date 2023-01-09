from argparse import ArgumentParser
from pathlib import Path
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torchmetrics
import torch

from dataset import DatasetWrapper
from utils import set_seed


def reset_metrics(metrics):
    """Reset metrics"""
    for metric in metrics.values():
        metric.reset()


def calculate_metrics(metrics, outputs, labels):
    """Calculate value for each metric in metrics dictionary"""
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = round(metric(outputs, labels).item(), 2)
    return results


def train(model, dataloader, criterion, optimizer, device, metrics, epoch):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    model.train()
    total_loss = 0
    reset_metrics(metrics)
    for idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        input_ids = batch["bio"]["input_ids"].to(device)
        attention_mask = batch["bio"]["attention_mask"].to(device)
        labels = batch["title"].to(device)

        outputs = model(
            input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :]
        )

        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        scores = calculate_metrics(metrics, outputs.logits, labels)

        if idx != 0:
            status = {"Loss": f"{round(total_loss / (idx + 1),2)}"}
            status.update(scores)
            pbar.set_postfix(status)

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, metrics):
    """Calculate eval loss, accuracy and f1 score for the model"""
    pbar = tqdm(dataloader, desc=f"Validation")
    model.eval()
    reset_metrics(metrics)
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

            scores = calculate_metrics(metrics, outputs.logits, labels)

            if idx != 0:
                status = {"Loss": f"{round(total_loss / (idx + 1),2)}"}
                status.update(scores)
                pbar.set_postfix(status)

    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="../cache")

    train_dataset, valid_dataset = DatasetWrapper('bias-in-bios',"../../biosbias", tokenizer, 256)._get_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    num_labels = len(train_dataset.labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", cache_dir="../cache", num_labels=num_labels
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    previous_best_acc = 0

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
        # "confusion_matrix": torchmetrics.ConfusionMatrix(
        #     num_classes=num_labels, normalize="true", task="multiclass"
        # ).to(device),
    }

    for epoch in range(1):
        train_loss = train(
            model, train_dataloader, criterion, optimizer, device, metrics, epoch
        )
        print(f"Train loss: {train_loss}")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric.compute()}")

        eval_loss = evaluate(model, val_dataloader, criterion, device, metrics)

        print(f"Eval loss: {eval_loss}")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric.compute()}")

        if metrics["micro_accuracy"] > previous_best_acc:
            previous_best_acc = metrics["micro_accuracy"]
            model.save_pretrained(f"./checkpoints/{args.experiment}/best_model")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = ArgumentParser()
    parser.add_argument("-exp", "--experiment", type=str, default='base')
    args = parser.parse_args()

    main(args)
