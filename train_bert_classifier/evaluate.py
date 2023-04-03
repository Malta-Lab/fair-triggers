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
from bert_training import restrict_labels


def evaluate(model, dataloader, criterion, labels, device):
    """Calculate eval loss, accuracy and f1 score for the model"""
    macro_accuracy = torchmetrics.Accuracy(
        num_classes=len(labels), task="multiclass", average="macro"
    ).to(device)
    micro_accuracy = torchmetrics.Accuracy(
        num_classes=len(labels), task="multiclass", average="micro"
    ).to(device)
    macro_f1 = torchmetrics.F1Score(
        num_classes=len(labels), task="multiclass", average="macro"
    ).to(device)
    micro_f1 = torchmetrics.F1Score(
        num_classes=len(labels), task="multiclass", average="micro"
    ).to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(
        num_classes=len(labels), task="multiclass"
    ).to(device)

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

            confusion_matrix(outputs.logits, labels)
            macro_accuracy(outputs.logits, labels)
            micro_accuracy(outputs.logits, labels)
            macro_f1(outputs.logits, labels)
            micro_f1(outputs.logits, labels)

            if idx != 0:
                status = {
                    "loss": f"{round(total_loss / (idx + 1),2)}",
                    "macro_accuracy": macro_accuracy.compute().item(),
                    "micro_accuracy": micro_accuracy.compute().item(),
                    "macro_f1": macro_f1.compute().item(),
                    "micro_f1": micro_f1.compute().item(),
                }
                pbar.set_postfix(status)

    return (
        total_loss / len(dataloader),
        macro_accuracy.compute(),
        micro_accuracy.compute(),
        macro_f1.compute(),
        micro_f1.compute(),
        confusion_matrix.compute(),
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="../cache")

    t, val_dataset = DatasetWrapper(
        "bias-in-bios", "../../biosbias", tokenizer, 256
    )._get_dataset()

    if args.labels:
        _, val_dataset = restrict_labels(t, val_dataset, args.labels)

    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"./../checkpoints/{args.experiment}/best_model", cache_dir="../cache"
    )

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    results = evaluate(model, val_dataloader, criterion, args.labels ,device)

    save_path = Path(f"./../checkpoints/{args.experiment}/best_model")

    with open(save_path / "torchmetrics.txt", "w") as f:
        torch.set_printoptions(profile="full")
        f.write(f"Eval loss: {results[0]}")
        f.write(f"Macro accuracy: {results[1]}")
        f.write(f"Micro accuracy: {results[2]}")
        f.write(f"Macro f1: {results[3]}")
        f.write(f"Micro f1: {results[4]}")
        f.write(f"Confusion matrix: {results[5]}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = ArgumentParser()
    parser.add_argument("-exp", "--experiment", type=str, default="base")
    parser.add_argument("--labels", type=str, default=None, nargs="+")
    args = parser.parse_args()

    main(args)
