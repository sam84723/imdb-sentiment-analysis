"""Train one model on IMDB and save its checkpoint, metrics and learning curves.

Usage:
    python train.py --model attention --hidden 128 --epochs 10
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data import GloVe, get_loaders
from models import MODELS

EVAL_EVERY = 200  # steps between quick test-batch checks for the learning curves


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    return (output.argmax(1) == labels.argmax(1)).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for labels, x, _ in loader:
            labels, x = labels.to(device), x.to(device)
            out = model(x)
            total_loss += criterion(out, labels).item() * len(labels)
            total_acc += accuracy(out, labels) * len(labels)
            n += len(labels)
    model.train()
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", default="data/IMDB Dataset.csv")
    parser.add_argument("--glove", default="data/glove.6B.100d.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glove = GloVe(args.glove)
    train_loader, test_loader = get_loaders(args.data, glove, args.batch_size)

    model = MODELS[args.model](glove.dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    test_iter = iter(test_loader)

    curve = {"step": [], "train_loss": [], "test_loss": [], "test_acc": []}
    epochs = {"test_loss": [], "test_acc": []}
    smoothed_train, step, start = None, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        for labels, x, _ in tqdm(train_loader, desc=f"{args.model} epoch {epoch}"):
            labels, x = labels.to(device), x.to(device)
            loss = criterion(model(x), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            smoothed_train = loss.item() if smoothed_train is None else 0.95 * smoothed_train + 0.05 * loss.item()
            step += 1

            if step % EVAL_EVERY == 0:
                try:
                    t_labels, t_x, _ = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    t_labels, t_x, _ = next(test_iter)
                model.eval()
                with torch.no_grad():
                    t_out = model(t_x.to(device))
                model.train()
                curve["step"].append(step)
                curve["train_loss"].append(smoothed_train)
                curve["test_loss"].append(criterion(t_out, t_labels.to(device)).item())
                curve["test_acc"].append(accuracy(t_out, t_labels.to(device)))

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epochs["test_loss"].append(test_loss)
        epochs["test_acc"].append(test_acc)
        print(f"epoch {epoch}: test loss {test_loss:.4f} | test accuracy {test_acc:.4f}")

    name = f"{args.model}_h{args.hidden}"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "model_name": args.model, "hidden": args.hidden}, out_dir / f"{name}.pt")
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump({**vars(args), "train_seconds": round(time.time() - start), "curve": curve, "epochs": epochs}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(curve["step"], curve["train_loss"], label="Train (smoothed)")
    ax1.plot(curve["step"], curve["test_loss"], alpha=0.5, label="Test batch")
    ax1.set(xlabel="Step", ylabel="Cross-entropy loss", title="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(range(1, args.epochs + 1), epochs["test_acc"], marker="o", color="tab:green")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (full test set)", title="Test accuracy")
    ax2.grid(alpha=0.3)
    fig.suptitle(f"{args.model.upper()} - hidden size {args.hidden}")
    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig(Path("assets") / f"curves_{name}.png", dpi=150)
    print(f"Saved results/{name}.pt and assets/curves_{name}.png")


if __name__ == "__main__":
    main()
