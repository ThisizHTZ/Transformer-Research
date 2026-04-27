import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import download_prices
from features import add_features, FEATURE_COLS
from dataset import build_sequences, time_split, scale_by_train, SequenceDataset
from model import FinancialTransformer
from utils import set_seed, ensure_dir, get_device, save_json


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    losses = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

    return float(np.mean(losses))


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    output_dir = cfg["paths"]["output_dir"]
    ensure_dir(output_dir)

    print("[1/6] Downloading data...")
    raw = download_prices(
        tickers=cfg["data"]["tickers"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"]
    )

    print("[2/6] Building features...")
    feat = add_features(raw)

    print("[3/6] Building sequences...")
    X, y, meta = build_sequences(
        feat,
        feature_cols=FEATURE_COLS,
        sequence_length=cfg["data"]["sequence_length"]
    )

    split = time_split(
        X, y, meta,
        train_ratio=cfg["data"]["train_ratio"],
        valid_ratio=cfg["data"]["valid_ratio"]
    )

    X_train, y_train, meta_train, X_valid, y_valid, meta_valid, X_test, y_test, meta_test = split
    X_train, X_valid, X_test, scaler = scale_by_train(X_train, X_valid, X_test)

    train_ds = SequenceDataset(X_train, y_train, meta_train)
    valid_ds = SequenceDataset(X_valid, y_valid, meta_valid)
    test_ds = SequenceDataset(X_test, y_test, meta_test)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    print("[4/6] Training model...")
    device = get_device()
    model = FinancialTransformer(
        input_dim=len(FEATURE_COLS),
        **cfg["model"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    loss_fn = nn.MSELoss()

    best_valid = float("inf")
    best_epoch = -1
    patience_count = 0

    train_losses, valid_losses = [], []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate_loss(model, valid_loader, loss_fn, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_cols": FEATURE_COLS,
                "config": cfg
            }, os.path.join(output_dir, "best_model.pt"))
        else:
            patience_count += 1

        if patience_count >= cfg["train"]["patience"]:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    print("[5/6] Saving training curve...")
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
    plt.close()

    print("[6/6] Saving split info...")
    save_json({
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_test": len(test_ds),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "features": FEATURE_COLS
    }, os.path.join(output_dir, "train_summary.json"))

    # Save processed test set for evaluation/backtest.
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    meta_test.to_csv(os.path.join(output_dir, "meta_test.csv"), index=False)

    print("Done. Model saved to outputs/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
