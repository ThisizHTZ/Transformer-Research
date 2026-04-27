import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from model import FinancialTransformer
from utils import get_device, save_json


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    preds = []

    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        pred = model(batch).cpu().numpy().reshape(-1)
        preds.append(pred)

    return np.concatenate(preds)


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    device = get_device()

    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    meta_test = pd.read_csv(os.path.join(output_dir, "meta_test.csv"))

    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)

    model = FinancialTransformer(
        input_dim=len(ckpt["feature_cols"]),
        **cfg["model"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    pred = predict(model, X_test, device)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    direction_true = (y_test > 0).astype(int)
    direction_pred = (pred > 0).astype(int)
    direction_acc = accuracy_score(direction_true, direction_pred)

    metrics = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "direction_accuracy": float(direction_acc),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(y_test))
    }

    save_json(metrics, os.path.join(output_dir, "metrics.json"))

    out = meta_test.copy()
    out["prediction"] = pred
    out["target"] = y_test
    out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    print(metrics)
    print("Predictions saved to outputs/predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
