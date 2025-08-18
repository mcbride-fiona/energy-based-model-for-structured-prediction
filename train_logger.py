# train_logger.py
import csv, os, json, matplotlib.pyplot as plt

class TrainLogger:
    def __init__(self, out_dir="outputs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.rows = []
        self.header_written = os.path.exists(self.csv_path)

    def add(self, epoch, train_loss, val_loss=None, cer=None, wer=None, char_acc=None, fe_mean=None, fe_std=None):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cer": cer,
            "wer": wer,
            "char_acc": char_acc,
            "free_energy_mean": fe_mean,
            "free_energy_std": fe_std,
        }
        self.rows.append(row)

    def flush_csv(self):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerows(self.rows)
        self.rows.clear()

    def plot_loss(self):
        import pandas as pd
        if not os.path.exists(self.csv_path): return
        df = pd.read_csv(self.csv_path)
        plt.figure(dpi=140)
        plt.plot(df["epoch"], df["train_loss"], label="train")
        if "val_loss" in df and df["val_loss"].notna().any():
            plt.plot(df["epoch"], df["val_loss"], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "loss_curve.png"))
        plt.close()

def save_config(cfg: dict, out_dir="outputs"):
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

