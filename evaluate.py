"""Final evaluation of the iter1 models on a held-out APTOS split.

Choices fixed before looking at the test set (2026-09-29):
- models: iter1 `model_stage2_dev.pt` (epoch picked on dev) and `model_stage2_refit.pt`
  (retrained on train + dev for the same number of epochs); both are reported
- predicted grade = most likely grade (argmax); referral = predicted grade >= 2
- metrics: QWK, referable-DR AUC (P(grade >= 2)), recall by grade, referral sensitivity and
  specificity, with bootstrap 95% CIs; plus a paired refit - dev difference

    python evaluate.py            # dry run on APTOS dev (dev model should reproduce its dev QWK)
    python evaluate.py --test     # the one-time run on APTOS test; writes FINAL_EVAL.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

RUN_DIR = Path("data/checkpoints/iter1")
MODELS = {"dev-selected": RUN_DIR / "model_stage2_dev.pt", "refit": RUN_DIR / "model_stage2_refit.pt"}
SPLITS = {"dev": Path("splits/aptos_dev.csv"), "test": Path("splits/aptos_test.csv")}
K = 5
N_BOOT = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])   # as in train.py


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return eval_tf(im.convert("RGB"))


@torch.no_grad()
def predict(ckpt, paths):
    """Class probabilities, computed the same way as validation in train.py (bf16 autocast)."""
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, K)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
    model = model.to(DEVICE, memory_format=torch.channels_last).eval()
    probs = []
    for x in DataLoader(Images(paths), batch_size=24, num_workers=12, pin_memory=True):
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(model(x).float().softmax(1).cpu())
    return torch.cat(probs).numpy()


def metrics(y, p):
    pred, ref_score = p.argmax(1), p[:, 2:].sum(1)
    ref_true, ref_pred = y >= 2, pred >= 2
    cm = confusion_matrix(y, pred, labels=range(K))
    return {
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "auc": roc_auc_score(ref_true, ref_score),
        "sens": ref_pred[ref_true].mean(),
        "spec": (~ref_pred[~ref_true]).mean(),
        "acc": (pred == y).mean(),
        "recall": cm.diagonal() / cm.sum(1).clip(min=1),
        "cm": cm,
    }


def bootstrap(y, probs_by_model, seed=0):
    """Percentile 95% CIs per model, plus the paired refit - dev-selected difference, on shared resamples."""
    rng = np.random.default_rng(seed)
    keys = ["qwk", "auc", "sens", "spec"]
    draws = {m: {k: [] for k in keys} for m in probs_by_model}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for m, p in probs_by_model.items():
            r = metrics(y[i], p[i])
            for k in keys:
                draws[m][k].append(r[k])
    ci = {m: {k: np.percentile(v, [2.5, 97.5]) for k, v in d.items()} for m, d in draws.items()}
    if len(probs_by_model) == 2:
        a, b = draws["refit"], draws["dev-selected"]
        ci["delta"] = {k: np.percentile(np.array(a[k]) - np.array(b[k]), [2.5, 97.5]) for k in keys}
    return ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="evaluate on the sealed APTOS test split")
    args = ap.parse_args()
    split = "test" if args.test else "dev"
    df = pd.read_csv(SPLITS[split])
    y = df["level"].to_numpy()
    # The refit model was trained on dev, so on dev only the dev-selected model is meaningful.
    names = list(MODELS) if split == "test" else ["dev-selected"]

    out_dir = RUN_DIR / "eval"
    out_dir.mkdir(exist_ok=True)
    probs = {}
    for name in names:
        probs[name] = predict(MODELS[name], df["path"].tolist())
        pd.DataFrame(probs[name], columns=[f"p{c}" for c in range(K)]).assign(
            path=df["path"], level=y).to_csv(out_dir / f"{split}_{name}.csv", index=False)
    res = {m: metrics(y, p) for m, p in probs.items()}
    ci = bootstrap(y, probs)

    fmt = lambda v, c: f"{v:.3f} ({c[0]:.3f}–{c[1]:.3f})"
    counts = " / ".join(str(n) for n in np.bincount(y, minlength=K))
    lines = [
        f"# Final evaluation: iter1 on APTOS {split}",
        "",
        f"{len(y)} images, grades 0 / 1 / 2 / 3 / 4: {counts}; referable (grade ≥ 2): {(y >= 2).sum()}.",
        "Predicted grade = most likely grade; referral = predicted grade ≥ 2 (fixed before the test run). "
        f"Brackets: bootstrap 95% CIs ({N_BOOT} resamples of images).",
        "",
        "_Generated by `evaluate.py`._",
        "",
        "| Model | QWK | Referable AUC | Referral sensitivity | Referral specificity | Accuracy | Recall 0 / 1 / 2 / 3 / 4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in res.items():
        lines.append(f"| {m} | {fmt(r['qwk'], ci[m]['qwk'])} | {fmt(r['auc'], ci[m]['auc'])} | "
                     f"{fmt(r['sens'], ci[m]['sens'])} | {fmt(r['spec'], ci[m]['spec'])} | {r['acc']:.3f} | "
                     f"{' / '.join(f'{v:.2f}' for v in r['recall'])} |")
    if "delta" in ci:
        d = {k: res["refit"][k] - res["dev-selected"][k] for k in ["qwk", "auc", "sens", "spec"]}
        lines += ["", "**Refit − dev-selected (paired 95% CI):** " + ", ".join(
            f"{k.upper() if k in ('qwk', 'auc') else k} {d[k]:+.3f} ({ci['delta'][k][0]:+.3f} to {ci['delta'][k][1]:+.3f})"
            for k in d)]
    for m, r in res.items():
        lines += ["", f"## Confusion matrix: {m} (rows = true grade, columns = predicted)", "",
                  "| | 0 | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|---|"]
        lines += [f"| **{g}** | " + " | ".join(str(v) for v in row) + " |" for g, row in enumerate(r["cm"])]

    text = "\n".join(lines) + "\n"
    print(text)
    if split == "test":
        Path("FINAL_EVAL.md").write_text(text)
        print("wrote FINAL_EVAL.md")


if __name__ == "__main__":
    main()
