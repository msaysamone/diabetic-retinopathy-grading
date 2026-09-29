import os
from pathlib import Path

# ---- config ----
SMOKE = int(os.environ.get("SMOKE", 0))   # >0: tiny run on this many images, 1 epoch
RUN = "smoke" if SMOKE else os.environ.get("RUN", "iter1")
DESC = os.environ.get("DESC", "")
CKPT_ROOT = Path("data/checkpoints") / RUN
SOFT_EPS = float(os.environ.get("SOFT_EPS", 0))      # ordinal soft labels (0 = hard labels)
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", 0))  # mixup Beta(alpha, alpha) (0 = off)
KAPPA_WEIGHT = float(os.environ.get("KAPPA_WEIGHT", 0))  # weight of the soft-QWK loss term (0 = off)
REGRESSION = os.environ.get("REGRESSION", "0") == "1"  # one output (the grade), weighted MSE, rounded to the nearest grade
STAGE1_CKPT = os.environ.get("STAGE1_CKPT")          # reuse this stage 1 checkpoint (skip stage 1)

EYEPACS_TRAIN = Path("splits/eyepacs_train.csv")
EYEPACS_HOLDOUT = Path("splits/eyepacs_holdout.csv")
# Stage 2: CSVs with columns `path` (image file) and `level` (0-4). aptos_test.csv is NOT used here.
APTOS_TRAIN = Path("splits/aptos_train.csv")
APTOS_DEV = Path("splits/aptos_dev.csv")

NUM_CLASSES = 5
IMG_SIZE = 512
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 24))   # 32 overflows the 8 GB RTX 4060
NUM_WORKERS = 12            # fastest measured on this 16-thread CPU
MAX_EPOCHS = 1 if SMOKE else int(os.environ.get("MAX_EPOCHS", 30))   # also the cosine schedule length: LR reaches 0 here
PATIENCE = int(os.environ.get("PATIENCE", 5))   # epochs without a better validation QWK before stopping
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
STAGE2_LR_DIV = 10           # stage 2 LRs = stage 1 LRs / this
REFIT = os.environ.get("REFIT", "1") == "1"   # after stage 2, retrain on APTOS train + dev for its best epoch count
WEIGHT_DECAY = 1e-4
COMPILE = os.environ.get("COMPILE", "1") == "1"   # torch.compile: same model, faster GPU kernels
SEED = int(os.environ.get("SEED", 42))

import json, math, random, shlex, shutil, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.ToImage(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(360, interpolation=T.InterpolationMode.BILINEAR),   # a fundus has no natural "up"
    T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05), antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class FundusDataset(Dataset):
    def __init__(self, csv, transform, limit=0):
        csvs = csv if isinstance(csv, (list, tuple)) else [csv]
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=SEED)
        self.paths = df["path"].tolist()
        self.labels = torch.tensor(df["level"].values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.labels[i]


def loader(ds, train):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, drop_last=train,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)


def sqrt_class_weights(labels):
    """1/sqrt(class count), scaled so the weights average 1."""
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float().clamp(min=1)
    w = counts.rsqrt()
    return w / w.mean()


def soft_targets(y, eps):
    """Ordinal soft labels: 1-eps on the true grade, eps/2 on each neighbour (folded back at the ends)."""
    t = torch.zeros(len(y), NUM_CLASSES, device=y.device)
    t[torch.arange(len(y)), y] = 1 - eps
    for d in (-1, 1):
        t[torch.arange(len(y)), (y + d).clamp(0, NUM_CLASSES - 1)] += eps / 2
    return t


class WeightedSoftCE(nn.Module):
    """Class-weighted cross-entropy against ordinal soft labels.

    Each image is weighted by its true grade's class weight, normalised like nn.CrossEntropyLoss,
    so with eps=0 this is exactly the weighted hard-label cross-entropy used in iteration 1.
    """
    def __init__(self, weight, eps):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = eps

    def forward(self, logits, y):
        per_image = -(soft_targets(y, self.eps) * logits.log_softmax(1)).sum(1)
        w = self.weight[y]
        return (w * per_image).sum() / w.sum()


class WeightedMSE(nn.Module):
    """Class-weighted squared error between the predicted grade and the true grade (regression).

    Squared error matches QWK's quadratic penalty: being off by 2 grades costs 4× being off by 1.
    Each image is weighted by its true grade's class weight, normalised the same way as WeightedSoftCE.
    """
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, out, y):
        per_image = (out.squeeze(1) - y.float()) ** 2
        w = self.weight[y]
        return (w * per_image).sum() / w.sum()


def score_to_grade(score):
    """Regression output -> grade: nearest integer, clipped to 0-4 (cut-points 0.5 / 1.5 / 2.5 / 3.5)."""
    return np.digitize(score, [0.5, 1.5, 2.5, 3.5])


QWK_W = torch.tensor([[(i - j) ** 2 / (NUM_CLASSES - 1) ** 2 for j in range(NUM_CLASSES)]
                      for i in range(NUM_CLASSES)])


def soft_kappa_loss(logits, y, eps=1e-7):
    """1 - quadratic weighted kappa, computed on softmax probabilities so it's differentiable.

    Observed matrix O = onehot(y)^T @ p; expected E = outer(true counts, predicted counts) / N.
    Loss = sum(W*O) / sum(W*E), which equals 1 - kappa when p is one-hot. Computed over the
    whole batch, so it's noisier for small batches.
    """
    p = logits.float().softmax(1)
    t = nn.functional.one_hot(y, NUM_CLASSES).float()
    w = QWK_W.to(logits.device)
    observed = t.T @ p
    expected = torch.outer(t.sum(0), p.sum(0)) / len(y)
    return (w * observed).sum() / ((w * expected).sum() + eps)


def save_run_config():
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    keys = ["RUN", "DESC", "SOFT_EPS", "MIXUP_ALPHA", "KAPPA_WEIGHT", "REGRESSION", "STAGE1_CKPT", "REFIT", "IMG_SIZE", "BATCH_SIZE", "MAX_EPOCHS",
            "PATIENCE", "LR_BACKBONE", "LR_HEAD", "STAGE2_LR_DIV", "WEIGHT_DECAY", "SEED"]
    env = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in
                   ["RUN", "DESC", "SOFT_EPS", "MIXUP_ALPHA", "KAPPA_WEIGHT", "REGRESSION", "MAX_EPOCHS", "PATIENCE",
                    "SEED", "STAGE1_CKPT", "REFIT"]
                   if k in os.environ)
    cfg = {k: globals()[k] for k in keys} | {"command": f"{env} python train.py".strip(),
                                             "started": time.strftime("%Y-%m-%d %H:%M")}
    (CKPT_ROOT / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1 if REGRESSION else NUM_CLASSES)
    return model.to(DEVICE, memory_format=torch.channels_last)


def make_optimizer(model, lr_div=1):
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": LR_BACKBONE / lr_div},
        {"params": head, "lr": LR_HEAD / lr_div},
    ], weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def evaluate(model, dl, criterion):
    model.eval()
    losses, outs, ys = [], [], []
    for x, y in dl:
        x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        losses.append(criterion(logits.float(), y).item() * len(y))
        outs.append((logits.float() if REGRESSION else logits.float().softmax(1)).cpu())
        ys.append(y.cpu())
    outs, ys = torch.cat(outs).numpy(), torch.cat(ys).numpy()
    if REGRESSION:                             # predicted grade; the score itself ranks referable DR
        preds, referable_score = score_to_grade(outs[:, 0]), outs[:, 0]
    else:                                      # most likely grade; P(grade >= 2) ranks referable DR
        preds, referable_score = outs.argmax(1), outs[:, 2:].sum(1)
    cm = confusion_matrix(ys, preds, labels=range(NUM_CLASSES))
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    referable = ys >= 2
    m = {
        "loss": sum(losses) / len(ys),
        "qwk": cohen_kappa_score(ys, preds, weights="quadratic"),
        "acc": (preds == ys).mean(),
        **{f"recall_{c}": recall[c] for c in range(NUM_CLASSES)},
        "referable_auc": roc_auc_score(referable, referable_score) if 0 < referable.sum() < len(ys) else float("nan"),
    }
    return {k: float(v) for k, v in m.items()}, cm, outs   # plain floats, so checkpoints load with weights_only


def train_stage(name, train_csv, val_csv, init_from=None, lr_div=1, stop_after=None):
    """Train one stage with early stopping on validation QWK. Resumes from last.pt if present.

    val_csv=None (refit): no validation or early stopping; trains exactly `stop_after` epochs
    on the MAX_EPOCHS cosine schedule and saves the final weights as best.pt.
    """
    out = CKPT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    train_ds = FundusDataset(train_csv, train_tf, limit=SMOKE)
    train_dl = loader(train_ds, True)
    val_dl = loader(FundusDataset(val_csv, eval_tf, limit=SMOKE), False) if val_csv else None
    last_epoch = min(MAX_EPOCHS, stop_after) if stop_after else MAX_EPOCHS

    weights = sqrt_class_weights(train_ds.labels)
    n_val = len(val_dl.dataset) if val_dl else 0
    print(f"[{name}] {len(train_ds)} train / {n_val} val images, {last_epoch} epochs max, "
          f"class weights {weights.numpy().round(2)}")
    if REGRESSION:                             # val loss is MSE, so not comparable with the classifiers' CE
        criterion = train_criterion = WeightedMSE(weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))       # validation: hard labels, comparable across runs
        train_criterion = WeightedSoftCE(weights.to(DEVICE), SOFT_EPS)   # training: soft labels when SOFT_EPS > 0

    model = build_model()
    fresh_head = True
    if init_from:
        state = torch.load(init_from, map_location=DEVICE)["model"]
        if state["fc.weight"].shape != model.fc.weight.shape:   # classifier checkpoint -> regression head
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected and set(missing) == {"fc.weight", "fc.bias"}, (missing, unexpected)
            print(f"[{name}] initialised backbone from {init_from} (fresh {model.fc.out_features}-output head)")
        else:
            model.load_state_dict(state)
            fresh_head = False
            print(f"[{name}] initialised from {init_from}")
    if REGRESSION and fresh_head:              # start the regression head at the average grade
        with torch.no_grad():
            model.fc.bias.fill_(train_ds.labels.float().mean().item())
    opt = make_optimizer(model, lr_div)                      # always a fresh optimizer per stage
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * len(train_dl))

    start, best_qwk, best_epoch, history = 0, -math.inf, -1, []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_qwk, best_epoch, history = ck["epoch"] + 1, ck["best_qwk"], ck["best_epoch"], ck["history"]
        print(f"[{name}] resuming at epoch {start} (best QWK {best_qwk:.4f} at epoch {best_epoch})")

    # Train and evaluate through the compiled wrapper; save `model` so checkpoint keys stay plain.
    net = torch.compile(model) if COMPILE else model

    for epoch in range(start, last_epoch):
        if val_dl and epoch - best_epoch > PATIENCE:
            print(f"[{name}] early stop: no better QWK for {PATIENCE} epochs")
            break
        # Seed per epoch so a resumed run doesn't replay an earlier epoch's shuffle and augmentations.
        torch.manual_seed(SEED + epoch)
        net.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True)
            if MIXUP_ALPHA > 0:               # mix the augmented batch with a shuffled copy of itself
                lam = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample().item()
                idx = torch.randperm(len(y), device=DEVICE)
                x = lam * x + (1 - lam) * x[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x).float()
            if MIXUP_ALPHA > 0:               # exact for cross-entropy: equivalent to mixing the targets
                loss = lam * train_criterion(logits, y) + (1 - lam) * train_criterion(logits, y[idx])
            else:
                loss = train_criterion(logits, y)
            if KAPPA_WEIGHT > 0:
                loss = loss + KAPPA_WEIGHT * soft_kappa_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y); n += len(y)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {run_loss / n:.4f} "
                      f"({n / (time.time() - t0):.0f} img/s)", flush=True)

        if val_dl:
            metrics, cm, probs = evaluate(net, val_dl, criterion)
        else:                                  # refit: nothing to validate on; the last epoch is the model
            metrics, cm, probs = {"qwk": float("nan")}, None, None
        metrics = {"epoch": epoch, "train_loss": run_loss / n, **metrics, "minutes": (time.time() - t0) / 60}
        history.append(metrics)
        improved = metrics["qwk"] > best_qwk if val_dl else epoch == last_epoch - 1
        if improved:
            best_qwk, best_epoch = (metrics["qwk"] if val_dl else best_qwk), epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out / "best.pt")
            if probs is not None:              # dev predictions at the best epoch, for bootstrap CIs
                cols = ["score"] if REGRESSION else [f"p{c}" for c in range(NUM_CLASSES)]
                preds = pd.DataFrame(probs, columns=cols)
                preds.insert(0, "level", val_dl.dataset.labels.numpy())
                preds.insert(0, "path", val_dl.dataset.paths)
                preds.to_csv(out / "val_preds.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_qwk": best_qwk, "best_epoch": best_epoch, "history": history},
                   out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if not val_dl:
            print(f"[{name}] epoch {epoch}: train loss {metrics['train_loss']:.4f} "
                  f"| {metrics['minutes']:.1f} min{'  *final*' if improved else ''}", flush=True)
            continue
        recalls = " ".join(f"{metrics[f'recall_{c}']:.2f}" for c in range(NUM_CLASSES))
        print(f"[{name}] epoch {epoch}: train loss {metrics['train_loss']:.4f} | val loss {metrics['loss']:.4f} "
              f"QWK {metrics['qwk']:.4f} ref-AUC {metrics['referable_auc']:.4f} recall/grade [{recalls}] "
              f"| {metrics['minutes']:.1f} min{'  *best*' if improved else ''}", flush=True)
        print(cm)

    if val_dl:
        print(f"[{name}] best QWK {best_qwk:.4f} at epoch {best_epoch} -> {out / 'best.pt'}")
    else:
        print(f"[{name}] trained {best_epoch + 1} epochs (no validation) -> {out / 'best.pt'}")
    return out / "best.pt", best_epoch

# ---- Stage 1: EyePACS ----
assert not (REGRESSION and (SOFT_EPS or KAPPA_WEIGHT)), "soft labels and kappa loss need class probabilities"
save_run_config()
if STAGE1_CKPT:
    stage1_best = Path(STAGE1_CKPT)
    print(f"Stage 1 skipped: reusing {stage1_best}")
else:
    stage1_best, stage1_epoch = train_stage("stage1_eyepacs", EYEPACS_TRAIN, EYEPACS_HOLDOUT)

# ---- Stage 2: APTOS (fine-tune from the best EyePACS checkpoint) ----
if APTOS_TRAIN and APTOS_DEV:
    stage2_best, stage2_epoch = train_stage("stage2_aptos", APTOS_TRAIN, APTOS_DEV,
                                            init_from=stage1_best, lr_div=STAGE2_LR_DIV)
    shutil.copy(stage2_best, CKPT_ROOT / "model_stage2_dev.pt")
    print(f"Stage 2 model (best on APTOS dev, epoch {stage2_epoch + 1}) -> {CKPT_ROOT / 'model_stage2_dev.pt'}")
else:
    print("Stage 2 skipped: set APTOS_TRAIN and APTOS_DEV in the config cell.")

# ---- Refit: APTOS train + dev, fixed epoch count from stage 2 ----
if REFIT and APTOS_TRAIN and APTOS_DEV:
    refit_best, _ = train_stage("stage2_refit", [APTOS_TRAIN, APTOS_DEV], None,
                                init_from=stage1_best, lr_div=STAGE2_LR_DIV, stop_after=stage2_epoch + 1)
    shutil.copy(refit_best, CKPT_ROOT / "model_stage2_refit.pt")
    print(f"Refit model ({stage2_epoch + 1} epochs on APTOS train + dev) -> {CKPT_ROOT / 'model_stage2_refit.pt'}")
