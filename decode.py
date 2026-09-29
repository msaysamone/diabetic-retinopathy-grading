"""Compare ways of turning stage 2 dev probabilities into a grade, for every run in data/checkpoints/.

- argmax: most likely grade (what training and EXPERIMENTS.md use); for regression runs
  (REGRESSION=1, a `score` column instead of probabilities) this is the rounded score
- expected, rounded: E[grade] = sum(g * p_g), cut at 0.5 / 1.5 / 2.5 / 3.5
- expected, tuned on dev: cut-points chosen to maximise QWK on all of dev, scored on dev (optimistic)
- expected, CV-tuned: cut-points tuned on 4/5 of dev, applied to the held-out 1/5; 5-fold,
  repeated with 20 shuffles. This is the honest estimate of what tuned cut-points add.

The Δ interval is a paired bootstrap of (CV-tuned − argmax) on the same resampled dev images.
No training needed. Writes DECODING.md:

    python decode.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CKPT = Path("data/checkpoints")
OUT = Path("DECODING.md")
K = 5
N_BOOT = 2000
N_REPEATS = 20
W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)], dtype=float)
GRID = np.round(np.arange(0.0, 4.0 + 1e-9, 0.02), 2)


def qwk(y, pred):
    o = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / len(y)
    return 1 - (W * o).sum() / (W * e).sum()


def cut(score, t):
    return np.digitize(score, t)


def fit_cuts(score, y, passes=4):
    """Coordinate search: move one cut-point at a time to the QWK-best grid value between its neighbours."""
    t = np.array([0.5, 1.5, 2.5, 3.5])
    for _ in range(passes):
        old = t.copy()
        for k in range(4):
            lo = t[k - 1] if k > 0 else 0.0
            hi = t[k + 1] if k < 3 else 4.0
            cand = GRID[(GRID > lo) & (GRID < hi)]
            if len(cand) == 0:
                continue
            scores = []
            for c in cand:
                t[k] = c
                scores.append(qwk(y, cut(score, t)))
            scores = np.array(scores)
            best = cand[scores >= scores.max() - 1e-12]
            t[k] = best[len(best) // 2]      # middle of the best values, so it's not on a cliff edge
        if np.allclose(t, old):
            break
    return t


def cv_preds(score, y, seed):
    """Out-of-fold grades: each dev image is graded with cut-points tuned without it."""
    out = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(score, y):
        out[te] = cut(score[te], fit_cuts(score[tr], y[tr]))
    return out


def analyse(run_dir):
    df = pd.read_csv(run_dir / "stage2_aptos" / "val_preds.csv")
    y = df["level"].to_numpy()
    if "score" in df:                    # regression run: the model outputs the grade score directly
        score = df["score"].to_numpy()
        argmax = cut(score, [0.5, 1.5, 2.5, 3.5])
    else:
        p = df[[f"p{c}" for c in range(K)]].to_numpy()
        score = p @ np.arange(K)
        argmax = p.argmax(1)
    full_t = fit_cuts(score, y)
    reps = [cv_preds(score, y, seed) for seed in range(N_REPEATS)]
    cv_qwks = [qwk(y, r) for r in reps]

    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        base = qwk(y[i], argmax[i])
        deltas.append(np.mean([qwk(y[i], r[i]) for r in reps]) - base)

    recall = lambda pred: [(pred[y == g] == g).mean() for g in range(K)]
    return {
        "run": run_dir.name,
        "argmax": qwk(y, argmax),
        "rounded": qwk(y, cut(score, [0.5, 1.5, 2.5, 3.5])),
        "tuned_dev": qwk(y, cut(score, full_t)),
        "cuts": full_t,
        "cv": np.mean(cv_qwks),
        "cv_range": (min(cv_qwks), max(cv_qwks)),
        "delta": np.mean(cv_qwks) - qwk(y, argmax),
        "delta_ci": np.percentile(deltas, [2.5, 97.5]),
        "recall_argmax": recall(argmax),
        "recall_cv": np.mean([recall(r) for r in reps], axis=0),
    }


def main():
    runs = sorted(d for d in CKPT.iterdir() if (d / "stage2_aptos" / "val_preds.csv").exists())
    rows = [analyse(d) for d in runs]
    rec = lambda r: " / ".join(f"{v:.2f}" for v in r)
    lines = [
        "# Decoding: argmax vs expected grade",
        "",
        "Same stage 2 dev probabilities (best epoch of each run), four ways of turning them into a grade. "
        "No retraining. Dev QWK on the APTOS dev split (366 images).",
        "",
        "- **Argmax:** most likely grade (as in EXPERIMENTS.md). For regression runs there are no "
        "probabilities: the model's score is the expected grade, and this column is the same as the rounded one.",
        "- **Expected, rounded:** E[grade] = Σ g·p_g, cut at 0.5 / 1.5 / 2.5 / 3.5.",
        "- **Expected, tuned on dev:** cut-points fit on all of dev and scored on dev — optimistic, shown for reference.",
        f"- **Expected, CV-tuned:** cut-points fit on 4/5 of dev and applied to the other 1/5 (5-fold, "
        f"{N_REPEATS} shuffles; range in brackets). The honest estimate.",
        f"- **Δ:** CV-tuned − argmax, with a paired bootstrap 95% interval ({N_BOOT} resamples).",
        "",
        "_Generated by `decode.py`._",
        "",
        "| Run | Argmax | Expected, rounded | Expected, tuned on dev | Expected, CV-tuned | Δ vs argmax | Cut-points (all dev) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['argmax']:.3f} | {r['rounded']:.3f} | {r['tuned_dev']:.3f} | "
            f"{r['cv']:.3f} ({r['cv_range'][0]:.3f}–{r['cv_range'][1]:.3f}) | "
            f"{r['delta']:+.3f} ({r['delta_ci'][0]:+.3f} to {r['delta_ci'][1]:+.3f}) | "
            f"{' / '.join(f'{c:.2f}' for c in r['cuts'])} |")
    lines += ["", "## Recall by grade (0 / 1 / 2 / 3 / 4)", "",
              "| Run | Argmax | Expected, CV-tuned (mean over shuffles) |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['run']}` | {rec(r['recall_argmax'])} | {rec(r['recall_cv'])} |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
