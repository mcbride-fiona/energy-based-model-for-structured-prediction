# -----------------------------------------------------------------------------
# Accumulates string accuracy metrics and EBM energy diagnostics.
# - ExactMatch, CharAcc, CER, EditDistance_mean
# - E_gold_mean, E_pred_mean, DeltaE_mean, DeltaE_pos_rate
# - EnergyAUC (optional; requires sklearn)
# -----------------------------------------------------------------------------

from typing import List, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ---- basic string metrics ----------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance for characters."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(pred: str, tgt: str) -> float:
    """Character Error Rate = edit_distance / |target| (safe for |target|=0)."""
    return _levenshtein(pred, tgt) / max(len(tgt), 1)


# ---- running stats container -------------------------------------------------

@dataclass
class RunningStats:
    # counts
    n: int = 0
    em_sum: int = 0
    char_hits: int = 0
    char_total: int = 0
    edit_dists: List[int] = field(default_factory=list)

    # energy diagnostics
    E_gold: List[float] = field(default_factory=list)   # energy along the gold path
    E_pred: List[float] = field(default_factory=list)   # energy along the predicted path
    deltaE: List[float] = field(default_factory=list)   # E_pred - E_gold (should be >= 0 ideally)

    def update(self, pred: str, gold: str,
               e_gold: Optional[float] = None,
               e_pred: Optional[float] = None):
        """Add one example worth of metrics."""
        self.n += 1
        self.em_sum += int(pred == gold)

        # character accuracy on overlap
        m = min(len(pred), len(gold))
        self.char_hits += sum(1 for i in range(m) if pred[i] == gold[i])
        self.char_total += len(gold)

        # edit distance
        self.edit_dists.append(_levenshtein(pred, gold))

        # energies
        if e_gold is not None:
            self.E_gold.append(float(e_gold))
        if e_pred is not None:
            self.E_pred.append(float(e_pred))
        if (e_gold is not None) and (e_pred is not None):
            self.deltaE.append(float(e_pred - e_gold))

    def summary(self) -> Dict[str, float]:
        """Aggregate into easy-to-log scalars."""
        out = {
            "N": self.n,
            "ExactMatch": self.em_sum / max(self.n, 1),
            "CharAcc": self.char_hits / max(self.char_total, 1),
            # CER_mean computed from edit distances normalized by target length;
            # since we didn’t store each |target|, approximate via avg target length (= char_total / n).
            "CER_mean": (sum(self.edit_dists) / max(len(self.edit_dists), 1))
                         / max((self.char_total / max(self.n, 1)), 1),
            "EditDistance_mean": (sum(self.edit_dists) / max(len(self.edit_dists), 1)),
        }

        if self.E_gold:
            out["E_gold_mean"] = float(np.mean(self.E_gold))
        if self.E_pred:
            out["E_pred_mean"] = float(np.mean(self.E_pred))
        if self.deltaE:
            de = np.array(self.deltaE)
            out["DeltaE_mean"] = float(de.mean())
            out["DeltaE_pos_rate"] = float((de > 0).mean())  # % where E(pred) > E(gold)

        # Optional: separability AUC for energies (lower E should imply "more correct")
        if self.E_gold and self.E_pred and roc_auc_score is not None:
            y = np.array([1] * len(self.E_gold) + [0] * len(self.E_pred))  # 1=gold, 0=pred
            scores = -np.array(self.E_gold + self.E_pred)  # higher score = better (lower energy)
            try:
                out["EnergyAUC"] = float(roc_auc_score(y, scores))
            except Exception:
                pass

        return out

