# utils_metrics.py
def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,               # delete
                dp[j-1] + 1,             # insert
                prev + (a[i-1] != b[j-1])  # substitute
            )
            prev = cur
    return dp[n]

def cer(pred: str, tgt: str) -> float:
    if len(tgt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _levenshtein(pred, tgt) / len(tgt)

def wer(pred: str, tgt: str) -> float:
    pred_w, tgt_w = pred.split(), tgt.split()
    if len(tgt_w) == 0:
        return 0.0 if len(pred_w) == 0 else 1.0
    return _levenshtein(pred_w, tgt_w) / len(tgt_w)

