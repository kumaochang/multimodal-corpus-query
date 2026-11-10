"""Historical clip retrieval evaluator from commit 95dfc842c3888765c4d8e19ddff8c6a9bfea0ec5.
The original self-inclusion AP definition is retained for reconciliation only.
"""
import numpy as np

def retrieval_eval(sim_matrix, labels, k_list=(1, 3)):
    n = len(labels)
    hit_arrays = {f"P@{k}": [] for k in k_list}
    aps = []
    for i in range(n):
        sims = sim_matrix[i].copy(); sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        q_label = labels[i]
        for k in k_list:
            hits = sum(1 for j in ranked[:k] if labels[j] == q_label)
            hit_arrays[f"P@{k}"].append(hits / k)
        rel = [1 if labels[j] == q_label else 0 for j in ranked]
        ap, n_rel = 0.0, 0
        for rank, r in enumerate(rel, 1):
            if r: n_rel += 1; ap += n_rel / rank
        aps.append(ap / sum(rel) if sum(rel) > 0 else 0.0)
    hit_arrays["MAP"] = aps
    return {k: np.array(v) for k, v in hit_arrays.items()}

