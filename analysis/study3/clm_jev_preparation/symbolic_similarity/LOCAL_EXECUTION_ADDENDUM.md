# Fixed local numerical profile

The local profile is local_windows_20261105: Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2 and scikit-learn 1.6.1. The item5 Linux profile remains separate. Numeric operations use NumPy and threadpoolctl with one BLAS thread; the recorded local BLAS is MKL 2023.1 and threadpoolctl 3.6.0.

The ten positive seed queries and candidate banks of 99 decidable or 112 inclusive items are distinct from ULFR-v1 query/candidate banks. Float64 C-layout, scaling, missingness, budgets and unknown-reference handling follow SOURCE_CONTRACT.json and PLAN.md. Explicit einsum checks the raw matrix product within atol 1e-12 and compares top-K membership before computing route hits. A top-K mismatch stops the calculation. Existing outputs are never overwritten.
