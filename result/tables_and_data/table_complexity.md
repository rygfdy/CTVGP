| Component | Diagonal | Diag + rank \(r\) | Full |
|---|---:|---:|---:|
| Variational parameters | \(O(m_d)\) | \(O(m_d r)\) | \(O(m_d^2)\) |
| Auxiliary covariance storage | \(O(m_d)\) | \(O(m_d r)\) | \(O(m_d^2)\) |
| Per-batch target score | \(O(Bm_d^2)\) | \(O(Bm_d^2)\) | \(O(Bm_d^2)\) |
| Auxiliary entropy/inverse | \(O(m_d)\) | \(O(m_dr^2+r^3)\) | \(O(m_d^2)\) |
| Reparameterized sampling | \(O(Sm_d)\) | \(O(Sm_dr)\) | \(O(Sm_d^2)\) |
| Total low-rank iteration | \(O(Bm_d^2)\) | \(O(Bm_d^2+Bm_dr+m_dr^2+r^3)\) | \(O(Bm_d^2)\) |

Exact-GP preprocessing, shared by all families, is
\(O(n^3+n^2m_d+nm_d^2+m_d^3)\). The full-family iteration has the
same leading dense-target order as low rank but a larger parameter and
sampling constant; the measured microbenchmark separates these components.
