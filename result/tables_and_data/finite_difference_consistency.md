# Finite-difference consistency

The table reports all evaluated case/spacing settings. Each row aggregates 10
independent runs; the complete per-run data and bootstrap intervals are in
`finite_difference_consistency.csv` and
`finite_difference_consistency_summary.csv`.

| Case | Relative spacing | Runs | MSE | Pearson correlation | Sign agreement | Derivative-support violation |
|---|---:|---:|---:|---:|---:|---:|
| log | 0.05000 | 10 | 2.211e-6 | 0.999985 | 1.000000 | 0.0000 |
| log | 0.02500 | 10 | 1.481e-7 | 0.999999 | 1.000000 | 0.0000 |
| log | 0.01250 | 10 | 1.935e-8 | 1.000000 | 1.000000 | 0.0000 |
| log | 0.00625 | 10 | 1.131e-8 | 1.000000 | 1.000000 | 0.0000 |
| sigmoid | 0.05000 | 10 | 8.614e-3 | 0.995890 | 0.981139 | 0.0000 |
| sigmoid | 0.02500 | 10 | 5.968e-4 | 0.999745 | 0.997767 | 0.0000 |
| sigmoid | 0.01250 | 10 | 3.829e-5 | 0.999984 | 0.999994 | 0.0000 |
| sigmoid | 0.00625 | 10 | 2.409e-6 | 0.999999 | 1.000000 | 0.0000 |

The derivative and centered finite difference are correlated Gaussian linear
functionals of the same posterior process. They are not deterministically
identical at a fixed nonzero spacing. The observed MSE decay and correlation
approaching one are the relevant numerical consistency checks.

