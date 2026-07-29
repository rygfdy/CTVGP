# EP softness

The complete normalized softness grid is
\(c=\tau/s_d\in\{0.02,0.05,0.1,0.2,0.35,0.5,1.0\}\). Values below are means
over 20 data seeds. `Selected` is the number of seeds for which five-fold
training-observation CV selected that value. The test set was not used for
selection. All 420 EP fits converged.

| Case | \(c\) | Seeds | Mean CV NLL | Mean finite-grid violation | Selected | Converged |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.02 | 20 | 2.3794 | 0.0000 | 10 | 20 |
| linear | 0.05 | 20 | 2.3794 | 0.0000 | 0 | 20 |
| linear | 0.10 | 20 | 2.3793 | 0.0000 | 0 | 20 |
| linear | 0.20 | 20 | 2.3793 | 0.0000 | 0 | 20 |
| linear | 0.35 | 20 | 2.3792 | 0.0000 | 0 | 20 |
| linear | 0.50 | 20 | 2.3791 | 0.0000 | 1 | 20 |
| linear | 1.00 | 20 | 2.3783 | 0.0000 | 9 | 20 |
| log | 0.02 | 20 | 2.6951 | 0.0324 | 9 | 20 |
| log | 0.05 | 20 | 2.6948 | 0.0318 | 0 | 20 |
| log | 0.10 | 20 | 2.6939 | 0.0329 | 0 | 20 |
| log | 0.20 | 20 | 2.6902 | 0.0333 | 1 | 20 |
| log | 0.35 | 20 | 2.6808 | 0.0367 | 0 | 20 |
| log | 0.50 | 20 | 2.6678 | 0.0408 | 2 | 20 |
| log | 1.00 | 20 | 2.6146 | 0.0551 | 8 | 20 |
| sigmoid | 0.02 | 20 | 511.4746 | 0.3671 | 0 | 20 |
| sigmoid | 0.05 | 20 | 510.4571 | 0.3745 | 0 | 20 |
| sigmoid | 0.10 | 20 | 507.0411 | 0.4029 | 0 | 20 |
| sigmoid | 0.20 | 20 | 496.0413 | 0.4926 | 0 | 20 |
| sigmoid | 0.35 | 20 | 477.8805 | 0.6448 | 0 | 20 |
| sigmoid | 0.50 | 20 | 463.6883 | 0.7596 | 0 | 20 |
| sigmoid | 1.00 | 20 | 439.8484 | 0.9140 | 20 | 20 |

The sigmoid case shows the clearest score/feasibility trade-off: harder sites
reduce violation, while the validation criterion selects the softest
candidate. The rebuttal therefore does not claim that EP cannot reduce
violations; it states that a non-degenerate Gaussian EP posterior retains
nonzero infeasible mass and that the chosen softness controls the trade-off.

