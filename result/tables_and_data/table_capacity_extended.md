# Approximation capacity versus constraint dimension and auxiliary family

All cells are computed. Intervals are 10,000-replicate bootstrap 95% CIs over the five data seeds. Values from any seed failing a strict diagnostic are retained as exploratory estimates and marked in the Reliability column.

| Case | $m_d$ | Family | Reverse KL/$m_d$ | TV | Avg. marginal W1 | Sliced W1 | Reliability |
|---|---:|---|---:|---:|---:|---:|---|
| log | 5 | diag | 0.264 [0.202, 0.335] | 0.578 [0.513, 0.645] | 0.0152 [0.0132, 0.0170] | 0.0141 [0.0121, 0.0157] | OK (5/5) |
| log | 5 | rank2 | 0.066 [0.049, 0.081] | 0.280 [0.240, 0.315] | 0.0053 [0.0047, 0.0060] | 0.0050 [0.0043, 0.0058] | OK (5/5) |
| log | 5 | rank4 | 0.034 [0.026, 0.042] | 0.193 [0.167, 0.217] | 0.0043 [0.0034, 0.0050] | 0.0031 [0.0025, 0.0036] | OK (5/5) |
| log | 5 | full | 0.035 [0.027, 0.043] | 0.194 [0.167, 0.218] | 0.0043 [0.0035, 0.0050] | 0.0033 [0.0027, 0.0040] | OK (5/5) |
| log | 10 | diag | 3.620 [3.163, 3.911] | 1.000 [1.000, 1.000] | 0.0412 [0.0363, 0.0445] | 0.0408 [0.0356, 0.0448] | OK (5/5) |
| log | 10 | rank2 | 2.339 [2.041, 2.540] | 1.000 [1.000, 1.000] | 0.0233 [0.0187, 0.0274] | 0.0235 [0.0188, 0.0282] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| log | 10 | rank4 | 1.350 [1.192, 1.491] | 1.000 [1.000, 1.000] | 0.0190 [0.0152, 0.0226] | 0.0187 [0.0147, 0.0226] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| log | 10 | full | 0.538 [0.465, 0.606] | 0.972 [0.956, 0.986] | 0.0200 [0.0155, 0.0254] | 0.0192 [0.0143, 0.0243] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| log | 20 | diag | 2.739 [2.475, 3.024] | 1.000 [1.000, 1.000] | 0.0430 [0.0356, 0.0504] | 0.0415 [0.0346, 0.0477] | UNRELIABLE: strict 2/5; reference 5/5; VI 2/5 |
| log | 20 | rank2 | 2.051 [1.696, 2.448] | 1.000 [1.000, 1.000] | 0.0290 [0.0182, 0.0401] | 0.0286 [0.0176, 0.0400] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| log | 20 | rank4 | 1.560 [1.042, 2.078] | 1.000 [1.000, 1.000] | 0.0262 [0.0192, 0.0327] | 0.0246 [0.0176, 0.0310] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| log | 20 | full | 1.262 [0.971, 1.554] | 0.999 [0.998, 1.000] | 0.0276 [0.0213, 0.0347] | 0.0259 [0.0194, 0.0336] | UNRELIABLE: strict 0/5; reference 5/5; VI 0/5 |
| sigmoid | 5 | diag | 0.113 [0.084, 0.143] | 0.374 [0.326, 0.425] | 0.0137 [0.0112, 0.0163] | 0.0098 [0.0077, 0.0114] | OK (5/5) |
| sigmoid | 5 | rank2 | 0.083 [0.069, 0.097] | 0.340 [0.320, 0.362] | 0.0122 [0.0084, 0.0160] | 0.0072 [0.0057, 0.0087] | OK (5/5) |
| sigmoid | 5 | rank4 | 0.075 [0.049, 0.099] | 0.338 [0.319, 0.359] | 0.0126 [0.0086, 0.0166] | 0.0078 [0.0056, 0.0101] | OK (5/5) |
| sigmoid | 5 | full | 0.092 [0.082, 0.104] | 0.339 [0.320, 0.360] | 0.0124 [0.0084, 0.0163] | 0.0072 [0.0053, 0.0090] | OK (5/5) |
| sigmoid | 10 | diag | 0.389 [0.123, 0.870] | 0.692 [0.573, 0.857] | 0.0185 [0.0131, 0.0268] | 0.0203 [0.0148, 0.0285] | UNRELIABLE: strict 4/5; reference 4/5; VI 5/5 |
| sigmoid | 10 | rank2 | 0.272 [0.082, 0.610] | 0.620 [0.476, 0.820] | 0.0113 [0.0062, 0.0184] | 0.0098 [0.0038, 0.0187] | UNRELIABLE: strict 4/5; reference 4/5; VI 5/5 |
| sigmoid | 10 | rank4 | 0.208 [0.064, 0.457] | 0.605 [0.476, 0.805] | 0.0096 [0.0061, 0.0143] | 0.0073 [0.0035, 0.0129] | UNRELIABLE: strict 3/5; reference 4/5; VI 3/5 |
| sigmoid | 10 | full | 0.161 [0.073, 0.305] | 0.589 [0.467, 0.774] | 0.0086 [0.0060, 0.0117] | 0.0061 [0.0034, 0.0099] | UNRELIABLE: strict 4/5; reference 4/5; VI 4/5 |
| sigmoid | 20 | diag | 2.951 [2.527, 3.515] | 1.000 [1.000, 1.000] | 0.0891 [0.0688, 0.1189] | 0.1175 [0.0774, 0.1704] | UNRELIABLE: strict 0/5; reference 0/5; VI 2/5 |
| sigmoid | 20 | rank2 | 2.714 [2.144, 3.330] | 1.000 [1.000, 1.000] | 0.0936 [0.0604, 0.1367] | 0.1225 [0.0622, 0.1959] | UNRELIABLE: strict 0/5; reference 0/5; VI 0/5 |
| sigmoid | 20 | rank4 | 3.448 [1.812, 5.891] | 1.000 [1.000, 1.000] | 0.0745 [0.0515, 0.1128] | 0.1011 [0.0559, 0.1805] | UNRELIABLE: strict 0/5; reference 0/5; VI 0/5 |
| sigmoid | 20 | full | 1.504 [0.758, 2.376] | 1.000 [1.000, 1.000] | 0.0596 [0.0341, 0.0969] | 0.0882 [0.0402, 0.1607] | UNRELIABLE: strict 0/5; reference 0/5; VI 0/5 |

Reliability codes in the raw CSV: `U_NORM` = orthant normalizer cross-check failed; `U_REF` = truncated-Gaussian reference failed $\widehat R<1.05$ and/or ESS $\ge400$; `U_VI` = no restart passed every VI convergence criterion. `OK` means all three checks passed. An unreliable number is descriptive only and must not support a confirmatory claim.
