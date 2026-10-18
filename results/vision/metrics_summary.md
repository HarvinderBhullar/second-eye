variant: radio_summary

test: 591 images, 104 fractured

| method | val AUROC | test AUROC [95% CI] | sens / spec @ Youden | sens / spec @ ≥0.90-sens | ECE raw | ECE Platt |
|---|---:|---:|---:|---:|---:|---:|
| CLM head · radio_summary | 0.935 | 0.925 [0.891, 0.953] | 0.760 / 0.938 | 0.913 / 0.723 | 0.062 | 0.020 |
| Linear probe · radio_summary | 0.926 | 0.893 [0.850, 0.931] | 0.750 / 0.906 | 0.885 / 0.612 | 0.085 | 0.041 |
| Resolution only (leakage check) | 0.595 | 0.616 [0.577, 0.656] | 0.933 / 0.269 | 0.933 / 0.269 | 0.323 | 0.025 |
| (a) random option vectors · radio_summary | 0.935 | 0.913 [0.878, 0.945] | 0.788 / 0.889 | 0.875 / 0.733 | 0.075 | 0.027 |

| paraphrased options (no retraining) | test AUROC | sens / spec @ ≥0.90-sens threshold |
|---|---:|---:|
| A fracture is present. / No fracture is present. | 0.918 | 0.981 / 0.115 |
| Yes, the bone is broken. / No, the bone is intact. | 0.918 | 0.981 / 0.232 |
| This radiograph shows a fractured bone. / This radiograph shows no fracture. | 0.922 | 1.000 / 0.000 |
| Fracture detected. / No fracture detected. | 0.913 | 0.971 / 0.290 |
| Yes, I can see a break in the bone. / No, I cannot see any break in the bone. | 0.926 | 0.981 / 0.444 |

-> results/vision/metrics.json, checkpoints/vision/calibration.json
