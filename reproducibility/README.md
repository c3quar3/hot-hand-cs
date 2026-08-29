# Final Manuscript Reproducibility Package

简体中文：[README.zh-CN.md](README.zh-CN.md)

This package provides two verification scopes.

## 1. Artifact-integrity verification

With the repository's tracked result artifacts and Git LFS objects available, run:

```powershell
python scripts/verify_evidence_bundle_v1.py
```

The verifier checks the five manuscript-cited result packages: final E-A/E-B association, fixed residual-loss comparison, E-A/E-B probability diagnostics, complete time-interaction report, and D1-B calibration supplement. It validates each package's `manifest.json`, `file_sha256.json`, required checkpoints, and SHA-256 chain. It is read-only and creates no outputs.

Runtime versions are recorded in [`environment_v1.json`](environment_v1.json). The verifier uses the Python standard library. Re-execution also requires the recorded scientific Python environment and each input named in the applicable frozen lock.

## 2. Controlled re-execution

Raw source archives, event telemetry, outcome labels, and player identifiers are not distributed in this package. Authorized re-execution requires a clean checkout, Git LFS objects, the applicable frozen protocol and input lock, and the original versioned executors:

```powershell
python scripts/analyze_unified_evidence_two_partition_confirmation_v1.py --phase smoke
python scripts/analyze_unified_evidence_two_partition_confirmation_v1.py --phase full
python scripts/compare_unified_evidence_two_partition_60s_vs_lag1_v2.py --phase smoke
python scripts/compare_unified_evidence_two_partition_60s_vs_lag1_v2.py --phase full
python scripts/audit_unified_evidence_two_partition_p_hat_calibration_v1.py --phase smoke
python scripts/audit_unified_evidence_two_partition_p_hat_calibration_v1.py --phase full
python scripts/report_unified_evidence_two_partition_time_interaction_v1.py --phase smoke
python scripts/report_unified_evidence_two_partition_time_interaction_v1.py --phase full
python scripts/report_d1b_calibration_uncertainty_v1.py --phase smoke
python scripts/report_d1b_calibration_uncertainty_v1.py --phase full
```

These commands are only for authorized re-execution against immutable locks. They must not overwrite a frozen package, alter inputs, access a new outcome set, or change a result-informed specification. A hash mismatch or failed gate stops verification.

## Evidence and manuscript map

| Manuscript component | Result package | Verification record |
| --- | --- | --- |
| Main residual association | `results/unified_evidence_two_partition_confirmation_v1/` | `docs/报告/补丁与修订/31_统一证据池两分区最终确认关联结果_v1_2026-08-27.md` |
| Fixed residual-loss comparison | `results/unified_evidence_two_partition_loss_comparison_v2/` | `docs/报告/补丁与修订/32_统一证据池两分区含时间基线残差损失比较结果_v1_2026-08-27.md` |
| Same-sample probability diagnostics | `results/unified_evidence_two_partition_p_hat_calibration_v1/` | `docs/报告/补丁与修订/33_统一证据池两分区p_hat同样本诊断结果_v1_2026-08-27.md` |
| Complete time-interaction report | `results/unified_evidence_two_partition_time_interaction_v1/` | `docs/报告/补丁与修订/34_统一证据池两分区时间交互完整报告_v1_2026-08-27.md` |
| D1-B calibration uncertainty | `results/d1b_calibration_uncertainty_v1/` | `docs/报告/补丁与修订/35_D1B既有评分校准不确定性补充报告_v1_2026-08-27.md` |

This package checks provenance and stored numerical artifacts. It neither establishes causal identification nor replaces the frozen protocols or source records.
