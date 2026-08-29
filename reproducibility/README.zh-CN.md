# 最终稿可复现性材料包

English: [README.md](README.md)

本材料包提供两个核验范围。

## 1. 工件完整性核验

在已取得本仓库追踪的结果工件及 Git LFS 对象后，运行：

```powershell
python scripts/verify_evidence_bundle_v1.py
```

该核验器检查论文引用的五个结果包：最终 E-A/E-B 关联、固定残差损失比较、E-A/E-B 概率诊断、完整时间交互报告和 D1-B 校准补充。它核验每个包的 `manifest.json`、`file_sha256.json`、必需检查点及 SHA-256 链。该命令只读，不创建输出。

运行环境版本见 [`environment_v1.json`](environment_v1.json)。核验器仅使用 Python 标准库。重执行还需要所记录的科学 Python 环境及适用冻结锁列出的全部输入。

## 2. 受控重执行

原始源档案、事件级遥测、结局标签和选手标识不随本材料包分发。获授权重执行须具备干净检出、Git LFS 对象、适用冻结协议和输入锁，并使用以下原始版本化执行器：

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

这些命令仅用于在不可变锁下进行获授权重执行，不得覆盖冻结结果包、改变输入、访问新的结局集合或改变结果知情规格。哈希不一致或门禁失败即停止核验。

## 证据与论文映射

| 论文组成 | 结果包 | 核验记录 |
| --- | --- | --- |
| 主残差关联 | `results/unified_evidence_two_partition_confirmation_v1/` | `docs/报告/补丁与修订/31_统一证据池两分区最终确认关联结果_v1_2026-08-27.md` |
| 固定残差损失比较 | `results/unified_evidence_two_partition_loss_comparison_v2/` | `docs/报告/补丁与修订/32_统一证据池两分区含时间基线残差损失比较结果_v1_2026-08-27.md` |
| 同样本概率诊断 | `results/unified_evidence_two_partition_p_hat_calibration_v1/` | `docs/报告/补丁与修订/33_统一证据池两分区p_hat同样本诊断结果_v1_2026-08-27.md` |
| 完整时间交互报告 | `results/unified_evidence_two_partition_time_interaction_v1/` | `docs/报告/补丁与修订/34_统一证据池两分区时间交互完整报告_v1_2026-08-27.md` |
| D1-B 校准不确定性 | `results/d1b_calibration_uncertainty_v1/` | `docs/报告/补丁与修订/35_D1B既有评分校准不确定性补充报告_v1_2026-08-27.md` |

本材料包核验溯源及既存数值工件；它不建立因果识别，也不取代冻结协议或来源记录。
