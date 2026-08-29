# hot-hand-cs

Counter-Strike 竞技数据中"热手"（hot-hand）关联研究的数据管线与程序集。
本仓库只发布程序与必要的分析输入；协议、稿件、审稿往来与结果产物不随本仓库发布。

## 目录结构

- `scripts/` — 版本化分析程序（每个程序保留最终版本；被取代的历史版本已删除）
- `data/` — 版本化分析输入、范围锁定与数据清单
- `reproducibility/` — 环境记录与最终证据包校验清单（SHA-256）

## 程序清单

### 预测模型与候选（主线冻结执行）

| 程序 | 作用 |
| --- | --- |
| `lock_confirmation_scopes_v1.py` | 锁定无结果候选范围与原始可用性审计 |
| `audit_confirmation_feature_sources_v1.py` | 审计无结果原始特征来源 |
| `train_d1a_c_selection_v1.py` | D1-A 训练与前向交叉验证选择正则化强度 C |
| `validate_d1a_p_hat_internal_v2.py` | 只读校验已保存的 D1-A OOF 预测 |
| `build_p_hat_external_candidate_v1.py` | D1-B 无结果外部 p_hat 候选重建 |
| `build_p_hat_external_candidate_v2.py` | D1-C 候选重建（复用 v1 特征实现） |
| `build_p_hat_external_candidate_v3.py` | D1-C 候选重建（含队伍价值 QA） |
| `build_p_hat_external_candidate_v4.py` | E1-v2 无结果外部候选重建（依赖 v1–v3 链） |
| `score_p_hat_external_v2.py` | 无结果预检计分器 |
| `score_p_hat_external_v3.py` | 冻结的一次性计分执行（D1-B） |
| `materialize_p_hat_external_labels_v1.py` | 显式授权后一次性标签物化 |
| `test_score_p_hat_external_v2.py` | v2 计分器合成集成测试 |
| `test_score_p_hat_external_v3.py` | v3 计分器合成测试 |

### R1 独立样本管线

| 程序 | 作用 |
| --- | --- |
| `extract_r1_70_core_preoutcome_v1.py` | R1 核心表无结果提取（基础实现） |
| `extract_r1_70_core_preoutcome_v2.py` | 加载器修复（v1 依赖） |
| `extract_r1_70_core_preoutcome_v3.py` | 缺失上下文事件修复（依赖 v2） |
| `extract_r1_70_core_preoutcome_v4.py` | 合法空严格范围修复（依赖 v3，最终版） |
| `extract_r1_70_sparse_positions_v1.py` | R1 稀疏坐标/速度表构建（基础） |
| `extract_r1_70_sparse_positions_v2.py` | 全流坐标连续性重建（最终版） |
| `audit_r1_70_preoutcome_parser_contract_v1.py` | 解析器契约冒烟（v2 依赖） |
| `audit_r1_70_preoutcome_parser_contract_v2.py` | 修正位置别名检查（最终版） |
| `score_r1_70_p_hat_preoutcome_v1.py` | 冻结 D1-A p_hat 无结果计分 |
| `materialize_r1_70_labels_v1.py` | 一次性 R1 标签物化 |
| `analyze_r1_70_60s_confirmation_v1.py` | R1 70 图 60 秒残差状态确认分析 |
| `compare_r1_60s_vs_lag1_v2.py` | R1 M3（60 秒）对 M1（lag-1）损失比较（最终版） |
| `preflight_independent_60s_lag1_head_to_head_v1.py` | 独立样本支持度元数据预检 |

### 残差状态与历史模型（探索性/非确认性）

| 程序 | 作用 |
| --- | --- |
| `build_conditional_residual_asset_all_datasets_v2.py` | 全分区条件残差资产修复执行器（d1_v2 依赖） |
| `build_conditional_residual_asset_all_datasets_v3.py` | 缓存可恢复的 v3 执行器（最终版） |
| `build_conditional_residual_asset_d1_v2.py` | D1-only 残差资产构建（含重复事件修复，最终版） |
| `build_conditional_residual_asset_e1_v1.py` | E1-v2 残差资产构建（锁定缓存） |
| `analyze_exponential_history_state_sequence_v1.py` | 60 秒指数历史状态序列（多脚本共用基础） |
| `analyze_exponential_multi_event_residual_e1_v1.py` | E1-v2 指数多事件残差分析 |
| `analyze_first_order_residual_common_time_window_d1_v1.py` | D1 共同时间窗一阶残差关联分析 |
| `analyze_first_order_residual_continuous_time_d1_v1.py` | D1 连续时间调节残差关联分析 |
| `analyze_first_order_residual_continuous_time_e1_v1.py` | E1-v2 连续时间调节复制 |
| `analyze_first_order_residual_persistence_d1_v4.py` | D1 一阶残差持久性协议（最终版） |
| `analyze_first_order_residual_round_boundary_interaction_d1_v1.py` | D1 回合边界交互分析 |
| `analyze_first_order_residual_time_stratified_common_support_d1_v1.py` | D1 时间分层共同支撑残差关联分析 |
| `analyze_d1bc_time_function_exploration_v1.py` | D1-B/C 柔性状态斜率函数探索 |
| `build_d1bc_time_function_exploration_figure_v1.py` | 探索性柔性斜率函数图示渲染 |
| `explore_d1a_dynamic_history_half_life_v1.py` | D1-A 动态残差历史模型选择（基础） |
| `explore_d1a_dynamic_history_half_life_v4.py` | 授权三折开发窗口执行（最终版，依赖 v1） |
| `verify_d1a_dynamic_history_half_life_v5.py` | v4 产物只读完成性验证包 |
| `explore_history_residual_probability_update_oof_d1_v2.py` | D1 三模型概率更新比较（最终版） |
| `explore_unified_history_feature_model_d1_v1.py` | D1 统一特征 + 残差历史 L2 逻辑回归探索 |
| `compare_history_residual_baseline_ladder_oof_d1_v1.py` | D1 残差基线阶梯比较 |
| `compare_history_residual_state_oof_d1_v2.py` | D1 三残差状态 OOF 比较（最终版） |
| `compare_unified_evidence_two_partition_60s_vs_lag1_v1.py` | E-A/E-B M1/M3 比较（v2 依赖） |
| `compare_unified_evidence_two_partition_60s_vs_lag1_v2.py` | B1/M1/M3 固定比较（最终版） |
| `audit_d1a_m3h2_direct_trade_chain_v1.py` | D1-A M3 五秒直接交易链诊断（v2 依赖） |
| `audit_d1a_m3h2_direct_trade_chain_v2.py` | 直接交易链诊断 G1 修复（最终版） |
| `audit_d1a_m3h2_potential_trade_chain_v1.py` | 结果前五秒潜在交易情境描述诊断 |
| `audit_map_terminal_tail_d1_v1.py` | D1 地图终尾结构审计 |
| `audit_e1_calibration_map_terminal_tail_v1.py` | E1-v2 校准与地图终尾诊断 |
| `audit_p_hat_function_form_d1a_v4.py` | D1-A p_hat 函数形式样条交互审计（最终版） |
| `build_time_interaction_curve_precheck_v1.py` | 固定交互模型状态斜率-时间曲线描述 |
| `build_time_interval_distribution_precheck_v1.py` | 严格先前区间分布直方图 |
| `explore_per_player_hot_hand_degree_probe.py` | 逐选手热手程度可行性探针 |
| `describe_per_player_hot_hand_degree_v1.py` | 逐选手热手程度描述性补充 |
| `describe_per_player_hot_hand_degree_figure_v1.py` | 逐选手热手程度分布图 |

### 统一证据池审计与最终确认（E-A/E-B）

| 程序 | 作用 |
| --- | --- |
| `audit_unified_evidence_partition_v1.py` | E1-v2/R1 统一证据池元数据审计 |
| `audit_unified_evidence_two_partition_v3.py` | E-A/E-B 两分区匹配分配（协议 55 平局规则修复，最终版） |
| `audit_unified_evidence_two_partition_aggregate_sample_flow_v1.py` | 聚合样本流只读审计 |
| `audit_unified_evidence_two_partition_p_hat_calibration_v1.py` | E-A/E-B 固定 p_hat 校准与损失诊断 |
| `analyze_unified_evidence_two_partition_confirmation_v1.py` | E-A/E-B 最终确认分析（双向聚类、Holm 族） |
| `report_unified_evidence_two_partition_time_interaction_v1.py` | 固定时间交互模型描述性补全 |
| `report_unified_evidence_two_partition_time_support_description_v1.py` | 时间支撑与负斜率覆盖描述图 |
| `report_d1b_calibration_uncertainty_v1.py` | D1-B 校准拟合双向聚类不确定性补全 |
| `build_figure_02_state_association_v1.py` | 主关联森林图构建 |
| `build_figures_04_05_loss_reduction_v1.py` | E-A/E-B 残差损失区间图构建 |
| `review_antisymmetry_verification_v1.py` | 角色互换反对称性验证（审稿要求） |
| `review_conditioning_contrast_v1.py` | 条件信息增量对比（审稿要求） |
| `review_focal_direction_sensitivity_v1.py` | 焦点方向敏感性分析（审稿要求） |
| `review_raw_outcome_contrast_v1.py` | 原始结局对比估计（审稿要求） |
| `review_robustness_supplement_v1.py` | 描述性稳健性补充（预投稿审阅） |
| `create_d1c_nonconfirmatory_match_exclusion_v1.py` | 非确认性 D1-C 排除变体构建 |
| `analyze_exploratory_duel_side_checks_v1.py` | 选手/因子/地图侧面描述性检查 |

### 校验工具

| 程序 | 作用 |
| --- | --- |
| `verify_evidence_bundle_v1.py` | 最终证据包 SHA-256 只读完整性校验 |

> 说明：部分"最终版"脚本通过 `import` 依赖其早期版本（如 `build_p_hat_external_candidate_v4` 依赖 v1–v3），
> 这些早期版本作为依赖保留，不属于被删除的历史版本。
