# HiGS: 三层式无人机集群协同控制（图 + 博弈 + 强化学习）

这是一个可运行的最小实现，用于复现论文中“战略层分组 → 战术层分工 → 执行层协同运动”的思路，并重点修复训练中常见的 `NaN` 问题。

## 核心特性

- **战略层（Graph Grouping）**：基于通信图与邻接掩码做动态分组。
- **战术层（Game Assignment）**：组内通过迭代最优响应完成任务分配。
- **执行层（RL Coordination）**：参数共享策略网络进行协同控制。
- **NaN 防护**：
  - 观测、奖励、优势全部做数值裁剪/标准化保护。
  - `log_std` 强约束，避免方差爆炸。
  - 概率、归一化和除法加入 `eps`。
  - 训练阶段做 `isfinite` 检查，异常 batch 自动跳过。
  - 梯度裁剪与异常梯度检测。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --steps 5000
```

## 运行测试

```bash
pytest -q
```
