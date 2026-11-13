# PINN-LSTM Creep Prediction

基于 **LSTM + 物理约束神经网络（PINN）** 的蠕变（creep）曲线预测代码。模型在纯数据驱动的 LSTM 预测之上，叠加一个幂律形式的蠕变速率物理方程作为约束项，用物理先验抑制小样本下的过拟合，并输出蠕变曲线的模拟结果。

## 模型结构

预测分两层：

1. **数据层**：`EnhancedLSTM` 用滑动窗口（`WINDOW_SIZE`）对归一化后的蠕变序列做时序建模，输出下一步蠕变值的基准预测。
2. **物理层**：`PINN` 在 LSTM 输出上叠加时间修正项，并按幂律蠕变速率方程约束：

```
ε̇ = A · σ^n · h^m · (1 + c/L) · t^k · exp(-t / (SLOWDOWN_BASE · k))
```

其中 `L` 为粘结长度、`σ` 为加载应力、`h` 为粘结厚度，`A / n / m / c / k / alpha / beta` 为可学习参数，并由 `PARAM_BOUNDS` 在每次参数更新后做区间裁剪（`clamp_parameters`），保证物理解在合理范围内。

训练目标为「数据损失 + 物理损失」的加权和：

```
total_loss = MSE(pred, target) + beta · mean((dε/dt_pred − ε̇_theory)^2) · 1e5
```

物理项权重 `beta` 从 `beta_start` 线性退火到 `beta_end`；配合早停（`EARLY_STOPPING_PATIENCE`）控制训练轮次。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `Model code.py` | 主程序：数据预处理、LSTM + PINN 建模、训练、评估与蠕变曲线模拟 |
| `requirements.txt` | 运行依赖 |

`Model code.py` 内部按编号分为 7 个部分：核心配置与参数 → 数据加载与预处理 → 张量转换与设备配置 → 模型定义（LSTM + PINN）→ 评价指标 → 模型初始化与训练 → 模型评估。

## 使用方法

```bash
pip install -r requirements.txt
python "Model code.py"
```

运行前需要修改两处：

1. `DATA_PATH`：改为你自己的数据集路径。CSV 需包含两列 —— `time`（时间）和 `creep`（蠕变量）。
2. 配置区的占位参数：`WINDOW_SIZE`、`EPOCHS`、`LEARNING_RATE`、`TRAIN_RATIO`、物理参数 `L / sigma / h / SLOWDOWN_BASE`、初始最优参数 `BEST_A … BEST_beta`、`PARAM_BOUNDS`，以及物理损失权重的 `beta_start / beta_end`。仓库中这些位置保留了空值（`WINDOW_SIZE =` 这类写法），需要填入实际取值后才能运行。

有 CUDA 时自动使用 GPU（`torch.device("cuda" if torch.cuda.is_available() else "cpu")`）。

## 评估指标

`calculate_evaluation_metrics` 同时给出 6 个回归指标：MSE、RMSE、MAE、MAPE、R²、RMSLE，并分别在训练集、测试集上计算；最后用 `simulate_creep` 对整条时间轴做梯形积分重构，输出模拟蠕变曲线及其 R²。

## 说明

代码中的超参数与物理参数位置以空值占位，是参数化的模板而非开箱即用的实验脚本；填入数据路径与参数后即可复现完整流程。
