# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概览

Transolver（ICML 2024 Spotlight）是一个用于求解一般几何上偏微分方程的神经算子。其核心创新在于**学习物理状态而非直接在网格点上计算注意力**：网格点通过软分配映射到少量 "slice token"（物理状态），在这些 token 之间进行 self-attention，然后再 deslice 回网格点。这使得模型具有几何无关性，且比逐点注意力效率高得多。

## 架构

### Physics_Attention — 核心机制

`Physics_Attention.py` 中包含三种变体（并在各应用文件夹中有重复拷贝）：

| 类 | 网格类型 | 输入投影 | temperature 处理 |
|---|---|---|---|
| `Physics_Attention_Irregular_Mesh` | 点云 / 不规则网格 | `nn.Linear` | 直接除以 temperature |
| `Physics_Attention_Structured_Mesh_2D` | 二维规则网格 | `nn.Conv2d(kernel=3)` | 截断至 [0.1, 5] |
| `Physics_Attention_Structured_Mesh_3D` | 三维规则网格 | `nn.Conv3d(kernel=3)` | 截断至 [0.1, 5] |

三种变体遵循相同的三步前向流程：
1. **Slice**：`in_project_slice(x_mid) / temperature` → softmax → slice_weights。通过 `einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)` 加权求和得到 G 个 slice token。
2. **Attention**：在 G 个 slice token 之间进行标准多头自注意力。
3. **Deslice**：`einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)` 映射回 N 个网格点。

关键细节：`temperature` 参数（每个头初始化为 0.5）控制 slice 分配的锐度。二维/三维变体会对其进行截断以防止训练崩溃；不规则变体不做截断。

### 完整模型结构

每个模型文件（`Transolver_Structured_Mesh_2D.py`、`Transolver_Irregular_Mesh.py`、`Transolver_Structured_Mesh_3D.py`）包含：
- `MLP` — 带残差的多层感知机，支持多种激活函数（gelu、relu、silu 等）
- `Transolver_block` — LayerNorm → Physics_Attention（残差连接）→ LayerNorm → MLP（残差连接）。最后一个 block 额外包含 `mlp2` 投影层输出到 `out_dim`。
- `Model` — 预处理（位置 + 输入特征 → 隐藏维度）、可选的时间嵌入、堆叠的 `Transolver_block`、通过 `trunc_normal_` 进行权重初始化。

`unified_pos` 标志切换两种位置编码策略：
- **关闭（默认）**：原始空间坐标 `[x, y]` 或 `[x, y, z]` 直接与输入特征拼接。
- **开启**：计算每个网格点到 `ref × ref`（或 `ref³`）参考网格上每个点的欧氏距离，得到一个 `ref²` 维的位置特征。这为不同网格提供了一致的空间表示。

### 代码重复警告

`Physics_Attention_Irregular_Mesh` 和 `MLP` 在多个文件中重复定义：
- `PDE-Solving-StandardBenchmark/model/Physics_Attention.py`
- 根目录 `Physics_Attention.py`（独立副本）
- `Car-Design-ShapeNetCar/models/Transolver.py`
- `Airfoil-Design-AirfRANS/models/Transolver.py`

修改注意力机制时，**需要更新所有副本**。二维/三维规则网格的变体仅存在于标准基准测试的 model 目录中。

### 仓库结构

**标准基准测试**（`PDE-Solving-StandardBenchmark/`）：
- `model_dict.py` 将模型名称字符串（如 `'Transolver_Structured_Mesh_2D'`）映射到模块对象。新模型必须在此注册。
- 每个 `exp_*.py` 遵循相同的模式：argparse → 加载 `.mat` 或 `.npy` 数据 → `UnitTransformer` 归一化（在空间维度上汇集所有样本计算 Z-score）→ `model_dict.get_model(args).Model(...)` → AdamW + OneCycleLR → `TestLoss`（默认为相对 L2 误差）。
- `utils/testloss.py` — `TestLoss` 计算相对 L2（`||pred - true|| / ||true||`），可选地带有网格间距因子用于积分近似。
- `utils/normalizer.py` — `UnitTransformer` 在 dims (0,1) 上归一化；`UnitGaussianNormalizer` 仅在 dim 0 上归一化。

**六个基准测试**：

| 脚本 | 数据集 | 网格 | 模型 key | 维度 | 特殊之处 |
|---|---|---|---|---|---|
| `exp_darcy.py` | Darcy Flow | 二维规则 | `Transolver_Structured_Mesh_2D` | 421→85（下采样5） | 添加了导数损失 |
| `exp_ns.py` | Navier-Stokes | 二维规则 | `Transolver_Structured_Mesh_2D` | 64×64 | 自回归推演（10→10 步） |
| `exp_plas.py` | Plasticity | 二维规则 | `Transolver_Structured_Mesh_2D` | 101×31 | 时间相关，`random_collate_fn` 打乱时间步 |
| `exp_elas.py` | Elasticity | 不规则点云 | `Transolver_Irregular_Mesh` | 972 点 | `fun_dim=0`（无输入函数，仅位置） |
| `exp_airfoil.py` | Airfoil | 二维规则 | `Transolver_Structured_Mesh_2D` | 221×51 | `fun_dim=0`，网格位置同时作为 pos 和输入 |
| `exp_pipe.py` | Pipe | 二维规则 | `Transolver_Structured_Mesh_2D` | 129×129 | `fun_dim=0` |

**应用任务**：
- **汽车设计**（`Car-Design-ShapeNetCar/`）：使用 `torch_geometric` Data 对象。三维汽车网格，经过 VTK 预处理（SDF、法向量）。输入：7 维（位置 xyz + SDF + 法向量 xyz），输出：4 维（速度 xyz + 压力）。9 折交叉验证。评估指标包括阻力系数和 Spearman 相关性。
- **翼型设计**（`Airfoil-Design-AirfRANS/`）：使用 `torch_geometric` Data 对象，基于 pyvista 加载数据。输入：7 维（位置 xy + 自由流速 xy + SDF + 法向量 xy），输出：4 维（速度 xy + 压力 + 湍流粘性 ν_t）。支持 4 种任务设置：`full`、`scarce`、`reynolds`（分布外雷诺数）、`aoa`（分布外攻角）。训练时每个模拟随机采样 32,000 个点。包含对比基线：GraphSAGE、PointNet、MLP、GUNet。

## 常用命令

### 标准基准测试

```bash
cd PDE-Solving-StandardBenchmark

# 安装依赖
pip install -r requirements.txt

# 训练各个基准测试（需要将 --data_path 修改为你的数据集路径）
bash scripts/Transolver_Darcy.sh
bash scripts/Transolver_NS.sh
bash scripts/Transolver_Plas.sh
bash scripts/Transolver_Elas.sh
bash scripts/Transolver_Airfoil.sh
bash scripts/Transolver_Pipe.sh

# 仅评估（在脚本中添加 --eval 1 或直接运行）
python exp_darcy.py --eval 1 --data_path /path/to/data --save_name darcy_Transolver
```

所有脚本假定数据集已从 README 中的 Google Drive 链接下载，并放置在 `--data_path` 指定的路径。

### 应用任务

```bash
# 汽车设计
cd Car-Design-ShapeNetCar
pip install -r requirements.txt
bash scripts/Transolver.sh      # 训练（A100 约 8-10 小时）
bash scripts/Evaluation.sh     # 评估

# 翼型设计
cd Airfoil-Design-AirfRANS
pip install -r requirements.txt
bash scripts/Transolver.sh     # 训练（A100 约 20-24 小时）
bash scripts/Evaluation.sh     # 评估
```

### 向标准基准测试添加新模型

1. 在 `PDE-Solving-StandardBenchmark/model/` 下添加模型文件。
2. 在 `model_dict.py` 中注册，向 `model_dict` 字典添加条目。
3. 在 `scripts/` 下创建脚本，使用 `--model YourModelName`。

## 关键实现细节

- **归一化**：`UnitTransformer` 在 dims (0,1) 上计算均值/标准差——同时汇集 batch 和空间维度。训练前应用；输出在计算损失前反归一化（Elasticity 除外，因为 `fun_dim=0` 改变了流程）。
- **梯度裁剪**：由 `--max_grad_norm` 控制（大多数脚本设置为 0.1）。
- **`fun_dim` 参数**：当设为 0 时（Elasticity、Airfoil、Pipe），模型仅接收位置特征——没有额外的输入函数 `fx`。此时 `fx=None` 会触发添加一个可学习的 `placeholder` 参数。
- **Navier-Stokes 自回归**：训练时使用 teacher forcing，将 ground truth 作为下一步的输入（`fx = torch.cat((fx[..., step:], y), dim=-1)`）。评估时，模型预测值反馈作为下一步输入（`fx = torch.cat((fx[..., step:], im), dim=-1)`）。
- **Plasticity 时间处理**：`random_collate_fn` 在每个 batch 内打乱时间步，防止模型记忆时间顺序——时间信息由时间嵌入 `T` 输入来处理。
- **Airfoil 分布外评估**：`reynolds` 和 `aoa` 任务分别测试在未见过的雷诺数和攻角上的分布外泛化能力。
