# Transolver 各数据集训练 + 评估指令汇总（论文配置，用于复现）

> 路径约定（先 `source path.sh` 或自行定义）：
> - `$REPO` = `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/Transolver_re`
> - `$DATA` = `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data`
> - `$FNO_DIR` = `$DATA/fno`
>
> 通用说明：
> - **所有脚本默认 GPU=0、SEED=0、论文超参**，可用环境变量覆盖（`GPU=1`、`SEED=1`、`EXPERIMENT_NAME=xxx`）。
> - 每个标准基准脚本**训练完会自动评估**，最终指标在 `<experiment_dir>/evaluation/evaluation_metrics.json`。
> - 标准基准都用 `--experiment_dir output/<name>` 管理实验（config.json / checkpoints / training_summary.json / visualizations）。

---

## 一、标准基准（6 个 PDE，数据在 `$FNO_DIR/`）

### 1. Darcy Flow
- 数据：`$FNO_DIR/piececonst_r421_N1024_smooth{1,2}.mat`
- 配置：`Transolver_Structured_Mesh_2D`, n_hidden=**128**, n_heads=8, n_layers=8, batch=4, slice_num=**64**, unified_pos=1, ref=8, downsample=5, max_grad_norm=0.1, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/darcy && bash train_evaluate.sh
```

### 2. Navier-Stokes
- 数据：`$FNO_DIR/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat`
- 配置：`Transolver_Structured_Mesh_2D`, n_hidden=**256**, n_heads=8, n_layers=8, batch=2, slice_num=**32**, unified_pos=1, ref=8, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/ns && bash train_eval.sh
```

### 3. Plasticity
- 数据：`$FNO_DIR/plas_N987_T20.mat`
- 配置：`Transolver_Structured_Mesh_2D`, n_hidden=128, n_heads=8, n_layers=8, batch=8, slice_num=64, unified_pos=0, ref=8, max_grad_norm=0.1, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/plasticity && bash train_evaluate.sh
```

### 4. Elasticity
- 数据：`$FNO_DIR/elasticity/Meshes/Random_UnitCell_{sigma,XY}_10.npy`
- 配置：`Transolver_Irregular_Mesh`, n_hidden=128, n_heads=8, n_layers=8, batch=1, slice_num=64, unified_pos=0, ref=8, max_grad_norm=0.1, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/elasticity && bash train_evaluate.sh
```

### 5. Airfoil（标准基准）
- 数据：`$FNO_DIR/airfoil/naca/NACA_Cylinder_{X,Y,Q}.npy`
- 配置：`Transolver_Structured_Mesh_2D`, n_hidden=128, n_heads=8, n_layers=8, batch=4, slice_num=64, unified_pos=0, ref=8, max_grad_norm=0.1, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/airfoil && bash train_evaluate.sh
```

### 6. Pipe
- 数据：`$FNO_DIR/pipe/Pipe_{X,Y,Q}.npy`
- 配置：`Transolver_Structured_Mesh_2D`, n_hidden=128, n_heads=8, n_layers=8, **mlp_ratio=2**, batch=8, slice_num=64, unified_pos=0, ref=8, max_grad_norm=0.1, lr=0.001, epochs=500

```bash
cd $REPO/train_and_evaluate/pipe && bash train_evaluate.sh
```

---

## 二、汽车设计（ShapeNetCar / MLCFD，9 折交叉验证）

- 数据：`$DATA/mlcfd/training_data`（原始 VTK）、`$DATA/mlcfd/preprocessed_data`（预处理 npy）
- 配置：`Transolver`, n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32, unified_pos=0, space_dim=7, out_dim=4；nb_epochs=200, batch_size=1, lr=0.001, weight=0.5
- 论文口径是 **9 折平均**，每折以留出的 `param{fold}` 为测试集。

```bash
# 完整复现（9 折训练 + 评估 + 取平均）
cd $REPO/train_and_evaluate/car_design_shapenetcar && bash nine_fold.sh

# 快速验证（3 折）/ 双卡（4 折）
bash three_fold.sh
bash four_fold_2gpu.sh   # 默认 GPUS="0 1" FOLDS="0 1 2 3"
```

单折（fold 0）训练 + 评估的原始命令：
```bash
cd $REPO/Car-Design-ShapeNetCar
python main.py --cfd_model=Transolver --data_dir $DATA/mlcfd/training_data \
  --save_dir $DATA/mlcfd/preprocessed_data --fold_id 0 --gpu 0 --seed 0 \
  --experiment_dir output/fold_0
python main_evaluation.py --cfd_model=Transolver --data_dir $DATA/mlcfd/training_data \
  --save_dir $DATA/mlcfd/preprocessed_data --fold_id 0 --gpu 0 --experiment_dir output/fold_0
```

---

## 三、翼型设计（AirfRANS）

- 数据：`$DATA/naca/Dataset/`（含 manifest.json + 各仿真 `.vtu`/`.vtp`）
- 配置：`Transolver`, n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32, unified_pos=1, space_dim=7, out_dim=4
- 四种任务设置：`-t full`（全量）/ `scarce` / `reynolds` / `aoa`（论文主结果用 `full`）

```bash
cd $REPO/Airfoil-Design-AirfRANS

# 训练（full 任务）
python main.py --model Transolver -t full --my_path $DATA/naca/Dataset --gpu 0

# 评估
python main_evaluation.py --my_path $DATA/naca --gpu 0
```

> 注意：训练 `--my_path` 指向 `.../naca/Dataset`（含 manifest.json），评估 `--my_path` 指向 `.../naca`（代码内部会拼 `/Dataset`）。

---

## 附：所有数据集一键自检

```bash
bash $REPO/check/check_other.sh   # 校验 6 标准基准 + 汽车 + 翼型是否就位可加载
```
