# Transolver 跨层传播核监控

该目录提供一个完全可选、运行时注入的诊断工具。它不会修改 Transolver 的模型、损失、优化器、训练脚本或原有实验输出。删除整个 `LINEARNO/monitor` 即可移除本功能。

## 测量内容

对每层、每个 head，记点到 slice 的权重为 `W`，slice 归一化矩阵为 `D = diag(W^T 1 + 1e-5)`，slice-token 自注意力为 `A`。工具同时测量：

```text
P_base = W D^-1 W^T
P_full = W A D^-1 W^T
```

`P_base` 用于判断切片/反切片路由是否重复，`P_full` 用于判断加入 slice-token 注意力后的有效空间传播是否重复。主图是各层 head 经过最优匹配后的 `P_full` Frobenius 余弦相似度。工具还保存同编号 head 的均值和完整的两两 head 相似度，避免把 head 或 slice 的置换误判成差异。

这里的核描述 value/output 通道投影之前的空间路由，不应称为包含全部通道混合的完整 Transformer block 算子。

## 运行

从仓库根目录执行。`--` 后的内容是原训练脚本及其原始参数；包装器默认把工作目录设为该脚本所在目录。

```bash
python LINEARNO/monitor/run.py --dataset darcy -- \
  PDE-Solving-StandardBenchmark/exp_darcy.py --gpu 0 --epochs 500

python LINEARNO/monitor/run.py --dataset navier_stokes -- \
  PDE-Solving-StandardBenchmark/exp_ns.py --gpu 0

python LINEARNO/monitor/run.py --dataset elasticity -- \
  PDE-Solving-StandardBenchmark/exp_elas.py --gpu 0

python LINEARNO/monitor/run.py --dataset plasticity -- \
  PDE-Solving-StandardBenchmark/exp_plas.py --gpu 0

python LINEARNO/monitor/run.py --dataset pipe -- \
  PDE-Solving-StandardBenchmark/exp_pipe.py --gpu 0

python LINEARNO/monitor/run.py --dataset standard_airfoil -- \
  PDE-Solving-StandardBenchmark/exp_airfoil.py --gpu 0

python LINEARNO/monitor/run.py --dataset airfrans -- \
  Airfoil-Design-AirfRANS/main.py --gpu 0

python LINEARNO/monitor/run.py --dataset shapenetcar --run-name fold0 -- \
  Car-Design-ShapeNetCar/main.py --cfd_model Transolver --fold_id 0 --gpu 0
```

只有通过此包装器启动时才会启用监控。原命令仍负责自己的 checkpoint、最终权重和可视化；监控器不会重定向或复制这些文件。

## 频率与大网格

默认捕获每次进入 `model.eval()` 后的验证前向，每个快照最多汇总 8 个样本。以下选项将捕获第 1 次验证以及第 10、20、30 次验证：

```bash
--capture-every-validations 10 --max-samples-per-snapshot 8
```

加上 `--no-capture-first` 后只捕获第 10、20、30 次。验证编号表示监控器观察到的验证过程，不推测训练脚本内部的 epoch 编号。

默认对每个点轴分别、确定性地采样最多 4096 行，并始终使用全部点计算 `D`。两组独立行样本对应 `P` 的两个点轴，可以减少使用同一子矩阵带来的估计偏差。所有层共享采样索引，且私有 CPU 随机生成器不会推进训练 RNG。小网格自动精确计算；可用 `--max-points 0` 强制精确计算，但 ShapeNetCar/AirfRANS 可能明显增加内存和时间。

## 输出

输出固定位于：

```text
LINEARNO/monitor/output/<dataset>/<YYYYMMDD_HHMMSS[_name]>/
```

根目录包含 `run_config.json`、`model_structure.json`、`monitor.log` 和 `summary.json`。`run_config.json` 同时记录原始命令和训练脚本经 `argparse` 解析后的完整参数（包括默认值）。每个被捕获的验证过程位于 `snapshots/validation_XXXXXX/`，其中包含：

- 原始数值与均值/标准差 `similarity_values.npz`
- 完整 head 匹配结果 `head_matching.npz`
- 每样本指标和主矩阵 CSV
- 采样索引与精确/估计标记 `sampling.json`
- `base`、`full`、二者对照以及标准差热力图
- 300 DPI PNG 和矢量 PDF

若训练节点不安装 Matplotlib，可加 `--no-plots` 先保存数值，之后运行：

```bash
python LINEARNO/monitor/plot_snapshot.py \
  LINEARNO/monitor/output/<dataset>/<run>/snapshots/validation_000001
```

监控计算或绘图异常只记录到 `monitor.log`，不会中止原训练；原训练自身的异常仍以非零状态退出。

## 测试

```bash
python -m unittest discover -s LINEARNO/monitor/tests -v
```
