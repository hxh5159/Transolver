# 数据下载脚本

下载本仓库标准基准（6 个）与翼型设计（AirfRANS）所需的数据，落到与现有 `mlcfd` 数据同级的远端路径：

```
DATA=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data
```

## 文件说明

| 脚本 | 下载内容 | 落地位置 |
|---|---|---|
| `download_fno.sh` | Darcy + Navier-Stokes（FNO Drive 文件夹） | `$DATA/fno/` |
| `download_geofno.sh` | Elasticity + Plasticity + Airfoil + Pipe（Geo-FNO Drive 文件夹） | `$DATA/fno/` |
| `download_airfrans.sh` | AirfRANS 翼型设计 `Dataset.zip`（约 9.3GB） | `$DATA/naca/Dataset/` |
| `verify.sh` | 校验 + 按文件名归位缺失文件 | — |
| `run_all.sh` | 并行启动上面 3 个下载，完成后统一校验 | — |
| `common.sh` | 公共变量与 `retry` 重试函数 | — |

## 使用

```bash
# 方式一：一次性并行下载（利用多核）+ 校验
bash run_all.sh

# 方式二：单独下载某一块
bash download_fno.sh
bash download_geofno.sh
bash download_airfrans.sh
bash verify.sh
```

## 重试逻辑

- `common.sh` 里的 `retry` 会在命令失败后 **sleep 10 秒再自动重试**，直到成功（Ctrl+C 可中断）。
- 单个文件下载（`download_file`）会在**配额限制**（"Too many users..."）时等 `QUOTA_DELAY` 秒（默认 1800=30 分钟）再重试，普通失败仍 10s。
- 已下载的文件会自动跳过，重复执行等价于断点续传。
- AirfRANS 优先用 `aria2c -x16 -s16` 多连接下载，无 aria2c 时回退 `wget -c`。

## Google Drive 配额与绕过

公开数据集被下载过多时会报 `Too many users have viewed or downloaded this file recently`，需等待数小时恢复，单纯重试无效。可选绕过方式：

```bash
# 用浏览器登录 Google 后导出 cookies.txt（如 "Get cookies.txt" 扩展），再：
export GDRIVE_COOKIE=/path/to/cookies.txt
bash download_geofno.sh
```

脚本会在 gdown 命令上自动加 `--cookie $GDRIVE_COOKIE`。

## 说明

- 标准基准的 6 个数据集来自两个 Google Drive 文件夹，`gdown` 只能整文件夹下载，无法按文件并行，故按「下载源」分脚本；多核并行由 `run_all.sh` 同时跑 3 个源实现。
- 下载完成后 `verify.sh` 会检查每个训练所需文件是否存在且非空；若 `gdown` 把文件放到了子目录，会按文件名自动归位到正确路径。
