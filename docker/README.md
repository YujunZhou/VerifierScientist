# 镜像是怎么做的

打包的机器上没有 Docker，所以镜像不是用 Dockerfile 构建的，而是把需要的目录直接打成镜像层，叠在官方底座上推送。运行方不需要重做这些，本页只是记录。

## 底座

`nvidia/cuda:13.2.1-devel-rockylinux9`。选它是因为 CUDA 工具链版本和操作系统都与原来训练用的机器一致（那边是 RHEL 9，CUDA 13.2.1）。

## 叠加的层

所有文件保持原机器上的绝对路径（`/groups/xzhang33/tguo2taicheng/...`），因为代码、配置和校验清单里都用的是这些路径。属主统一改成 root。

| 层 | 内容 | 大小（未压缩） |
|---|---|---|
| `01_sys` | git（从原机器的系统包里取出）、`/etc/gitconfig`、把 `/software/c/cuda/13.2.1` 指向底座自带 CUDA 的符号链接 | 11 MB |
| `02_python` | 各版本 Python 解释器（3.8–3.11）和两个小工具目录 | 0.5 GB |
| `03_env_jit`、`04_env_main` | 训练用的 Python 环境（PyTorch 2.11、verl 0.8、SGLang 0.5.12、Ray 2.56 等），因体积拆成两层 | 20 GB |
| `05_tasks` | 软件工程类任务用到的 30 个代码库（带 git 历史）和各自的虚拟环境 | 9 GB |
| `06_model` | 基础模型 Qwen3.5-4B（公开模型，Hugging Face 缓存目录结构） | 9 GB |

## 用到的工具

- `build_layers.sh`：用 `tar` 加 `pigz` 生成各层。
- `crane`（go-containerregistry）：`crane append` 把各层叠到底座上并推送，不需要 Docker 守护进程，也不需要管理员权限。

## 推送前做过的验证（都不用显卡）

在底座镜像解包出的副本里，按运行方的实际流程执行了 `envcheck`、`verify`、`netcheck`，并试启动了冒烟训练：它通过了全部启动前检查，停在"检测不到 4 张显卡"这一步，符合预期。显卡上的部分没有验证过。

## 镜像里没有什么

没有任何 API 密钥；没有训练代码、题库和配置（这些在本仓库的 `tree/` 下，运行时铺进去）。

各层打好后还做了一次独立验证：把六个层解到一个全新的底座副本里（不借用原机器上的任何目录），配合从 Git 克隆下来的本仓库，重跑 `envcheck` 和 `verify`，两项都通过，约 4 万个文件的指纹与审核时一致。

## 已推送的镜像

- 名称：`docker.io/yujun0017/verifierscientist-train:2026-11-11`（私有仓库）
- 指纹：`sha256:fd411f339cd73834659de70567d63b9de404d8c1ad247ec80b85d95713c92d50`
- 17 层（底座 11 层加上表的 6 层），压缩后约 23.8 GB，解开约 50 GB。
- 推送后核对过：仓库里 6 个新增层的指纹与本地做过验证的层逐一相同。

拉取后可以用 `docker inspect --format '{{index .RepoDigests 0}}' docker.io/yujun0017/verifierscientist-train:2026-11-11` 核对指纹。
