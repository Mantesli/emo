# 问题一数据分析管线

当前工程只建立分析代码骨架，不执行特征提取、时间戳（PTS）提取或
manifest 构建，也不移动原始数据。

## 数据处理流程

```text
原始数据
  → 数据审计
  → manifest
  → 基础统计
  → 时间轴审计
  → 后续特征提取和时序对齐
```

问题一的默认原始数据目录为 `data1/`，标签文件为
`data1/label-100.xlsx`。所有工程路径集中定义在 `src/config.py`；生成的
元数据、图形和日志默认写入 `outputs/metadata/`、`outputs/figures/` 和
`outputs/logs/`，与原始数据目录分离。

## 当前模块职责

- `src/config.py`：集中定义项目、原始数据、标签及输出目录路径。
- `src/utils/io_utils.py`：提供 ID 规范化、目录创建和安全浮点数转换。
- `src/utils/video_utils.py`：提供视频文件枚举、路径校验和系统 `ffprobe`
  可用性检查；暂不进行复杂 PTS 提取。

后续实现应先完成数据审计，再生成 manifest，并在此基础上进行统计、时间轴
审计和特征/时序对齐。系统存在 `ffprobe` 时，直接调用系统可执行文件即可，
不要求引入 Python ffmpeg 封装库。

## 数据与版本管理

原始数据目录 `data1/`–`data4/`、标签文件、生成的 `outputs/` 结果和本机 IDE 配置不会提交到 Git。数据文件体积较大，且部分结果记录了本机路径；请按项目约定将所需数据放在本地对应目录中。GitHub 仓库只保存源码、依赖清单和项目说明。

## 环境准备

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

部分时间戳分析脚本还需要系统安装 `ffprobe`，并确保它可从命令行调用。

