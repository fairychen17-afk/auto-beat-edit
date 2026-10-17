# 自动卡点剪辑 (Auto Beat Edit)

一个基于 Python 的游戏CG/动漫素材自动卡点混剪工具。自动检测音乐节拍，智能变速对齐，支持多种创意特效。

> ⚠️ **依赖版本要求：MoviePy 2.x**（本项目的 API 全部为 2.x 风格，如 `subclipped` / `with_speed_scaled` / `with_audio`，**不兼容 MoviePy 1.x**）。

## ✨ 功能特点

- 🎵 **自动节拍检测**：基于 librosa 分析音乐BPM，自动定位所有节拍点；检测不到可靠拍点时安全回退到合成网格，不会崩溃
- � **音乐自动选段**：按「节拍规律性 + 能量变化 + 拍点密度 + 能量水平」评分，自动挑出最适合卡点的区间并给出推荐理由（也可手动指定起点，手动优先）
- �🎬 **智能节奏设计**：内置 `normal / slow / fast` 三档节奏模板，自动生成「慢→密集快切→慢」的经典混剪结构
- ⏱️ **精确卡点对齐**：支持整拍/半拍级别对齐，每个切换点都卡上拍子
- 🎞️ **智能变速适配**：素材时长不合适时自动快放/慢放（0.8x-1.3x），**变速后精确对齐节拍，时间轴零漂移**
- 🎨 **丰富特效**：
  - 青橙色调色 + 暗角 + 胶片颗粒
  - 缩放脉冲（节拍呼吸感）+ 镜头推镜 + 重拍震动
  - RGB色差Glitch故障 + 动态色相偏移
  - 运动模糊拖影 + 撕裂转场
  - 鱼眼畸变脉冲 + 像素化故障闪烁
  - 镜头光晕光斑
- 📱 **竖屏输出**：720×960，适配短视频平台
- 📋 **剪辑计划（EDL）**：渲染前输出 `edit_plan.json`，记录每段的素材、源入出点、倍速、时间轴区间，并做执行前校验
- 🧪 **可复现测试**：`pytest` 全套测试，全部使用合成音频/视频，不依赖真实素材

## 🚀 快速开始

### 环境要求

- Python 3.9+
- ffmpeg（MoviePy 会自动通过 `imageio-ffmpeg` 获取）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

1. 把素材视频放到一个文件夹里，按顺序命名为 `1.mp4`, `2.mp4`, `3.mp4`, `4.mp4`...
   （把 `CLIP_ORDER` 设为 `[]` 时，会自动扫描文件夹并按 `1 < 2 < 10` 的自然序排序）
2. 把BGM音乐文件也放进去
3. 修改脚本顶部的配置：

```python
FOLDER = r"C:\path\to\your\folder"      # 素材文件夹路径
AUDIO_FILE = "your_music.mp3"           # 音乐文件名
AUDIO_START = 54.0                      # 从音乐第几秒开始（用户指定优先）
AUTO_SELECT_AUDIO = False               # True 时忽略 AUDIO_START，自动挑选最佳区间
AUDIO_DURATION = 10.0                   # 输出时长（秒）
TARGET_RESOLUTION = (720, 960)          # 输出分辨率
CLIP_ORDER = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]  # 素材顺序（留空则自动扫描）
EDIT_STRATEGY = "normal"                # normal / slow / fast
```

4. 运行：

```bash
python auto_beat_edit.py
```

### 运行测试

```bash
python -m pytest tests -q
```

## ⚙️ 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `FOLDER` | 素材文件夹路径 | - |
| `AUDIO_FILE` | 音乐文件名 | - |
| `AUDIO_START` | 音乐起始时间（秒）；设 `None` 启用自动选段 | 54.0 |
| `AUTO_SELECT_AUDIO` | 是否自动挑选最佳音乐区间 | False |
| `AUDIO_DURATION` | 输出视频时长（秒） | 10.0 |
| `TARGET_RESOLUTION` | 输出分辨率 (宽, 高) | (720, 960) |
| `CLIP_ORDER` | 素材文件名列表（留空自动扫描） | ["1.mp4", ...] |
| `EDIT_STRATEGY` | 节奏模板 normal / slow / fast | "normal" |
| `MIN_SHOT` | 最短镜头长度（秒），过短的切点会被跳过 | 0.25 |
| `ALLOW_CLIP_REUSE` | 素材用尽后是否回绕复用 | True |
| `MIN_SPEED` / `MAX_SPEED` | 变速范围 | 0.8 ~ 1.3 |
| `EXPORT_EDL` / `EDL_PATH` | 是否输出剪辑计划 JSON 及路径 | True / edit_plan.json |

### 特效开关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `FILTER_TEAL_ORANGE` | 青橙色调 | True |
| `FILTER_VIGNETTE` | 暗角强度 | 0.6 |
| `FILTER_FILM_GRAIN` | 胶片颗粒强度 | 0.1 |
| `PULSE_INTENSITY` | 缩放脉冲强度 | 0.08 |
| `SHAKE_INTENSITY` | 重拍震动强度（像素） | 3.0 |
| `ZOOM_SPEED` | 长片段推镜速度 | 0.12 |
| `GLITCH_INTENSITY` | Glitch色差强度（像素） | 6.0 |
| `HUE_SHIFT_AMOUNT` | 色相偏移角度 | 15.0 |
| `FISHEYE_ON` | 鱼眼脉冲效果 | True |
| `PIXEL_ON` | 像素化故障效果 | True |
| `FLARE_ON` | 镜头光晕效果 | True |

## ⏱️ 时间语义（重要）

代码中严格区分三种时间，互不混用：

| 名称 | 含义 | 相关 API |
|------|------|----------|
| 源时间 source time | 素材文件自身的时间轴 | `subclipped(start, end)` |
| 片段时间 clip time | 单段素材变速后、拼接前的时间轴 | `with_speed_scaled(speed)` |
| 时间轴时间 timeline time | 所有片段首尾相接后，成片的时间轴 | EDL 中的 `timeline_start/end` |

换算关系（本机 MoviePy 2.1.2 实测确认）：

```
with_speed_scaled(speed) 的结果时长 = 原时长 / speed      # speed>1 加速变短
speed             = source_duration / timeline_duration
source_duration   = timeline_duration * speed
timeline_duration = source_duration / speed
```

> 历史问题：旧版本误用 `with_speed_scaled(1.0 / speed)`，导致片段被拉长而非对齐，成片时长与节拍全部漂移。现已修正。

## 🎯 节奏模式

内置三档模板（`normal` 为经典 20 段 GMV 结构）：

```
开头（5段）：开头闪 → 长 → 短闪 → 中 → 铺垫最长
    ↓
中段（11段）：密集快切（半拍级别）+ 小停顿
    ↓
结尾（4段）：结尾长 → 结尾短闪 → 结尾中长 → 收尾
```

- `slow`：镜头更长、切点更疏
- `fast`：快切更密集
- 所有片段切换点都精确对齐音乐节拍（长片段对齐整拍，快切段对齐半拍）

## 📁 项目结构

```
auto-beat-edit/
├── auto_beat_edit.py        # 主脚本（音乐分析 / 选段 / 拍点 / 规划 / 变速 / 特效 / 导出）
├── tests/
│   └── test_auto_beat_edit.py  # pytest 测试（合成素材，可复现）
├── requirements.txt         # 依赖列表
├── .gitignore               # Git忽略文件
└── README.md                # 说明文档
```

## 🛠️ 技术栈

- **librosa** - 音乐节拍检测 / 选段评分
- **MoviePy 2.x** - 视频处理
- **NumPy** - 像素级特效计算
- **pytest** - 可复现测试

## 📝 License

MIT
