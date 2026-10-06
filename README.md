# 自动卡点剪辑 (Auto Beat Edit)

一个基于 Python 的游戏CG/动漫素材自动卡点混剪工具。自动检测音乐节拍，智能变速对齐，支持多种创意特效。

## ✨ 功能特点

- 🎵 **自动节拍检测**：基于 librosa 分析音乐BPM，自动定位所有节拍点
- 🎬 **智能节奏设计**：自动生成"慢→密集快切→慢"的经典混剪节奏结构
- ⏱️ **精确卡点对齐**：支持整拍/半拍级别对齐，每个切换点都卡上拍子
- 🎞️ **智能变速适配**：素材时长不合适时自动快放/慢放（0.85x-1.25x），精确对齐节拍
- 🎨 **丰富特效**：
  - 青橙色调色 + 暗角 + 胶片颗粒
  - 缩放脉冲（节拍呼吸感）+ 镜头推镜 + 重拍震动
  - RGB色差Glitch故障 + 动态色相偏移
  - 运动模糊拖影 + 撕裂转场
  - 鱼眼畸变脉冲 + 像素化故障闪烁
  - 镜头光晕光斑
- 📱 **竖屏输出**：720×960，适配短视频平台

## 🚀 快速开始

### 环境要求

- Python 3.8+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

1. 把素材视频放到一个文件夹里，按顺序命名为 `1.mp4`, `2.mp4`, `3.mp4`, `4.mp4`...
2. 把BGM音乐文件也放进去
3. 修改脚本顶部的配置：

```python
FOLDER = r"C:\path\to\your\folder"      # 素材文件夹路径
AUDIO_FILE = "your_music.mp3"           # 音乐文件名
AUDIO_START = 54.0                      # 从音乐第几秒开始
AUDIO_DURATION = 10.0                   # 输出时长（秒）
TARGET_RESOLUTION = (720, 960)          # 输出分辨率
CLIP_ORDER = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]  # 素材顺序
```

4. 运行：

```bash
python auto_beat_edit.py
```

## ⚙️ 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `FOLDER` | 素材文件夹路径 | - |
| `AUDIO_FILE` | 音乐文件名 | - |
| `AUDIO_START` | 音乐起始时间（秒） | 54.0 |
| `AUDIO_DURATION` | 输出视频时长（秒） | 10.0 |
| `TARGET_RESOLUTION` | 输出分辨率 (宽, 高) | (720, 960) |
| `CLIP_ORDER` | 素材文件名列表（按顺序） | ["1.mp4", ...] |
| `MIN_SPEED` / `MAX_SPEED` | 变速范围 | 0.8 ~ 1.3 |

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

## 🎯 节奏模式

默认采用经典GMV节奏结构（共20段）：

```
开头（5段）：开头闪 → 长 → 短闪 → 中 → 铺垫最长
    ↓
中段（11段）：密集快切（半拍级别，每段0.1-0.2秒）+ 小停顿
    ↓
结尾（4段）：结尾长 → 结尾短闪 → 结尾中长 → 收尾
```

所有片段切换点都精确对齐音乐节拍（长片段对齐整拍，快切段对齐半拍）。

## 📁 项目结构

```
auto-beat-edit/
├── auto_beat_edit.py    # 主脚本
├── requirements.txt     # 依赖列表
├── .gitignore           # Git忽略文件
└── README.md            # 说明文档
```

## 🛠️ 技术栈

- **librosa** - 音乐节拍检测
- **MoviePy** - 视频处理
- **NumPy** - 像素级特效计算

## 📝 License

MIT
