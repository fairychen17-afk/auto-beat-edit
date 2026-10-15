# autocut — 自动卡点剪辑流水线

把一堆**视频/图片素材**和一个 **BGM** 丢进去，自动产出一条**卡点转场成片**。

全流程无人值守：自动检测关键帧 → 分析素材「性格」→ 自动选曲 → 按音乐节拍编排剪辑点 →
自动配转场/滤镜/特效/变速 → ffmpeg 渲染出成品。**不需要手动打点，也不需要剪辑经验。**

```
素材目录 ──► 镜头分割 + 关键帧检测 ──► 素材画像（能量/景别/运动/明暗/色彩）
                                              │
BGM 目录 ──► 节拍·BPM·能量·结构分析 ──► 多目标打分选曲 ──► 最佳片段搜索
                                              │
                                              ▼
                        节拍网格编排（EDL）: 段长随能量起伏 + 素材感知手法分配
                                              │
                        ffmpeg 渲染: 规格化 → 镜头运动 → 卡点特效 → 调色 → 转场链 → 变速混音
                                              ▼
                                        output/final.mp4
```

---

## 能力总览

| 环节 | 做了什么 |
|---|---|
| **关键帧/镜头检测** | 抽帧做 HSV 直方图差异切分镜头；帧差估计运动强度曲线；每个镜头内**运动峰值帧即关键帧**（最佳卡点位置）。同时给出景别（特写/中景/全景）、明暗、色彩丰富度、横向运动方向、是否近乎静止 |
| **素材画像** | 聚合能量、镜头碎度、景别分布、运动均值 → 决定整体情绪、期望 BPM、剪辑风格 |
| **BGM 分析** | librosa 提取节拍点、强拍、BPM、onset 能量包络、高潮位置，并按节拍能量把曲子切成 low/mid/high **结构段落**（前奏 / build / drop / 间奏） |
| **BGM 匹配** | 五维打分选曲：情绪距离 + 能量接近 + **节奏匹配**（素材镜头均长 → 期望 BPM）+ 时长覆盖 + **结构分**（有明确 drop 的更适合作卡点） |
| **BGM 截取** | 在节拍网格上**搜索最佳片段**：同时照顾「包含高潮」「时长贴合目标」「能量够高」「起点落在小节头」，因此成片踩在高潮上且变速比接近 1（音质无损） |
| **自动卡点** | 所有剪辑点吸附节拍网格；**段长随音乐能量起伏**（高能处每拍一切、低能处给长镜头）；转场中点即卡点 |
| **剪辑手法** | 镜头运动 15 种 + 卡点特效 17 种 + 转场 40 余种（分 soft/mid/hard 三档）+ 调色预设 13 套 + 段内变速，全部由素材特征驱动（见下节） |
| **转场** | ffmpeg xfade，按段落能量分档；**每个边界可用不同时长**（跳切≈硬切 0.05s、硬档更短促、软档更长）；方向性转场可顺着画面运动方向做匹配剪辑 |
| **滤镜/调色** | 按情绪选预设池，相邻段不重复；素材本身很艳丽时自动换低饱和组避免叠 buff 过艳 |
| **变速** | 整体 atempo 变速（0.8~1.25）使节拍数匹配目标时长；单段可慢放/快放；支持**段内变速**（先慢后快做 build、先快后慢做落定） |
| **输出** | 片头片尾自动黑场淡入淡出、BGM 淡入淡出、可选压低混入素材原声、统一 yuv420p 保证播放器兼容 |
| **可解释** | `--dry-run` / `--verbose` 打印**逐段决策与理由**，并导出 EDL JSON（明确写了每一段为什么这样剪） |

---

## 素材感知：怎么"按素材选剪辑方式"

编排引擎（`autocut/moves.py::plan_shots`）不是随机撒手法，而是逐段读素材特征做决策：

| 素材特征 | 剪辑决策 |
|---|---|
| 主体居中明显（特写/人脸） | 禁用分屏/镜像/色差/故障/画中画等破坏性特效，运动也只用温和的 → **保护构图** |
| 空旷少动的全景 | 放得开：结构型特效、更猛更短的转场 |
| 镜头本身运动很强 | 特效收敛（不再叠频闪/故障/叠影），避免画面"花" |
| 画面偏暗 | 配**白闪**（看得见）；画面偏亮 → 配**黑闪** |
| 素材本身很艳丽 | 自动换低饱和调色组 |
| 画面在往左/往右移 | 转场选同方向滑动/擦除 → **匹配剪辑** |
| 同一镜头被切成连续两段 | 判定为**跳切**，转场压到 ≈硬切 |
| 音乐结构边界（能量阶跃） | 强制高档转场，做出段落感 |
| 音乐低能量段 | 氛围型特效（遮幅/复古/漏光）+ 长转场 |
| 音乐高潮段 | 节拍特效（每拍提亮/压暗/脉动/频闪）+ 短转场 |
| build 段 / 落定处 | **段内变速**：先慢后快 / 先快后慢 |

风格也会按素材自动推荐（`auto_style`）：大量特写 → `balanced`；高能量+全景/运动多 → `dynamic`；
低能量或画面平稳 → `cinematic`。

---

## 安装

```bash
pip install -r requirements.txt      # 或 pip install -e .
```

ffmpeg 由 `imageio-ffmpeg` 自带，**无需单独安装**。

## 使用

```bash
# 最简：素材目录 + BGM 目录（自动选曲）
python main.py --material ./material --bgm ./bgm --output ./output/final.mp4

# 指定单曲、竖屏、指定 BGM 起点、按素材顺序、不重复用素材
python main.py --material ./material --bgm ./bgm/song.mp3 --output ./output/out.mp4 \
       --resolution 1080x1920 --fps 30 --preserve-order --bgm-start 54

# 只看方案不渲染（导出决策表 + 逐段理由）
python main.py --material ./material --bgm ./bgm --dry-run --edl plan.json
```

安装为命令后也可用：`autocut --material ...` 或 `python -m autocut ...`

### 参数

| 参数 | 说明 |
|---|---|
| `--material` | 素材目录（视频/图片混放） |
| `--bgm` | BGM 目录（自动打分选曲）或单个音频文件 |
| `--output` | 输出路径，默认 `output/final.mp4` |
| `--duration 30` | 目标成片时长（秒），默认按素材量自动 |
| `--resolution 1080x1920` | 输出分辨率，默认 `1920x1080` |
| `--fps 30` | 输出帧率 |
| `--style dynamic` | 风格：`dynamic` / `balanced` / `cinematic` / `auto`（默认，按素材自动） |
| `--intensity 1.2` | 运动与特效强度倍率（0.3~1.5） |
| `--camera-motion off` | **镜头运动三档**：`off` 画面完全稳定（默认，只切镜/转场/调色/节拍特效）；`subtle` 仅温和缓推缓拉；`full` 完整运镜库 |
| `--preserve-order` | 按素材文件名/时间顺序排列（默认按能量与高潮匹配穿插） |
| `--allow-reuse` | 允许素材窗口重复使用（默认每段只用一次，用完即止） |
| `--bgm-start 54` | BGM 从第 54 秒起剪，自动吸附到最近节拍 |
| `--seed 42` | 随机种子，同素材+种子可复现同一版剪辑 |
| `--mix-original` | 保留素材原声（压低至 12% 混入） |
| `--dry-run` | 只分析+编排，导出 EDL 并打印报告，**不渲染** |
| `--edl plan.json` | 导出剪辑决策表（EDL）JSON |
| `--verbose` / `-v` | 打印逐段剪辑决策与理由 |

---

## 剪辑手法库

**镜头运动**（超采 1.5× + zoompan 逐帧驱动，周期性运动的相位与 BPM 对齐）
`zoom_in` `zoom_out` `pan_left/right/up/down` `drift` `ken_diag` `breath` `sway`
`zoom_punch` `punch_pan` `heartbeat` `shake` `crash_zoom`

**卡点特效**
`flash_white` `flash_black` `flash_freeze` `beat_flash` `beat_dim` `beat_pulse`
`strobe` `echo`(叠影) `chroma` `glitch`(故障) `hflip` `split_mirror`(镜像分屏)
`pip_card`(画中画卡片) `letterbox`(电影遮幅) `old_film`(复古胶片) `warm_leak`(暖调漏光) `grain`

**转场**：40 余种 xfade 分三档，含方向性的 slide/wipe/cover/reveal/smooth 系列、
结构性的 pixelize/diag/slice/squeeze/zoomin/distance 系列。

**调色**：energetic / chill / calm 各一套预设池 + 低饱和平衡组，相邻段不重复。

---

## 项目结构

```
autocut/
├── autocut/
│   ├── scene_detect.py     # 镜头分割、关键帧、运动曲线、景别/明暗/色彩/运动方向
│   ├── audio_analyzer.py   # 节拍/BPM/能量/高潮/结构段落 + 最佳片段搜索
│   ├── bgm_matcher.py      # 素材画像、多目标选曲、自动风格推荐
│   ├── moves.py            # 剪辑手法库 + 素材感知编排（规则引擎，输出决策理由）
│   ├── timeline.py         # 节拍网格 EDL：段长起伏、素材分配、变速、逐边界转场
│   ├── renderer.py         # ffmpeg 渲染：滤镜链、段内变速、xfade 链、混音
│   ├── edl.py              # 剪辑决策表导出（JSON）
│   └── cli.py              # 命令行入口与报告
├── tools/
│   ├── check_filters.py    # 手法自检：逐个渲染每种手法，验证滤镜链成立
│   ├── make_test_assets.py # 生成合成测试素材（多镜头视频/图片/节拍 BGM）
│   └── make_bgm.py         # 合成测试 BGM（EDM）
├── tests/test_pipeline.py  # 单元测试（编排规则/时长数学/滤镜链/选曲/EDL）
├── main.py                 # 便捷入口
└── pyproject.toml
```

## 自检与测试

```bash
python -m unittest discover -s tests -v     # 单元测试（纯逻辑，秒级）
python tools/check_filters.py 素材.mp4       # 逐个渲染所有手法，验证 ffmpeg 滤镜链
python tools/make_test_assets.py             # 造合成素材 → material/ bgm/
python main.py --material material --bgm bgm --output output/final.mp4
```

## 说明

- **素材/音乐版权**：仓库不包含任何素材与音乐文件（`.gitignore` 已排除），请自备合法素材。
- 图片素材会自动做轻微 Ken Burns 变化，避免静止图片显得死板；视频段在 `--camera-motion off` 下完全不动。
- 相邻段的调色/转场/特效都做了去重，避免"同一招连着用"的单调感。

## License

MIT
