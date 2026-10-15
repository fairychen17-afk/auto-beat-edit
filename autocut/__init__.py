"""autocut — 自动卡点剪辑流水线。

把一堆视频/图片素材和一个 BGM 丢进去，自动产出卡点转场成片：

    素材扫描 → 镜头/关键帧检测 → 素材画像
                                    ↓
    BGM 分析 → 最佳片段搜索 → 选曲 → 节拍网格编排（EDL）→ ffmpeg 渲染

对外主要接口：
    scene_detect.scan_materials(material_dir) -> list[Asset]
    audio_analyzer.analyze_bgm(path) -> BgmInfo
    bgm_matcher.pick_bgm(candidates, mood, profile=...) -> BgmInfo
    timeline.build_timeline(assets, bgm, mood, ...) -> Timeline
    renderer.render(timeline, output, width, height, fps) -> Path
"""
from .audio_analyzer import BgmInfo, analyze_bgm, scan_bgm_library
from .bgm_matcher import auto_style, material_mood, material_profile, pick_bgm
from .edl import export_edl, timeline_to_dict
from .renderer import render
from .scene_detect import Asset, Shot, scan_materials
from .timeline import Segment, Timeline, build_timeline

__version__ = "1.0.0"

__all__ = [
    "Asset", "Shot", "scan_materials",
    "BgmInfo", "analyze_bgm", "scan_bgm_library",
    "material_profile", "material_mood", "auto_style", "pick_bgm",
    "Segment", "Timeline", "build_timeline",
    "render", "export_edl", "timeline_to_dict",
    "__version__",
]
