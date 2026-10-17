"""剪辑决策表（EDL）导出。

把时间线打印成可读的 JSON / 文本，便于：
- 不渲染就检查剪辑方案是否合理（`--dry-run`）
- 交给别的工具（剪辑软件、绘声绘影 XML、FCPXML 转换器等）二次加工
- 归档"这一版是怎么剪的"
"""
from __future__ import annotations

import json
from pathlib import Path

from .timeline import Timeline

try:                                  # 版本号取包元信息，取不到就退回常量
    from importlib.metadata import version as _pkg_version
    VERSION = _pkg_version("autocut")
except Exception:                     # noqa: BLE001
    VERSION = "1.0.0"


def timeline_to_dict(tl: Timeline, width: int, height: int, fps: int) -> dict:
    """把 Timeline 转成可序列化的字典（同时是给人看的报告）。"""
    segs = tl.segments
    bdurs = tl.boundary_durs
    out = {
        "version": VERSION,
        "output": {"width": width, "height": height, "fps": fps,
                   "duration": round(tl.total_duration, 3)},
        "bgm": {
            "file": tl.bgm.path.name,
            "bpm": round(float(tl.bgm.bpm), 2),
            "mood": tl.bgm.mood,
            "tempo_ratio": round(float(tl.tempo_ratio), 4),
            "clip_start": round(tl.bgm_start, 3),
            "clip_end": round(tl.bgm_end, 3),
            "climax": round(float(tl.bgm.climax), 3),
            "sections": [{"start": round(s, 2), "end": round(e, 2), "level": lv}
                         for s, e, lv in (tl.bgm_sections or [])],
        },
        "edit": {"mood": tl.mood, "style": tl.style, "intensity": round(tl.amp, 3),
                 "default_transition_dur": round(tl.trans_dur, 3),
                 "segments": len(segs)},
        "segments": [],
    }
    cursor = 0.0
    for i, s in enumerate(segs):
        tdur = bdurs[i] if i < len(bdurs) else 0.0
        out["segments"].append({
            "index": i,
            "source": s.asset.path.name,
            "kind": s.asset.kind,
            "src_in": round(s.src_in, 3),
            "duration": round(s.duration, 3),
            "speed": round(s.speed, 4),
            "speed_ramp": (list(s.ramp) if s.ramp else None),
            "camera": s.move,
            "fx": s.fx,
            "grade": s.grade,
            "transition_out": s.transition,
            "transition_dur": round(tdur, 3),
            "music_energy": round(s.energy, 3),
            "music_tension": round(getattr(s, "tension", 0.5), 3),
            "material_quality": round(getattr(s, "quality", 0.5), 3),
            # 卡点位置：该段中点（转场中点重合处即卡点）
            "cut_at": round(cursor + s.duration / 2, 3),
            "why": s.why,
        })
        cursor += s.duration - tdur
    out["boundary_durs"] = [round(d, 3) for d in bdurs]
    return out


def export_edl(tl: Timeline, path: Path, width: int, height: int, fps: int) -> Path:
    """导出 EDL JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = timeline_to_dict(tl, width, height, fps)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
