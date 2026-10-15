"""BGM 智能匹配：按素材的情绪/能量/节奏/时长需求，多目标打分选曲。

匹配维度：
- 情绪：素材能量 → calm/chill/energetic，与 BGM 情绪距离
- 能量：素材实际能量值 vs BGM 能量均值（不再用固定目标值）
- 节奏：素材镜头平均时长 → 期望 BPM（镜头越碎，期望 BPM 越高）
- 时长：BGM 能否覆盖成片需求
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_analyzer import BgmInfo
from .scene_detect import Asset

MOOD_RANK = {"calm": 0, "chill": 1, "energetic": 2}


def material_profile(assets: list[Asset]) -> dict:
    """素材整体画像：能量 / 期望 BPM / 明暗 / 色彩 / 镜头碎度 / 景别分布。"""
    if not assets:
        return {"energy": 0.5, "desired_bpm": 110.0, "brightness": 0.5,
                "colorfulness": 0.5, "avg_shot_len": 1.5, "mood": "chill",
                "close_ratio": 0.0, "wide_ratio": 0.0, "motion": 0.5,
                "static_ratio": 0.0}
    e = float(np.mean([a.energy for a in assets]))
    br = float(np.mean([a.brightness for a in assets]))
    cf = float(np.mean([a.colorfulness for a in assets]))

    # 镜头平均时长 → 期望 BPM：剪辑时每段跨 2~4 拍，期望拍间隔≈镜头时长/3
    shot_lens = [sh.end - sh.start for a in assets if a.kind == "video"
                 for sh in a.shots]
    avg_shot = float(np.mean(shot_lens)) if shot_lens else 1.5
    desired_bpm = float(np.clip(180.0 / max(avg_shot, 0.4), 70.0, 180.0))

    # 景别/运动分布：决定「怎么剪」的风格取向
    shots = [sh for a in assets if a.kind == "video" for sh in a.shots]
    n_sh = max(len(shots), 1)
    close_ratio = sum(1 for s in shots if s.shot_type == "close") / n_sh
    wide_ratio = sum(1 for s in shots if s.shot_type == "wide") / n_sh
    motion = float(np.mean([s.motion for s in shots])) if shots else e
    static_ratio = sum(1 for s in shots if s.is_static) / n_sh

    # 能量主导情绪，明亮+艳丽画面上调一档
    if e < 0.35:
        mood = "calm"
    elif e < 0.62:
        mood = "chill"
    else:
        mood = "energetic"
    if mood == "calm" and br > 0.55 and cf > 0.55:
        mood = "chill"
    return {"energy": e, "desired_bpm": desired_bpm, "brightness": br,
            "colorfulness": cf, "avg_shot_len": avg_shot, "mood": mood,
            "close_ratio": close_ratio, "wide_ratio": wide_ratio,
            "motion": motion, "static_ratio": static_ratio}


def auto_style(profile: dict) -> str:
    """按素材画像推荐剪辑风格（比只看情绪更细）。

    - 大量特写（人脸/主体）→ balanced：别用花哨手法破坏构图
    - 高能量 + 全景/运动镜头多 → dynamic：快切、结构型特效放得开
    - 低能量或画面平稳 → cinematic：长镜头、慢节奏、电影感调色
    """
    e = profile.get("energy", 0.5)
    close = profile.get("close_ratio", 0.0)
    wide = profile.get("wide_ratio", 0.0)
    motion = profile.get("motion", 0.5)
    if e < 0.40 or (motion < 0.30 and close < 0.5):
        return "cinematic"
    if e >= 0.62 and (wide >= 0.35 or motion >= 0.55):
        return "dynamic"
    if close >= 0.55:
        return "balanced"
    return {"energetic": "dynamic", "chill": "balanced",
            "calm": "cinematic"}.get(profile.get("mood", "chill"), "balanced")


def material_mood(assets: list[Asset]) -> str:
    return material_profile(assets)["mood"]


def pick_bgm(candidates: list[BgmInfo], mood: str,
             min_duration: float = 15.0,
             profile: dict | None = None) -> BgmInfo | None:
    """多目标打分取最高分。"""
    if profile is None:
        profile = {"energy": {"calm": 0.3, "chill": 0.55, "energetic": 0.85}[mood],
                   "desired_bpm": {"calm": 80, "chill": 105, "energetic": 140}[mood]}
    target_energy = profile["energy"]
    desired_bpm = profile["desired_bpm"]

    best, best_score = None, -1e9
    for info in candidates:
        mood_dist = abs(MOOD_RANK[info.mood] - MOOD_RANK[mood])
        energy_match = 1.0 - abs(info.energy_mean - target_energy)
        bpm_match = 1.0 - min(abs(info.bpm - desired_bpm) / 80.0, 1.0)
        dur_score = (1.0 if info.duration >= min_duration
                     else min(1.0, info.duration / min_duration))
        # 结构分：有明确高潮段（副歌/drop）的曲子更适合做卡点
        sections = getattr(info, "sections", None) or []
        has_high = any(lv == "high" for _, _, lv in sections)
        span = max(info.duration - info.climax, info.climax)
        section_score = (0.6 if has_high else 0.2) + min(span / 60.0, 0.4)
        score = (-3.0 * mood_dist
                 + 2.0 * energy_match
                 + 1.5 * bpm_match
                 + 1.2 * dur_score
                 + 0.8 * section_score)
        if score > best_score:
            best, best_score = info, score
    return best
