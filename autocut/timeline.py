"""时间线编排：把镜头/图片吸附到 BGM 节拍网格，生成剪辑决策表（EDL）。

相比「等长硬切」，这里多了三件事：
1. **节奏起伏**：每段跨越的拍数由音乐能量决定（越强切得越密），
   所以 build-up 越来越急、drop 段最密、间奏给长镜头。
2. **手法编排**：每段分配镜头运动 / 卡点特效 / 调色 / 速度，避免相邻重复。
3. **变速收敛**：素材余量不足时自动降速（慢放）补时长，余量充裕时可快放。
"""
from __future__ import annotations

import random

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .audio_analyzer import BgmInfo
from .moves import plan_shots, resolve_style, style_amp, tier_of
from .scene_detect import Asset

BEATS_PER_SEG = {"calm": 4, "chill": 3, "energetic": 2}
TRANS_DUR = {"calm": 0.6, "chill": 0.5, "energetic": 0.28}
MAX_SEGMENTS = 24


@dataclass
class Segment:
    asset: Asset
    src_in: float          # 源素材入点（秒，图片忽略）
    duration: float        # 该段成片时长（秒，含两端转场重叠的部分）
    transition: str        # 与下一段之间的 xfade 转场
    grade: str             # 该段调色链
    move: str = "static"   # 镜头运动
    fx: str = "none"       # 卡点特效
    speed: float = 1.0     # 播放速度倍率（<1 慢放）
    ramp: tuple | None = None  # 段内变速 (起始速度, 结束速度)
    tdur: float | None = None  # 本段之后那次转场的时长（None → 用全局值）
    energy: float = 0.5    # 该段所处音乐能量
    why: str = ""          # 该段剪辑决策理由（可读性/调试用）
    ken_burns_in: bool = True


@dataclass
class Timeline:
    segments: list[Segment]
    bgm: BgmInfo
    bgm_start: float       # BGM 截取起点
    bgm_end: float
    tempo_ratio: float     # atempo 变速比
    trans_dur: float       # 默认转场时长（各段可用 tdur 覆盖）
    mood: str
    style: str = "balanced"
    amp: float = 0.85
    bgm_sections: list = field(default_factory=list)   # BGM 结构段落

    @property
    def boundary_durs(self) -> list[float]:
        """每个转场边界（共 n-1 个）的实际时长。"""
        out = []
        for i in range(len(self.segments) - 1):
            d = self.segments[i].tdur
            out.append(float(self.trans_dur if d is None else d))
        return out

    @property
    def total_duration(self) -> float:
        """成片时长 = 各段流长度之和 - 各次转场的重叠量。"""
        return sum(s.duration for s in self.segments) - sum(self.boundary_durs)


@dataclass
class Unit:
    """一个可用素材窗口：视频镜头可按节拍长度切成多个窗口。"""
    asset: Asset
    shot: object | None       # Shot or None（图片）
    s: float                  # 窗口可用起点
    e: float                  # 窗口可用终点
    center: float             # 最佳构图中心（关键帧）
    motion: float


def _collect_units(assets: list[Asset], wants: list[float]) -> list[Unit]:
    """按每段的需求长度依次从素材里切窗口（变长切分，利用率最高）。

    旧做法是等长切分，导致窗口长度与实际段长需求不匹配，素材余量不足时
    只能靠慢放补时长。这里改成「要多少切多少」，素材时间被充分用满。
    """
    units: list[Unit] = []
    qi = 0
    for a in assets:
        if a.kind != "video":
            continue
        for shot in a.shots:
            pos = float(shot.start)
            while qi < len(wants) and shot.end - pos >= min(0.5, wants[qi] * 0.5):
                need = wants[qi]
                w = min(need, shot.end - pos)
                s, e = pos, pos + w
                center = shot.keyframe if s <= shot.keyframe <= e else (s + e) / 2
                units.append(Unit(a, shot, s, e, float(center), shot.motion))
                pos = e
                qi += 1
    imgs = [a for a in assets if a.kind == "image"]
    while qi < len(wants) and imgs:
        a = imgs[qi % len(imgs)]
        units.append(Unit(a, None, 0.0, 0.0, 0.0, a.energy))
        qi += 1
    return units


def _usable_length(assets: list[Asset]) -> float:
    return sum(max(sh.end - sh.start, 0.0)
               for a in assets if a.kind == "video" for sh in a.shots)


def _structure_flags(bgm: BgmInfo, start_beat: int, counts: list[int]) -> list[bool]:
    """每段的开始是否落在音乐结构边界（能量阶跃点：前后 2s 平均能量突变）。"""
    flags: list[bool] = []
    beat_idx = start_beat
    for i, c in enumerate(counts):
        t = float(bgm.beats[min(beat_idx, len(bgm.beats) - 1)])
        before = bgm.energy_at(t - 2.0, win=1.5)
        after = bgm.energy_at(t + 1.0, win=1.5)
        flags.append(i > 0 and abs(after - before) > 0.22)
        beat_idx += c
    return flags


def _feat_of(u: Unit, energy: float, boundary: bool) -> dict:
    """把素材窗口转成编排需要的特征字典。"""
    if u.shot is not None:
        sh = u.shot
        return dict(energy=energy, brightness=sh.brightness,
                    colorfulness=sh.colorfulness,
                    subject_center=sh.subject_center, motion=sh.motion,
                    motion_dir=sh.motion_dir, shot_type=sh.shot_type,
                    kind="video", boundary=boundary)
    return dict(energy=energy, brightness=u.asset.brightness,
                colorfulness=u.asset.colorfulness, subject_center=0.0,
                motion=0.0, motion_dir=0.0, shot_type="image",
                kind="image", boundary=boundary)


def _beat_counts(bgm: BgmInfo, start_beat: int, n_seg: int, mood: str) -> list[int]:
    """按音乐能量决定每段跨越几拍：能量越高切得越密。"""
    counts: list[int] = []
    idx = start_beat
    for _ in range(n_seg):
        t = float(bgm.beats[min(idx, len(bgm.beats) - 1)])
        e = bgm.energy_at(t)
        if mood == "energetic":
            b = 1 if e > 0.72 else (2 if e > 0.45 else 4)
        elif mood == "chill":
            b = 2 if e > 0.62 else (3 if e > 0.38 else 4)
        else:
            b = 3 if e > 0.55 else 4
        remain = len(bgm.beats) - 1 - idx
        if remain <= 0:
            break
        b = int(min(b, remain))
        counts.append(b)
        idx += b
    return counts


def build_timeline(assets: list[Asset], bgm: BgmInfo, mood: str,
                   target_duration: float | None = None,
                   trans_dur: float | None = None,
                   allow_reuse: bool = False,
                   preserve_order: bool = False,
                   bgm_start: float | None = None,
                   style: str = "auto",
                   intensity: float = 1.0,
                   seed: int | None = None,
                   camera_motion: str = "off") -> Timeline:
    bps = BEATS_PER_SEG.get(mood, 3)
    trans_dur = trans_dur or TRANS_DUR[mood]
    beat_interval = 60.0 / max(bgm.bpm, 40.0)
    seg_len = beat_interval * bps
    pad = trans_dur

    style = resolve_style(style, mood)
    amp = style_amp(style, intensity)
    if camera_motion == "subtle":
        amp *= 0.5                       # 温和档：运动幅度减半
    rng = random.Random(seed)

    # ---- 段数上限：不复用时最多能切出多少个窗口 ----
    usable = _usable_length(assets)
    n_img = sum(1 for a in assets if a.kind == "image")
    est_win = int(usable / max(seg_len + pad, 0.3)) + (n_img if usable < 1 else 0)
    if allow_reuse:
        n_seg = (int(round(target_duration / seg_len)) if target_duration
                 else int(round(usable * 1.5 / seg_len)))
        n_seg = max(4, min(MAX_SEGMENTS, n_seg))
    else:
        n_seg = (int(round(target_duration / seg_len)) if target_duration
                 else max(2, est_win))
        n_seg = max(2, min(MAX_SEGMENTS, n_seg))

    # ---- BGM 窗口：在节拍网格上搜索最贴合的片段 ----
    # 目标时长 ≈ 段数 × 每段基础时长；搜索会同时照顾「包含高潮」「时长贴合」
    # 「能量够高」「起点落在小节头」，因此成片既踩在高潮上，变速比也接近 1。
    if bgm_start is not None:
        start_beat = int(np.clip(np.searchsorted(bgm.beats, bgm_start),
                                 0, max(0, len(bgm.beats) - 2)))
    else:
        start_beat, _ = bgm.best_window(max(n_seg * seg_len, 3.0))

    counts = _beat_counts(bgm, start_beat, n_seg, mood)
    while len(counts) < 2:                      # 极端短曲兜底
        counts = [2, 2]
    n_seg = len(counts)

    # ---- 变速比：BGM 窗口时长 → 目标成片时长 ----
    beat_dur_guess = beat_interval              # 先按原速估算段长
    seg_durs = [c * beat_dur_guess for c in counts]
    wants = [sd + pad for sd in seg_durs]

    # 变长切分素材窗口；切不够就缩减段数
    units = _collect_units(assets, wants)
    if not units:
        raise ValueError("没有可用素材单元")
    if len(units) < n_seg:
        counts = counts[:len(units)]
        n_seg = len(counts)
        seg_durs = seg_durs[:n_seg]
    need_beats = int(sum(counts))

    bgm_start_t = float(bgm.beats[start_beat])
    end_idx = min(start_beat + need_beats, len(bgm.beats) - 1)
    bgm_end_t = float(bgm.beats[end_idx])
    bgm_span = max(bgm_end_t - bgm_start_t, 0.5)

    desired = target_duration if target_duration else bgm_span
    ratio = float(np.clip(bgm_span / max(desired, 1.0), 0.8, 1.25))
    beat_dur = beat_interval / ratio            # 变速后每一拍占多少秒
    seg_durs = [c * beat_dur for c in counts]

    # ---- 素材分配 ----
    hi_pos = int(n_seg * 0.6)          # 高潮位置（高能镜头放这里）
    if preserve_order:
        slots = list(units[:n_seg])
        while len(slots) < n_seg:
            slots.append(units[len(slots) % len(units)])
    else:
        order = sorted(range(len(units)), key=lambda i: -units[i].motion)
        used: set[int] = set()
        slots: list[Unit | None] = [None] * n_seg
        slot_pref = sorted(range(n_seg), key=lambda s: abs(s - hi_pos))
        for ui in order[:n_seg]:
            for s in slot_pref:
                if slots[s] is None:
                    slots[s] = units[ui]
                    used.add(ui)
                    break
        rest = [u for i, u in enumerate(units) if i not in used]
        ri = 0
        for s in range(n_seg):
            if slots[s] is None:
                slots[s] = rest[ri % len(rest)] if rest else units[s % len(units)]
                ri += 1
        if not allow_reuse:
            seen: set[int] = set()
            for s, u in enumerate(slots):
                uid = id(u)
                while uid in seen and ri < len(rest):
                    slots[s] = rest[ri]
                    uid = id(slots[s])
                    ri += 1
                seen.add(uid)

    # ---- 手法编排（素材感知 + 音乐结构 + 跳切/匹配剪辑）----
    boundaries = _structure_flags(bgm, start_beat, counts)
    feats = []
    for i, u in enumerate(slots):
        t = float(bgm.beats[min(start_beat + sum(counts[:i]), len(bgm.beats) - 1)])
        f = _feat_of(u, bgm.energy_at(t), boundaries[i])
        f["section"] = bgm.section_at(t)
        # 跳切：与上一段来自同一镜头、且是紧接着的画面窗口
        if i > 0 and u.shot is not None and slots[i - 1].shot is u.shot:
            f["jump"] = abs(u.s - slots[i - 1].e) < 0.05
        else:
            f["jump"] = False
        feats.append(f)
    plan = plan_shots(feats, mood, style, rng, camera_motion=camera_motion)
    energies = [f["energy"] for f in feats]

    # 逐边界的实际转场时长（跳切≈硬切、硬档更短促）
    tdurs = [float(p.get("tdur") or trans_dur) for p in plan]

    def _pad(i: int) -> float:
        """第 i 段两端转场重叠量的一半之和。"""
        left = tdurs[i - 1] / 2 if i > 0 else 0.0
        right = tdurs[i] / 2 if i < n_seg - 1 else 0.0
        return left + right

    # ---- 生成 EDL ----
    segments: list[Segment] = []
    for i, u in enumerate(slots):
        seg_dur = seg_durs[i]
        want = seg_dur + _pad(i)
        p = plan[i]
        if u.asset.kind == "video":
            # 窗口已按需求变长切分，直接从起点取，余量利用最大化
            win_len = u.e - u.s
            win = min(want, max(win_len, 0.3))
            src_in = float(u.s)
            avail = win_len
            # 速度收敛：素材余量不足则轻微慢放补足（下限 0.7x，再不够才定格补帧）
            speed = float(p["speed"])
            ramp = p.get("ramp")
            need = win * speed
            if ramp:                        # 段内变速：按等效速度估算素材需求
                a, b = ramp
                eff = 2 * a * b / (a + b)
                need = win * eff
            if avail < need and not ramp:
                speed = max(0.7, avail / max(win, 0.05))
            seg = Segment(asset=u.asset, src_in=src_in, duration=float(win),
                          transition=p["transition"], grade=p["grade"],
                          move=p["move"], fx=p["fx"], speed=speed,
                          ramp=ramp, tdur=tdurs[i], energy=energies[i],
                          why=p.get("why", ""), ken_burns_in=True)
            seg._avail = float(avail)  # noqa: SLF001
        else:
            seg = Segment(asset=u.asset, src_in=0.0, duration=float(want),
                          transition=p["transition"], grade=p["grade"],
                          move=p["move"], fx=p["fx"], speed=1.0, ramp=None,
                          tdur=tdurs[i], energy=energies[i],
                          why=p.get("why", ""),
                          ken_burns_in=(i % 2 == 0))
            seg._avail = float(want)  # noqa: SLF001
        segments.append(seg)

    return Timeline(segments=segments, bgm=bgm, bgm_start=bgm_start_t,
                    bgm_end=bgm_end_t, tempo_ratio=ratio,
                    trans_dur=trans_dur, mood=mood, style=style, amp=amp,
                    bgm_sections=list(bgm.sections or []))
