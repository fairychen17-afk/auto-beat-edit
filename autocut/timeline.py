"""时间线编排：需求-供给匹配，把素材窗口吸附到 BGM 节拍/小节网格，生成 EDL。

核心机制（与旧版的本质区别）：
1. **张力驱动段长**：每段跨几拍由【张力 + 张力梯度】决定（build 渐密、
   drop 最密、回落放长），而不是瞬时能量阈值；切点优先落在小节头/乐句
   边界，画面段落与音乐段落对齐。
2. **需求-供给匹配**：每个槽位有需求画像（张力/乐句角色/结构边界），每个
   素材窗口有供给画像（画质分/运动/长度）。分配是逐对打分的贪心匹配——
   高张力段优先拿高质量窗口，而不是旧版的「motion 排序 + 固定高潮位 0.6」。
3. **手法评分制**：plan_shots 对每个候选手法打分取最高分（确定性），
   rng 仅用于平分。详见 moves.py。
"""
from __future__ import annotations

import random

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .audio_analyzer import BEATS_PER_BAR, BARS_PER_PHRASE, BgmInfo
from .moves import plan_shots, resolve_style, style_amp
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
    energy: float = 0.5    # 该段所处音乐瞬时能量（兼容字段）
    tension: float = 0.5   # 该段目标张力 0~1（决策主信号）
    quality: float = 0.5   # 该段素材窗口画质分 0~1
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
    score: float = 0.5        # 供给分：画质为主、内容活力为辅


def _unit_score(asset: Asset, shot) -> float:
    """窗口供给分：画质（清晰度×曝光）为主，运动活力为辅。

    只用可实现的信号：画质评估 + 运动分析。不含任何语义假设。
    """
    if shot is None:
        # 图片：画质存在 shots[0] 上
        q = asset.shots[0].quality if asset.shots else 0.5
        return float(np.clip(q, 0.0, 1.0))
    return float(np.clip(0.7 * shot.quality + 0.3 * shot.motion, 0.0, 1.0))


def _collect_units(assets: list[Asset], wants: list[float]) -> list[Unit]:
    """按每段的需求长度依次从素材里切窗口（变长切分，利用率最高）。"""
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
                units.append(Unit(a, shot, s, e, float(center), shot.motion,
                                  _unit_score(a, shot)))
                pos = e
                qi += 1
    imgs = [a for a in assets if a.kind == "image"]
    while qi < len(wants) and imgs:
        a = imgs[qi % len(imgs)]
        units.append(Unit(a, None, 0.0, 0.0, 0.0, a.energy, _unit_score(a, None)))
        qi += 1
    return units


def _usable_length(assets: list[Asset]) -> float:
    return sum(max(sh.end - sh.start, 0.0)
               for a in assets if a.kind == "video" for sh in a.shots)


def _structure_flags(bgm: BgmInfo, start_beat: int, counts: list[int]) -> list[bool]:
    """每段的开始是否落在音乐结构边界（直接读修正后的 sections 边界）。

    旧版用「前后 2s 能量阶跃」现算，与 sections 是两套数据源；
    现在统一：段落起点距某个 section 边界不足一拍即视为结构边界。
    """
    beat_interval = 60.0 / max(bgm.bpm, 40.0)
    bounds = [float(s) for s, _, _ in (bgm.sections or [])[1:]]
    flags: list[bool] = []
    beat_idx = start_beat
    for i, c in enumerate(counts):
        t = float(bgm.beats[min(beat_idx, len(bgm.beats) - 1)])
        flags.append(i > 0 and any(abs(t - b) <= beat_interval for b in bounds))
        beat_idx += c
    return flags


def _feat_of(u: Unit, energy: float, boundary: bool) -> dict:
    """把素材窗口转成编排需要的特征字典（供给画像）。"""
    if u.shot is not None:
        sh = u.shot
        return dict(energy=energy, brightness=sh.brightness,
                    colorfulness=sh.colorfulness,
                    subject_center=sh.subject_center, motion=sh.motion,
                    motion_dir=sh.motion_dir, shot_type=sh.shot_type,
                    quality=sh.quality, kind="video", boundary=boundary)
    q = u.asset.shots[0].quality if u.asset.shots else 0.5
    return dict(energy=energy, brightness=u.asset.brightness,
                colorfulness=u.asset.colorfulness, subject_center=0.0,
                motion=0.0, motion_dir=0.0, shot_type="image",
                quality=q, kind="image", boundary=boundary)


def _target_beats(tension: float, grad: float, mood: str) -> int:
    """目标段长（拍）：张力高→短；张力上升再压一档（渐密），回落放长一档。"""
    if mood == "energetic":
        want = 1 if tension > 0.72 else (2 if tension > 0.45 else 4)
    elif mood == "chill":
        want = 2 if tension > 0.62 else (3 if tension > 0.38 else 4)
    else:
        want = 3 if tension > 0.55 else 4
    if grad > 0.04:
        want = max(1, want - 1)
    elif grad < -0.04:
        want = min(4, want + 1)
    return want


def _beat_counts(bgm: BgmInfo, start_beat: int, n_seg: int, mood: str) -> list[int]:
    """张力驱动 + 小节对齐的段长决策（确定性打分取最优，不再只看瞬时能量）。

    每个候选拍数 b ∈ 1..4 打分：
      - 主项：密度贴合 -0.5·|b - 目标拍数|（目标由张力+梯度决定）
      - 加分：切点落在小节头 +0.5 / 乐句边界再 +0.3（剪辑点与音乐段落对齐
        是优先于"恰好 N 拍"的硬约束，权重设计保证偏 1 拍也要对齐小节）
      - 减分：与上一段等长 -0.1（避免节奏单调，但不破坏对齐）
    """
    counts: list[int] = []
    idx = start_beat
    phase = getattr(bgm, "downbeat_phase", 0)
    n_beats = len(bgm.beats)
    for _ in range(n_seg):
        remain = n_beats - 1 - idx
        if remain <= 0:
            break
        t = float(bgm.beats[idx])
        want = _target_beats(bgm.tension_at(t), bgm.tension_grad_at(t), mood)
        # 当前若已偏离小节网格，重新对齐是硬需求（漂移会越滚越大）
        misalign = (idx - phase) % BEATS_PER_BAR
        resync = (BEATS_PER_BAR - misalign) % BEATS_PER_BAR
        best_b, best_s = None, -1e9
        for b in (1, 2, 3, 4):
            if b > remain:
                continue
            end_rel = idx + b - phase            # 相对小节网格的位置
            s = -0.5 * float(abs(b - want))
            if end_rel % BEATS_PER_BAR == 0:
                s += 0.5                          # 小节头
                if end_rel % (BEATS_PER_BAR * BARS_PER_PHRASE) == 0:
                    s += 0.3                      # 乐句边界再加分
            if resync and b == resync:
                s += 0.8                          # 偏离网格时优先重新对齐
            if counts and b == counts[-1]:
                s -= 0.1                          # 避免连续等长
            if s > best_s:
                best_s, best_b = s, b
        if best_b is None:
            break
        counts.append(best_b)
        idx += best_b
    return counts


def _match_units(slots_tension: list[float], wants: list[float],
                 units: list[Unit], allow_reuse: bool) -> list[Unit]:
    """需求-供给匹配：高张力槽位优先拿高质量、长度贴合的窗口。

    逐对打分（确定性贪心）：
      pair = 0.50 × quality × (0.5 + 0.5·tension)   好素材优先给高张力段
           + 0.25 × min(motion, tension)            运动与张力互相成全
           + 0.25 × (1 - |窗长 - 需求| / 需求)      长度贴合（少变速失真）
    槽位按张力降序处理；窗口不重复；allow_reuse 时窗口耗尽后清空占用
    重新轮转（第二轮仍按分数高低取用，而不是简单取模——保证复用也优先
    用好素材）。
    """
    n_seg = len(slots_tension)
    slots: list[Unit | None] = [None] * n_seg
    order = sorted(range(n_seg), key=lambda s: -slots_tension[s])
    taken: set[int] = set()
    for s in order:
        avail = [ui for ui in range(len(units)) if ui not in taken]
        if not avail:
            if allow_reuse:
                taken.clear()
                avail = list(range(len(units)))
            else:
                break
        best_ui, best_pair = -1, -1e9
        for ui in avail:
            u = units[ui]
            want = max(wants[s], 0.3)
            length_fit = 1.0 - min(abs((u.e - u.s) - want) / want, 1.0) \
                if u.shot is not None else 0.8
            pair = (0.50 * u.score * (0.5 + 0.5 * slots_tension[s])
                    + 0.25 * min(u.motion, slots_tension[s])
                    + 0.25 * length_fit)
            if pair > best_pair:
                best_pair, best_ui = pair, ui
        slots[s] = units[best_ui]
        taken.add(best_ui)
    return [u for u in slots if u is not None]


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

    # ---- BGM 窗口：在节拍网格上搜索最贴合的片段（含张力峰值覆盖）----
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
    seg_durs = [c * beat_interval for c in counts]
    wants = [sd + pad for sd in seg_durs]

    # 变长切分素材窗口；切不够就缩减段数
    units = _collect_units(assets, wants)
    if not units:
        raise ValueError("没有可用素材单元")
    if len(units) < n_seg and not allow_reuse:
        counts = counts[:len(units)]
        n_seg = len(counts)
        seg_durs = seg_durs[:n_seg]
        wants = wants[:n_seg]
    need_beats = int(sum(counts))

    bgm_start_t = float(bgm.beats[start_beat])
    end_idx = min(start_beat + need_beats, len(bgm.beats) - 1)
    bgm_end_t = float(bgm.beats[end_idx])
    bgm_span = max(bgm_end_t - bgm_start_t, 0.5)

    desired = target_duration if target_duration else bgm_span
    ratio = float(np.clip(bgm_span / max(desired, 1.0), 0.8, 1.25))
    beat_dur = beat_interval / ratio            # 变速后每一拍占多少秒
    seg_durs = [c * beat_dur for c in counts]

    # ---- 需求画像：每个槽位的张力 / 梯度 / 乐句角色 ----
    slot_times = []
    beat_idx = start_beat
    for c in counts:
        slot_times.append(float(bgm.beats[min(beat_idx, len(bgm.beats) - 1)]))
        beat_idx += c
    tensions = [bgm.tension_at(t) for t in slot_times]
    grads = [bgm.tension_grad_at(t) for t in slot_times]
    phrase_starts = [bgm.phrase_position(t)[2] for t in slot_times]

    # ---- 素材分配 ----
    if preserve_order:
        slots = list(units[:n_seg])
        while len(slots) < n_seg:
            slots.append(units[len(slots) % len(units)])
    else:
        slots = _match_units(tensions, wants, units, allow_reuse)
        while len(slots) < n_seg:               # 兜底（理论上不会触发）
            slots.append(units[len(slots) % len(units)])

    # ---- 手法编排（评分制 + 乐句结构 + 跳切/匹配剪辑）----
    boundaries = _structure_flags(bgm, start_beat, counts)
    feats = []
    for i, u in enumerate(slots):
        t = slot_times[i]
        f = _feat_of(u, bgm.energy_at(t), boundaries[i])
        f["section"] = bgm.section_at(t)
        f["tension"] = tensions[i]
        f["tension_grad"] = grads[i]
        f["phrase_start"] = phrase_starts[i]
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
            win_len = u.e - u.s
            win = min(want, max(win_len, 0.3))
            src_in = float(u.s)
            avail = win_len
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
                          tension=tensions[i], quality=u.score,
                          why=p.get("why", ""), ken_burns_in=True)
            seg._avail = float(avail)  # noqa: SLF001
        else:
            seg = Segment(asset=u.asset, src_in=0.0, duration=float(want),
                          transition=p["transition"], grade=p["grade"],
                          move=p["move"], fx=p["fx"], speed=1.0, ramp=None,
                          tdur=tdurs[i], energy=energies[i],
                          tension=tensions[i], quality=u.score,
                          why=p.get("why", ""),
                          ken_burns_in=(i % 2 == 0))
            seg._avail = float(want)  # noqa: SLF001
        segments.append(seg)

    return Timeline(segments=segments, bgm=bgm, bgm_start=bgm_start_t,
                    bgm_end=bgm_end_t, tempo_ratio=ratio,
                    trans_dur=trans_dur, mood=mood, style=style, amp=amp,
                    bgm_sections=list(bgm.sections or []))
