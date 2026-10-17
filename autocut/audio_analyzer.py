"""BGM 音频分析：节拍 / 小节 / 乐句三级对齐 + 张力曲线 + 高潮 + 情绪标签。

数据对齐原则（本模块的核心约定，历史上最严重的 bug 就出在这里）：
- librosa 的 onset 包络是【逐帧】序列（帧率 ≈ sr/hop ≈ 43 帧/秒）；
- beats / downbeats 是【时间戳】序列（单位：秒）；
- 任何「按节拍 / 小节 / 乐句」的统计，都必须先把逐帧包络按**时间戳**
  映射到对应节拍领地（territory = 相邻两拍中点之间的区间）再聚合，
  绝不允许用节拍序号直接索引包络帧（frames ≠ beats）。
"""
from __future__ import annotations

import librosa
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

BEATS_PER_BAR = 4          # 按 4/4 拍处理（流行/电子乐的绝大多数情况）
BARS_PER_PHRASE = 4        # 一个乐句 = 4 小节 = 16 拍


def beat_energies(env: np.ndarray, fps: float, beats: np.ndarray,
                  duration: float) -> np.ndarray:
    """把逐帧能量包络映射为逐拍能量（按时间戳，而非按帧号）。

    第 i 拍的领地 = [mid(beats[i-1], beats[i]), mid(beats[i], beats[i+1]))，
    首尾拍分别扩展到 0 和 duration。领地内取包络均值作为该拍能量。
    """
    n = len(beats)
    if n == 0 or len(env) == 0 or fps <= 0:
        return np.zeros(n)
    mids = (beats[:-1] + beats[1:]) / 2.0 if n > 1 else np.array([])
    edges = np.concatenate([[0.0], mids, [duration]])
    out = np.zeros(n)
    for i in range(n):
        f0 = int(np.clip(edges[i] * fps, 0, len(env) - 1))
        f1 = int(np.clip(edges[i + 1] * fps, f0 + 1, len(env)))
        out[i] = float(env[f0:f1].mean())
    return out


def _align_downbeat_phase(beat_e: np.ndarray) -> int:
    """在 0..3 四个相位中搜索真正的 downbeat 位置。

    4/4 拍里小节头（第 1 拍）的 onset 通常最强：对每个相位 offset，
    统计 beats[offset::4] 的平均能量，取最大者。这样即使节拍检测器
    漏掉开头某个 downbeat，小节网格也能对齐到真实的重音上。
    """
    if len(beat_e) < 8:
        return 0
    best_phase, best_e = 0, -1.0
    for phase in range(BEATS_PER_BAR):
        e = float(beat_e[phase::BEATS_PER_BAR].mean())
        if e > best_e:
            best_phase, best_e = phase, e
    return best_phase


@dataclass
class BgmInfo:
    path: Path
    duration: float
    bpm: float
    beats: np.ndarray          # 全部节拍时间（秒）
    strong_beats: np.ndarray   # downbeat（小节头，经相位对齐）
    climax: float              # 能量峰值（高潮）时间
    energy_mean: float         # 0~1
    mood: str                  # "calm" | "chill" | "energetic"
    onset_env: np.ndarray = None   # 归一化 0~1 的 onset 能量包络（逐帧）
    onset_fps: float = 0.0         # 包络帧率（帧/秒）
    sections: list = None          # 结构段落 [(start, end, 能量等级), ...]
    beat_e: np.ndarray = None      # 逐拍能量（由包络按时间戳映射得到）
    bar_times: np.ndarray = None   # 每个小节的起始时间（downbeat）
    bar_e: np.ndarray = None       # 逐小节能量（= 4 拍能量均值）
    tension_t: np.ndarray = None   # 张力曲线采样时间（小节中心）
    tension_v: np.ndarray = None   # 张力值 0~1（平滑后）
    downbeat_phase: int = 0        # 小节网格相位（beats[phase] 是第 1 小节头）

    # ------------------------------------------------------------ 逐帧能量
    def energy_at(self, t: float, win: float = 0.35) -> float:
        """t 时刻附近 ±win/2 秒的相对能量（0~1），用于驱动剪辑密度。"""
        if self.onset_env is None or len(self.onset_env) == 0:
            return self.energy_mean
        i0 = int(max(0.0, (t - win / 2)) * self.onset_fps)
        i1 = int(min(self.duration, (t + win / 2)) * self.onset_fps)
        i0 = min(i0, len(self.onset_env) - 1)
        i1 = min(max(i1, i0 + 1), len(self.onset_env))
        return float(np.clip(self.onset_env[i0:i1].mean(), 0.0, 1.0))

    def energy_curve(self, times) -> np.ndarray:
        return np.array([self.energy_at(float(t)) for t in times])

    # ------------------------------------------------------------ 节拍/小节
    def beat_energy(self) -> np.ndarray:
        """逐拍能量（已按时间戳对齐；缓存友好，供窗口搜索用）。"""
        if self.beat_e is None:
            self.beat_e = beat_energies(self.onset_env, self.onset_fps,
                                        self.beats, self.duration)
        return self.beat_e

    def tension_at(self, t: float) -> float:
        """t 时刻的目标张力 0~1（乐句级平滑曲线，非瞬时能量）。

        张力与瞬时能量的区别：瞬时能量在一拍内剧烈抖动，张力是全曲结构
        的慢变量——build-up 张力持续爬升、drop 见顶、尾声回落。剪辑密度、
        特效强度跟随张力，画面才有「段落感」而不是「每拍一惊一乍」。
        """
        if self.tension_t is None or len(self.tension_t) == 0:
            return self.energy_at(t)
        return float(np.interp(t, self.tension_t, self.tension_v))

    def tension_grad_at(self, t: float) -> float:
        """t 时刻的张力梯度（每秒变化量）：>0 正在 build，<0 正在回落。"""
        if self.tension_t is None or len(self.tension_t) < 2:
            return 0.0
        i = int(np.searchsorted(self.tension_t, t))
        i = int(np.clip(i, 1, len(self.tension_t) - 1))
        dt = self.tension_t[i] - self.tension_t[i - 1]
        if dt <= 0:
            return 0.0
        return float((self.tension_v[i] - self.tension_v[i - 1]) / dt)

    def phrase_position(self, t: float) -> tuple[int, int, bool]:
        """t 所在的 (乐句号, 乐句内第几小节 0~3, 是否乐句首小节)。"""
        if self.bar_times is None or len(self.bar_times) == 0:
            return (0, 0, False)
        bar = int(np.searchsorted(self.bar_times, t, side="right")) - 1
        bar = max(bar, 0)
        phrase, bar_in = divmod(bar, BARS_PER_PHRASE)
        return (phrase, bar_in, bar_in == 0)

    def section_at(self, t: float) -> str:
        for s, e, lvl in (self.sections or []):
            if s <= t < e:
                return lvl
        return "mid"

    # ------------------------------------------------------------ 窗口搜索
    def best_window(self, target: float, min_span: float = 3.0) -> tuple[int, int]:
        """在节拍网格上搜索最贴合需求的片段，返回 (起始拍索引, 结束拍索引)。

        打分综合考虑：
        - **时长贴合**：窗口时长尽量接近 target（时长越贴，变速比越接近 1，音质无损）
        - **张力覆盖**：窗口应包含全曲张力峰值（副歌/drop）
        - **能量水平**：窗口内平均张力高（高潮段更适合当卡点背景）
        - **小节头起点**：起点落在 downbeat 上，段落感更整齐
        """
        beats = self.beats
        if len(beats) < 4:
            return 0, max(1, len(beats) - 1)
        e = self.beat_energy()
        cum = np.concatenate([[0.0], np.cumsum(e)])
        strong = set(np.searchsorted(beats, self.strong_beats).tolist())
        # 张力峰值位置（结构级高潮，比瞬时 RMS 峰值更稳）
        peak_t = (self.tension_t[int(np.argmax(self.tension_v))]
                  if self.tension_t is not None and len(self.tension_t)
                  else self.climax)

        target = max(target, min_span)
        best, best_score = (0, max(1, min(len(beats) - 1, 2))), -1e9
        for i in range(len(beats) - 1):
            j = int(np.searchsorted(beats, beats[i] + target))
            j = min(max(j, i + 2), len(beats) - 1)
            span = float(beats[j] - beats[i])
            if span < min_span:
                continue
            span_fit = 1.0 - min(abs(span - target) / target, 1.0)
            mid = (beats[i] + beats[j]) / 2
            if beats[i] <= peak_t <= beats[j]:
                cover = 1.0
            else:
                cover = max(0.0, 1.0 - abs(mid - peak_t) / max(target, 1.0))
            level = float((cum[j] - cum[i]) / max(j - i, 1))
            head = 0.3 if i in strong else 0.0
            score = 1.6 * cover + 1.3 * span_fit + 0.9 * level + head
            if score > best_score:
                best, best_score = (i, j), score
        return best

    def window_span(self, i: int, j: int) -> float:
        return float(self.beats[j] - self.beats[i])


def _detect_sections(beats: np.ndarray, beat_e: np.ndarray,
                     bar_times: np.ndarray, bar_e: np.ndarray,
                     duration: float) -> list[tuple[float, float, str]]:
    """按【小节能量】把曲子切成「低/中/高」结构段落（近似 前奏/build/drop/间奏/尾声）。

    输入必须是已对齐的逐小节能量（bar_e / bar_times），时间边界取小节头
    时间戳。相邻同档合并；短于 1 个乐句一半（2 小节）的碎片并入邻居。
    """
    if bar_times is None or len(bar_e) < 2:
        return [(0.0, duration, "mid")]
    spread = float(bar_e.max() - bar_e.min())
    if spread < 0.05:                      # 能量近乎平坦 → 无结构可分
        return [(0.0, duration, "mid")]
    q1, q2 = np.percentile(bar_e, [38, 68])
    # 闭区间取值：二值化信号（低/高两组）下分位数恰等于组值，
    # 开区间会把所有小节都误判为 mid
    levels = ["low" if v <= q1 else ("high" if v >= q2 else "mid")
              for v in bar_e]

    secs: list[list] = []
    for k, lv in enumerate(levels):
        s = float(bar_times[k])
        e = (float(bar_times[k + 1]) if k + 1 < len(bar_times)
             else float(duration))
        if secs and secs[-1][2] == lv:
            secs[-1][1] = e
        else:
            secs.append([s, e, lv])
    # 过短段落并入前一段，避免结构碎片化（阈值：约 2 小节的时长）
    min_sec = max(2.0, 2 * np.median(np.diff(bar_times))) if len(bar_times) > 2 else 2.0
    merged: list[list] = []
    for s, e, lv in secs:
        if merged and (e - s) < min_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e, lv])
    # 合并后可能出现相邻同档，再收敛一次
    out: list[list] = []
    for s, e, lv in merged:
        if out and out[-1][2] == lv:
            out[-1][1] = e
        else:
            out.append([s, e, lv])
    return [(float(s), float(e), lv) for s, e, lv in out]


def _tension_curve(bar_times: np.ndarray, bar_e: np.ndarray,
                   duration: float) -> tuple[np.ndarray, np.ndarray]:
    """由小节能量生成全曲张力曲线（平滑 + 归一化到 0~1）。

    张力 = 结构级能量趋势：对小节能量做 ±1 小节的滑动平均消除瞬时抖动，
    再用乐句内相对位置微调（乐句末小节张力略降，给「回合感」）。
    """
    n = len(bar_e)
    if n == 0:
        return np.array([0.0, duration]), np.array([0.5, 0.5])
    k = min(3, n)
    kernel = np.ones(k) / k
    sm = np.convolve(bar_e, kernel, mode="same")
    # 乐句回合感：每个乐句最后一个小节张力轻微衰减
    for bar in range(n):
        if bar % BARS_PER_PHRASE == BARS_PER_PHRASE - 1:
            sm[bar] *= 0.92
    lo, hi = float(sm.min()), float(sm.max())
    v = (sm - lo) / (hi - lo + 1e-9)
    # 采样点取小节中心，端点补齐以便 np.interp 外推稳定
    centers = []
    for i, t0 in enumerate(bar_times):
        t1 = float(bar_times[i + 1]) if i + 1 < n else duration
        centers.append((float(t0) + t1) / 2)
    t = np.concatenate([[0.0], np.array(centers), [duration]])
    vv = np.concatenate([[v[0]], v, [v[-1]]])
    return t, vv.astype(np.float64)


def analyze_bgm(path: Path, sr: int = 22050) -> BgmInfo:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) == 0:
        beats = np.arange(0.0, duration, 60.0 / max(bpm, 40.0))

    # 归一化 onset 能量包络（逐帧；一切节拍级统计的时间戳来源）
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    env = onset.astype(np.float32)
    if env.max() > 0:
        env = env / (env.max() + 1e-9)
    env_fps = float(len(env) / max(duration, 1e-3))

    # ---- 逐帧 → 逐拍（按时间戳映射，这是修正对齐 bug 的核心一步）----
    beat_e = beat_energies(env, env_fps, beats, duration)

    # ---- 小节对齐：搜索 downbeat 相位，而不是盲目从 beats[0] 起每 4 拍一组 ----
    phase = _align_downbeat_phase(beat_e)
    downbeat_idx = np.arange(phase, len(beats), BEATS_PER_BAR)
    strong_beats = beats[downbeat_idx] if len(downbeat_idx) else beats
    bar_times = strong_beats
    bar_e = np.array([beat_e[i:i + BEATS_PER_BAR].mean()
                      for i in downbeat_idx]) if len(downbeat_idx) else np.array([])

    # ---- 高潮位置（5s 滑窗 RMS 峰值中心，时间驱动，无对齐问题）----
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    win = max(1, int(5.0 / (times[1] - times[0]))) if len(times) > 1 else 1
    smooth = np.convolve(rms, np.ones(win) / win, mode="same")
    climax = float(times[int(np.argmax(smooth))]) if len(times) else duration / 2
    energy_mean = float(np.clip(rms.mean() / (rms.max() + 1e-9) * 1.6, 0, 1))

    if bpm < 85:
        mood = "calm"
    elif bpm < 125:
        mood = "chill"
    else:
        mood = "energetic"
    if energy_mean > 0.65 and mood == "chill":
        mood = "energetic"
    elif energy_mean < 0.3 and mood == "chill":
        mood = "calm"

    # ---- 张力曲线 + 结构段落（全部基于已对齐的小节能量）----
    tension_t, tension_v = _tension_curve(bar_times, bar_e, duration)
    sections = _detect_sections(beats, beat_e, bar_times, bar_e, duration)

    return BgmInfo(path=path, duration=duration, bpm=bpm, beats=beats,
                   strong_beats=strong_beats, climax=climax,
                   energy_mean=energy_mean, mood=mood,
                   onset_env=env, onset_fps=env_fps, sections=sections,
                   beat_e=beat_e, bar_times=bar_times, bar_e=bar_e,
                   tension_t=tension_t, tension_v=tension_v,
                   downbeat_phase=phase)


def scan_bgm_library(bgm_dir: Path) -> list[BgmInfo]:
    out = []
    for p in sorted(bgm_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXTS:
            try:
                out.append(analyze_bgm(p))
            except Exception as e:
                print(f"[warn] BGM 分析失败 {p.name}: {e}")
    return out
