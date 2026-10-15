"""BGM 音频分析：节拍点、BPM、能量曲线、高潮位置、情绪标签。"""
from __future__ import annotations

import librosa
import numpy as np
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class BgmInfo:
    path: Path
    duration: float
    bpm: float
    beats: np.ndarray          # 全部节拍时间（秒）
    strong_beats: np.ndarray   # 强拍（每小节第一拍近似）
    climax: float              # 能量峰值（高潮）时间
    energy_mean: float         # 0~1
    mood: str                  # "calm" | "chill" | "energetic"
    onset_env: np.ndarray = None   # 归一化 0~1 的 onset 能量包络
    onset_fps: float = 0.0         # 包络帧率（帧/秒）
    sections: list = None          # 结构段落 [(start, end, 能量等级), ...]

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

    def beat_energy(self) -> np.ndarray:
        """每个节拍点的能量（缓存友好，供窗口搜索用）。"""
        if getattr(self, "_beat_e", None) is None:
            self._beat_e = self.energy_curve(self.beats)
        return self._beat_e

    def best_window(self, target: float, min_span: float = 3.0) -> tuple[int, int]:
        """在节拍网格上搜索最贴合需求的片段，返回 (起始拍索引, 结束拍索引)。

        打分综合考虑：
        - **时长贴合**：窗口时长尽量接近 target（时长越贴，变速比越接近 1，音质无损）
        - **高潮覆盖**：窗口应包含全曲能量峰值（副歌/drop）
        - **能量水平**：窗口内平均能量高（高潮段更适合当卡点背景）
        - **强拍起点**：起点落在小节头（强拍）上，段落感更整齐
        """
        beats = self.beats
        if len(beats) < 4:
            return 0, max(1, len(beats) - 1)
        e = self.beat_energy()
        cum = np.concatenate([[0.0], np.cumsum(e)])
        strong = set(np.searchsorted(beats, self.strong_beats).tolist())

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
            if beats[i] <= self.climax <= beats[j]:
                cover = 1.0
            else:
                cover = max(0.0, 1.0 - abs(mid - self.climax) / max(target, 1.0))
            level = float((cum[j] - cum[i]) / max(j - i, 1))
            head = 0.3 if i in strong else 0.0
            score = 1.6 * cover + 1.3 * span_fit + 0.9 * level + head
            if score > best_score:
                best, best_score = (i, j), score
        return best

    def window_span(self, i: int, j: int) -> float:
        return float(self.beats[j] - self.beats[i])

    def section_at(self, t: float) -> str:
        for s, e, lvl in (self.sections or []):
            if s <= t < e:
                return lvl
        return "mid"


def _detect_sections(beats: np.ndarray, energy: np.ndarray,
                     duration: float) -> list[tuple[float, float, str]]:
    """按节拍能量把曲子切成「低/中/高」结构段落（近似 前奏/build/drop/间奏/尾声）。

    做法：每 4 拍（一小节）取平均能量，用三分位阈值定档，再把相邻同档合并，
    过短的段落并入邻居。用于让剪辑知道当前处在曲子的哪个部分。
    """
    if len(beats) < 8:
        return [(0.0, duration, "mid")]
    n_bar = len(beats) // 4
    if n_bar < 2:
        return [(0.0, duration, "mid")]
    bar_e = np.array([energy[k * 4: k * 4 + 4].mean() for k in range(n_bar)])
    q1, q2 = np.percentile(bar_e, [38, 68])
    levels = ["low" if v < q1 else ("high" if v > q2 else "mid") for v in bar_e]

    secs: list[list] = []
    for k, lv in enumerate(levels):
        s = float(beats[k * 4])
        e = (float(beats[(k + 1) * 4]) if (k + 1) * 4 < len(beats)
             else float(duration))
        if secs and secs[-1][2] == lv:
            secs[-1][1] = e
        else:
            secs.append([s, e, lv])
    # 过短段落（<3s）并入前一段，避免结构碎片化
    merged: list[list] = []
    for s, e, lv in secs:
        if merged and (e - s) < 3.0:
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


def analyze_bgm(path: Path, sr: int = 22050) -> BgmInfo:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) == 0:
        beats = np.arange(0.0, duration, 60.0 / max(bpm, 40.0))

    # 强拍：按 onset 强度在每 4 拍中选能量最高的一拍对齐，近似小节头
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = np.array([onset[f] if f < len(onset) else 0 for f in beat_frames]) \
        if len(beat_frames) else np.zeros(len(beats))
    strong_idx = []
    for i in range(0, len(beats), 4):
        grp = beat_strength[i:i + 4]
        if len(grp):
            strong_idx.append(i + int(np.argmax(grp)))
    strong_beats = beats[strong_idx] if strong_idx else beats

    # 能量曲线 → 高潮位置（5s 滑窗 RMS 峰值中心）
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

    # 归一化 onset 能量包络（供剪辑密度/手法强度使用）
    env = onset.astype(np.float32)
    if env.max() > 0:
        env = env / (env.max() + 1e-9)
    env_fps = float(len(env) / max(duration, 1e-3))

    return BgmInfo(path=path, duration=duration, bpm=bpm, beats=beats,
                   strong_beats=strong_beats, climax=climax,
                   energy_mean=energy_mean, mood=mood,
                   onset_env=env, onset_fps=env_fps,
                   sections=_detect_sections(beats, env, duration))


def scan_bgm_library(bgm_dir: Path) -> list[BgmInfo]:
    out = []
    for p in sorted(bgm_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXTS:
            try:
                out.append(analyze_bgm(p))
            except Exception as e:
                print(f"[warn] BGM 分析失败 {p.name}: {e}")
    return out
