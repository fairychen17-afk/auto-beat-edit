"""
游戏CG自动卡点剪辑（auto beat edit）

流程：音乐分析 -> （可选）自动选段 -> 拍点网格 -> 分段计划(EDL) -> 变速对齐 -> 特效 -> 拼接导出
适配 moviepy 2.x（不兼容 1.x）

时间语义约定（非常重要，三者不要混用）：
- 源时间 source time  ：素材文件自身的时间轴，subclipped(start, end) 使用它
- 片段时间 clip time  ：单段素材变速之后、拼接之前的时间轴
- 时间轴时间 timeline time：所有片段首尾相接后，成片的时间轴

换算关系（已在本机 moviepy 2.1.2 实测确认）：
    with_speed_scaled(speed) 的结果时长 = 原时长 / speed      # speed>1 加速变短
    speed           = source_duration / timeline_duration
    source_duration = timeline_duration * speed
    timeline_duration = source_duration / speed
"""

import json
import os
import re
import time as _time
from dataclasses import asdict, dataclass

import librosa
import numpy as np
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips, vfx


# ==========================================================================
# 配置
# ==========================================================================
FOLDER = r"C:\Users\甘雨\Desktop\虚照"
AUDIO_FILE = "@饺子WTF创作的原声_7656434362674048997(1).mp3"
OUTPUT_FILE = os.path.join(FOLDER, "虚照_卡点混剪_究极版.mp4")

# 音乐起始：给定数值则优先使用（用户指定优先）；
# 设为 None 时启用自动选段（见 AUTO_SELECT_AUDIO）
AUDIO_START = 54.0
AUDIO_DURATION = 10.0
AUTO_SELECT_AUDIO = False          # True 时忽略 AUDIO_START，自动挑选最佳区间

TARGET_RESOLUTION = (720, 960)
# 素材列表：留空则自动扫描文件夹并按自然序排序
CLIP_ORDER = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

MIN_SPEED = 0.8
MAX_SPEED = 1.3
AVG_SPEED_BASE = 1.06
AVG_SPEED_STEP = 0.015

# 剪辑策略档位：normal / slow / fast
EDIT_STRATEGY = "normal"
MIN_SHOT = 0.25                    # 最短镜头长度，低于该值的切点会被跳过
# 素材不够时的策略：loop=回绕复用（默认）/ error=直接报错
SHORT_SEGMENT_STRATEGY = "loop"

# 素材质量：按 quality_score 加权分配镜头（仅本地指标，不接云端 AI）
PREFER_QUALITY = True
QUALITY_SAMPLE_FRAMES = 5          # 每个素材采样多少帧做质量评估
QUALITY_MIN_WARN = 0.3            # 低于该质量分的素材给出提示

# 转场：cut 硬切（默认，卡点最准）/ dip 长片段淡入淡出黑场（时序安全，不改总时长）
TRANSITION_MODE = "cut"
TRANSITION_DUR = 0.12              # dip 模式下的淡入淡出时长（秒）

SYNTH_FALLBACK_BPM = 120.0         # 检测不到可靠拍点时的回退速度

FILTER_CONTRAST = 1.3
FILTER_SATURATION = 1.35
FILTER_BRIGHTNESS = 1.0
FILTER_VIGNETTE = 0.6
FILTER_TEAL_ORANGE = True
FILTER_FILM_GRAIN = 0.1

PULSE_INTENSITY = 0.08
SHAKE_INTENSITY = 3.0
ZOOM_SPEED = 0.12
GLITCH_INTENSITY = 6.0
MOTION_BLUR_INTENSITY = 0.4
HUE_SHIFT_AMOUNT = 15.0

# 新增究极效果
FISHEYE_ON = True
PIXEL_ON = True
FLARE_ON = True

# 剪辑计划（EDL）落盘
EXPORT_EDL = True
EDL_PATH = os.path.join(FOLDER, "edit_plan.json")

# 快速模式（仅调试用）：降低导出比特率
FAST_EXPORT = False


# ==========================================================================
# 数据结构
# ==========================================================================
@dataclass
class BeatGrid:
    """音乐拍点分析结果。"""
    beats: np.ndarray
    half_beats: np.ndarray
    tempo: float
    status: str = "ok"          # ok / few_beats_synthesized / empty_audio

    def __post_init__(self):
        self.beats = np.asarray(self.beats, dtype=float).flatten()
        self.half_beats = np.asarray(self.half_beats, dtype=float).flatten()

    @property
    def beat_duration(self):
        return 60.0 / max(self.tempo, 1e-6)


@dataclass
class EditSegment:
    """单个片段的剪辑计划（EDL 条目）。"""
    index: int
    clip_index: int
    clip_name: str
    label: str
    timeline_start: float
    timeline_end: float
    duration: float
    source_in: float
    source_out: float
    speed: float
    is_fast: bool
    reused: bool = False
    transition: str = "cut"
    note: str = ""


@dataclass
class AssetInfo:
    """轻量素材分析结果（仅本地指标，暂不接云端 AI）。

    质量分说明：sharpness/motion/quality_score 都是在本次素材集合内做 min-max 归一化后的
    **相对**指标，仅作「候选评分」，不代表绝对主观质量。
    """
    name: str
    duration: float
    width: int
    height: int
    aspect: float
    has_audio: bool
    estimated_usable: float = 0.0
    usable_ratio: float = 0.0
    sharpness_raw: float = 0.0       # 清晰度原始值（拉普拉斯方差均值）
    motion_raw: float = 0.0          # 运动量原始值（相邻采样帧平均差异/255）
    exposure_score: float = 0.0      # 曝光健康度 0~1（越接近 1 越好）
    sharpness: float = 0.0           # 归一化清晰度
    motion: float = 0.0              # 归一化运动量
    quality_score: float = 0.0       # 归一化综合质量 0~1
    candidate_score: float = 0.0     # 候选评分 = 时长可用度 + 质量
    note: str = ""


# ==========================================================================
# 时间换算（P0-2）
# ==========================================================================
def speed_for(source_duration, timeline_duration):
    """由「源时长 / 时间轴时长」求播放倍速。"""
    if timeline_duration <= 1e-9:
        raise ValueError("时间轴时长必须为正数")
    return source_duration / timeline_duration


def source_duration_for(timeline_duration, speed):
    """由「时间轴时长 × 倍速」求需要消耗的源时长。"""
    if speed <= 1e-9:
        raise ValueError("播放倍速必须为正数")
    return timeline_duration * speed


def timeline_duration_for(source_duration, speed):
    """变速后落在成片时间轴上的时长。"""
    if speed <= 1e-9:
        raise ValueError("播放倍速必须为正数")
    return source_duration / speed


def clamp_speed(speed, min_speed=MIN_SPEED, max_speed=MAX_SPEED):
    return float(min(max(speed, min_speed), max_speed))


def resolve_segment_timing(available, target_duration,
                           desired_speed, min_speed=MIN_SPEED, max_speed=MAX_SPEED):
    """在可用源时长约束下，求 (源时长, 倍速, 备注)。

    保证 source_duration <= available，尽量让变速后时长 == target_duration。
    素材实在不够（即使最慢速也填不满）时返回 note="footage_short"，由调用方决定策略。
    """
    if target_duration <= 1e-9:
        raise ValueError("目标时长必须为正数")

    speed = clamp_speed(min(desired_speed, max_speed), min_speed, max_speed)
    if source_duration_for(target_duration, speed) <= available:
        return source_duration_for(target_duration, speed), speed, ""

    # 素材不够，先在不低于 min_speed 的前提下尽量填满
    fit_speed = available / target_duration
    if fit_speed >= min_speed:
        return source_duration_for(target_duration, fit_speed), fit_speed, ""

    # 即使最慢也填不满
    return available, min_speed, "footage_short"


# ==========================================================================
# 音频分析（P0-4 / P0-5 / P1 第一项）
# ==========================================================================
def probe_audio_duration(audio_path):
    """返回音频总时长（秒）。失败时抛错并带上文件名。"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    try:
        clip = AudioFileClip(audio_path)
        try:
            return float(clip.duration)
        finally:
            clip.close()
    except Exception as exc:  # noqa: BLE001 - 需要把文件名带进错误信息
        raise RuntimeError(f"无法读取音频 {os.path.basename(audio_path)}: {exc}") from exc


def validate_audio_window(audio_path, start, duration, total_duration=None):
    """校验音乐起始/时长是否越界，返回实际可用区间 (start, end)。

    - start 越界 -> ValueError（明确报错，不再静默截断）
    - 区间超出音频尾部 -> 自动收窄并给出警告
    """
    if duration is None or duration <= 0:
        raise ValueError(f"AUDIO_DURATION 必须为正数，当前为 {duration}")
    total = total_duration if total_duration is not None else probe_audio_duration(audio_path)

    if start is None or start < 0:
        raise ValueError(f"AUDIO_START 不能为负数，当前为 {start}")
    if start >= total:
        raise ValueError(
            f"AUDIO_START={start:.3f}s 超出音频总长 {total:.3f}s（{os.path.basename(audio_path)}）"
        )

    end = start + duration
    if end > total:
        print(f"  警告: 请求区间 {start:.2f}s~{end:.2f}s 超出音频尾部，已收窄到 {total:.2f}s")
        end = total
    return start, end


def build_half_beats(beat_times):
    """在相邻整拍中点插入半拍。空/单拍时安全处理（修复原 IndexError）。"""
    beat_times = np.asarray(beat_times, dtype=float).flatten()
    if beat_times.size == 0:
        return np.array([], dtype=float)
    if beat_times.size == 1:
        return beat_times.copy()
    half = []
    for i in range(beat_times.size - 1):
        half.append(beat_times[i])
        half.append((beat_times[i] + beat_times[i + 1]) / 2.0)
    half.append(beat_times[-1])
    return np.asarray(half, dtype=float)


def detect_beats(audio_path, start_time=0, duration=None, fallback_bpm=SYNTH_FALLBACK_BPM):
    """检测拍点，返回 BeatGrid。

    拍点不足时使用 fallback_bpm 合成均匀网格，并置状态，绝不抛 IndexError。
    """
    y, sr = librosa.load(audio_path, offset=start_time, duration=duration, mono=True)
    win = duration if duration is not None else (len(y) / sr if sr else 0.0)

    if y.size == 0:
        beats = np.arange(0.0, max(win, 0.0), 60.0 / fallback_bpm)
        grid = BeatGrid(beats, build_half_beats(beats), fallback_bpm, "empty_audio")
        print(f"  警告: 音频为空，使用 {fallback_bpm:.0f}BPM 合成网格（{beats.size} 个拍点）")
        return grid

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = np.atleast_1d(
        librosa.frames_to_time(beat_frames, sr=sr)
    ).astype(float).flatten()

    status = "ok"
    if beat_times.size < 2 or not np.isfinite(tempo) or tempo <= 1.0:
        status = "few_beats_synthesized"
        tempo = fallback_bpm
        beat_times = np.arange(0.0, max(win, 0.0), 60.0 / tempo)
        print(f"  警告: 未检测到可靠拍点，回退到 {tempo:.0f}BPM 合成网格（{beat_times.size} 个拍点）")
    else:
        print(f"  BPM: {tempo:.1f}, 整拍: {beat_times.size} 个")

    return BeatGrid(beat_times, build_half_beats(beat_times), tempo, status)


def select_audio_segment(audio_path, duration=10.0, fallback_bpm=SYNTH_FALLBACK_BPM, verbose=True):
    """自动挑选一段最适合卡点的音乐区间（P1 第一项）。

    评分维度（全部本地计算，不接云端）：
      stability  节拍规律性：窗口内拍点间隔越稳越高
      contrast   能量变化：0.5s 子窗 RMS 的起伏，越大越有铺垫/高潮感
      density    拍点密度：窗口内拍点数量 / 理论上限
      energy     能量水平：避免选到静音段

    返回 {"start", "duration", "score", "reasons", "candidates"}
    """
    y, sr = librosa.load(audio_path, mono=True)
    if y.size == 0:
        raise ValueError(f"音频为空，无法自动选段: {audio_path}")
    total = len(y) / sr
    if duration >= total:
        return {
            "start": 0.0, "duration": float(total), "score": 1.0,
            "reasons": [f"音频总长 {total:.2f}s 不足请求时长 {duration:.2f}s，使用整段"],
            "candidates": [],
        }

    hop = 512
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(tempo) or tempo <= 1.0:
        tempo = fallback_bpm
    beat_t = np.atleast_1d(
        librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    ).astype(float).flatten()

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    global_max = float(rms.max()) + 1e-9

    candidates = beat_t[beat_t <= total - duration]
    if candidates.size == 0:
        candidates = np.linspace(0.0, max(0.0, total - duration), 20)

    scored = []
    for s in candidates:
        s = float(s)
        e = s + duration
        mask = (rms_t >= s) & (rms_t < e)
        seg_rms = rms[mask]
        if seg_rms.size == 0:
            continue
        energy_level = float(seg_rms.mean() / global_max)

        win = max(1, int(0.5 * sr / hop))
        nsub = seg_rms.size // win
        if nsub >= 2:
            sub = seg_rms[:nsub * win].reshape(nsub, win).mean(axis=1)
            contrast = float((sub.max() - sub.min()) / (sub.max() + 1e-9))
        else:
            contrast = 0.0

        b_in = beat_t[(beat_t >= s) & (beat_t < e)]
        if b_in.size >= 3:
            iv = np.diff(b_in)
            cv = float(iv.std() / (iv.mean() + 1e-9))
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.0
        density = min(b_in.size / max(1.0, duration * tempo / 60.0), 1.0)

        score = 0.35 * stability + 0.30 * contrast + 0.20 * density + 0.15 * energy_level
        scored.append({
            "start": s, "score": float(score),
            "stability": stability, "contrast": contrast,
            "density": float(density), "energy_level": energy_level,
            "beats": int(b_in.size),
        })

    if not scored:
        return {
            "start": 0.0, "duration": float(duration), "score": 0.0,
            "reasons": ["没有有效候选区间，回退到 0s 起点"], "candidates": [],
        }

    scored.sort(key=lambda d: d["score"], reverse=True)
    best = scored[0]
    reasons = [
        f"节拍规律性 stability={best['stability']:.2f}（越高越稳）",
        f"能量变化 contrast={best['contrast']:.2f}（越高越有起伏）",
        f"拍点密度 density={best['density']:.2f}，窗口内拍点 {best['beats']} 个",
        f"能量水平 energy={best['energy_level']:.2f}（避免静音段）",
        f"综合评分 score={best['score']:.3f}",
    ]
    if verbose:
        print(f"  自动选段推荐: {best['start']:.2f}s ~ {best['start'] + duration:.2f}s"
              f"（score={best['score']:.3f}）")
        for r in reasons:
            print(f"    - {r}")

    return {
        "start": best["start"], "duration": float(duration), "score": best["score"],
        "reasons": reasons, "candidates": scored[:3],
    }


# ==========================================================================
# 素材：扫描、自然排序、加载、轻量分析（P0-3 / P1 第三项）
# ==========================================================================
def natural_sort_key(name):
    """自然排序键：1.mp4 < 2.mp4 < 10.mp4。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_video_files(folder, clip_order=None, exts=VIDEO_EXTS):
    """收集素材文件，返回 (存在的文件路径列表, 缺失的文件名列表)。

    - clip_order 非空：按给定顺序取，缺失的名字记入 missing（指名报错用）
    - clip_order 为空或一个都没找到：自动扫描并按自然序排序
    """
    found, missing = [], []
    if clip_order:
        for name in clip_order:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                found.append(path)
            else:
                missing.append(name)

    if not found:
        if clip_order:
            print("  提示: CLIP_ORDER 中的素材均未找到，改用自动扫描")
        scanned = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        scanned.sort(key=natural_sort_key)
        found = [os.path.join(folder, f) for f in scanned]
        missing = [] if scanned else missing
    return found, missing


def prepare_video_clips(video_files, target_resolution=None):
    """加载并统一尺寸，返回 (clips, names, errors)。

    names 与 clips 一一对应，errors 为 [(文件名, 错误信息)]，供调用方指名报错。
    """
    clips, names, errors = [], [], []
    for i, vf in enumerate(video_files):
        name = os.path.basename(vf)
        try:
            clip = VideoFileClip(vf)
        except Exception as exc:  # noqa: BLE001
            errors.append((name, str(exc)))
            print(f"  错误: {name} 无法读取 - {exc}")
            continue
        if target_resolution:
            tw, th = target_resolution
            cw, ch = clip.size
            scale = max(tw / cw, th / ch) * 1.2
            clip = clip.resized(scale)
            new_w, new_h = clip.size
            xc, yc = new_w / 2, new_h / 2
            clip = clip.cropped(
                x1=int(xc - tw / 2), y1=int(yc - th / 2),
                x2=int(xc + tw / 2), y2=int(yc + th / 2),
            )
        clip = clip.without_audio()
        clips.append(clip)
        names.append(name)
        print(f"  [{i + 1}] {name} - {clip.duration:.2f}s")
    return clips, names, errors


_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _to_gray(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3:
        return (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1]
                + 0.114 * frame[:, :, 2]).astype(np.float32)
    return frame.astype(np.float32)


def _laplacian_variance(gray):
    """清晰度指标：3x3 拉普拉斯响应的方差，越大越锐利。"""
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    acc = np.zeros((h - 2, w - 2), dtype=np.float32)
    k = _LAPLACIAN_KERNEL
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx]:
                acc += k[dy, dx] * gray[dy:h - 2 + dy, dx:w - 2 + dx]
    return float(acc.var())


def _exposure_score(gray):
    """曝光健康度：过暗/过亮/大量死黑死白都会扣分，返回 0~1。"""
    mean = float(gray.mean())
    dark = max(0.0, (60.0 - mean) / 60.0)
    bright = max(0.0, (mean - 200.0) / 55.0)
    clipped = float(np.mean((gray <= 2.0) | (gray >= 253.0)))
    return float(np.clip(1.0 - max(dark, bright) - clipped, 0.0, 1.0))


def _sample_gray_frames(clip, n=QUALITY_SAMPLE_FRAMES, step=4):
    """均匀采样 n 帧并转灰度（按 step 降采样，控制开销）。"""
    dur = float(clip.duration or 0.0)
    if dur <= 0.0 or n <= 0:
        return []
    times = np.linspace(0.0, max(0.0, dur - 1e-3), n)
    out = []
    for t in times:
        gray = _to_gray(clip.get_frame(float(t)))
        out.append(gray[::step, ::step])
    return out


def _minmax(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros(0)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return np.full(v.size, 0.5)
    return (v - lo) / (hi - lo)


def _score_asset_quality(infos):
    """在本次素材集合内归一化，得到 sharpness / motion / quality_score。"""
    sharp = _minmax([i.sharpness_raw for i in infos])
    motion = _minmax([i.motion_raw for i in infos])
    for i, (s, m) in enumerate(zip(sharp, motion)):
        info = infos[i]
        info.sharpness = float(s)
        info.motion = float(m)
        # 清晰度 50% / 运动量 30% / 曝光 20%
        info.quality_score = float(np.clip(
            0.5 * info.sharpness + 0.3 * info.motion + 0.2 * info.exposure_score, 0.0, 1.0
        ))


def analyze_assets(video_files, target_duration=0.0, sample_frames=QUALITY_SAMPLE_FRAMES):
    """轻量素材分析：时长/分辨率/清晰度/运动量/曝光 -> 质量分与候选评分（P2 第五项）。

    仅使用本地可达指标，不联网；quality_score 是集合内相对值，仅为「候选评分」。
    """
    infos = []
    for vf in video_files:
        name = os.path.basename(vf)
        try:
            clip = VideoFileClip(vf)
        except Exception as exc:  # noqa: BLE001
            infos.append(AssetInfo(name=name, duration=0.0, width=0, height=0,
                                   aspect=0.0, has_audio=False, note=f"无法读取: {exc}"))
            continue

        w, h = clip.size
        info = AssetInfo(
            name=name,
            duration=float(clip.duration or 0.0),
            width=int(w),
            height=int(h),
            aspect=(w / h if h else 0.0),
            has_audio=clip.audio is not None,
        )
        info.estimated_usable = info.duration
        info.usable_ratio = (min(info.duration / target_duration, 1.0)
                             if target_duration > 0 else 1.0)
        try:
            grays = _sample_gray_frames(clip, sample_frames)
            if grays:
                info.sharpness_raw = float(np.mean([_laplacian_variance(g) for g in grays]))
                info.exposure_score = float(np.mean([_exposure_score(g) for g in grays]))
                if len(grays) >= 2:
                    info.motion_raw = float(np.mean([
                        np.mean(np.abs(grays[i + 1] - grays[i])) / 255.0
                        for i in range(len(grays) - 1)
                    ]))
        except Exception as exc:  # noqa: BLE001
            info.note = f"帧采样失败: {exc}"
        clip.close()
        infos.append(info)

    _score_asset_quality(infos)
    for info in infos:
        info.candidate_score = float(np.clip(
            0.5 * info.quality_score + 0.5 * info.usable_ratio, 0.0, 1.0
        ))
    return infos


def quality_weights(infos, names=None, prefer_quality=PREFER_QUALITY):
    """由素材质量分生成分配权重（把更多镜头时间给更优质的素材）。

    names 用于把权重对齐到实际加载成功的素材顺序；缺失的按 1.0 兜底。
    """
    if not prefer_quality or not infos:
        return None
    if names is None:
        return [max(1e-6, float(i.candidate_score)) for i in infos]
    by_name = {i.name: i for i in infos}
    return [max(1e-6, float(by_name[n].candidate_score)) if n in by_name else 1.0
            for n in names]


# ==========================================================================
# 时间轴：网格吸附、切点、分段计划
# ==========================================================================
def snap_to_grid(time, grid_times, min_gap=0.0, fallback=None):
    """把时间吸附到最近的网格点。

    - 只在 >= min_gap 的候选点里选最近（修复原版会返回更早点的问题）
    - 候选点为空时返回 fallback（默认原时间）
    """
    grid = np.asarray(grid_times, dtype=float).flatten()
    if grid.size == 0:
        return float(time if fallback is None else fallback)
    candidates = grid[grid >= min_gap - 1e-9]
    if candidates.size == 0:
        candidates = grid
    idx = int(np.argmin(np.abs(candidates - time)))
    return float(candidates[idx])


# 剪辑策略档位：每项为 (拍数, 标签, 吸附类型)
PATTERN_PRESETS = {
    "normal": [
        (0.5, "开头闪", "half"), (4.0, "长", "full"), (0.5, "短闪", "half"),
        (1.5, "中", "full"), (6.0, "铺垫最长", "full"), (0.5, "快切", "half"),
        (0.33, "快切", "half"), (0.5, "快切", "half"), (0.33, "快切", "half"),
        (0.66, "快切", "half"), (0.5, "快切", "half"), (0.5, "快切", "half"),
        (0.5, "快切", "half"), (0.66, "快切", "half"), (1.5, "快切小停顿", "full"),
        (0.33, "快切", "half"), (5.0, "结尾长", "full"), (0.5, "结尾短闪", "half"),
        (3.5, "结尾中长", "full"), (0.5, "收尾", "half"),
    ],
    "slow": [
        (1.0, "开头渐入", "half"), (4.0, "长", "full"), (1.0, "短闪", "half"),
        (2.0, "中", "full"), (6.0, "铺垫最长", "full"), (1.0, "中切", "half"),
        (0.66, "中切", "half"), (1.0, "中切", "half"), (0.66, "中切", "half"),
        (1.0, "中切", "half"), (1.5, "中停顿", "full"), (2.0, "中长", "full"),
        (5.0, "结尾长", "full"), (1.0, "结尾短闪", "half"), (3.5, "结尾中长", "full"),
        (1.0, "收尾", "half"),
    ],
    "fast": [
        (0.5, "开头闪", "half"), (2.0, "长", "full"), (0.33, "快切", "half"),
        (0.33, "快切", "half"), (0.5, "快切", "half"), (0.33, "快切", "half"),
        (0.33, "快切", "half"), (0.5, "快切", "half"), (0.33, "快切", "half"),
        (0.5, "快切", "half"), (0.33, "快切", "half"), (0.33, "快切", "half"),
        (0.5, "快切", "half"), (0.5, "快切", "half"), (0.33, "快切", "half"),
        (0.66, "快切小停顿", "half"), (1.5, "结尾长", "full"), (0.33, "结尾短闪", "half"),
        (2.0, "结尾中长", "full"), (0.5, "收尾", "half"),
    ],
}


def build_boundaries(beat_grid, target_dur, pattern, min_shot=MIN_SHOT, verbose=True):
    """根据拍点网格生成切点（时间轴时间，从 0 开始）。"""
    beat_times, half_beats = beat_grid.beats, beat_grid.half_beats
    beat_dur = beat_grid.beat_duration
    half_beat_dur = beat_dur / 2.0

    total_pattern_beats = sum(p[0] for p in pattern) or 1.0
    total_beats = target_dur / beat_dur
    scale = total_beats / total_pattern_beats

    boundaries = [0.0]
    cum_beats = 0.0
    for beats, _label, snap_type in pattern:
        cum_beats += beats * scale
        target_time = cum_beats * beat_dur
        if target_time > target_dur + 0.2:
            break
        grid = half_beats if snap_type == "half" else beat_times
        base_gap = half_beat_dur if snap_type == "half" else beat_dur
        snapped = snap_to_grid(target_time, grid,
                               boundaries[-1] + base_gap * 0.5, fallback=target_time)
        if snapped - boundaries[-1] < min_shot:
            continue
        if snapped > target_dur + 0.1:
            break
        boundaries.append(snapped)

    if boundaries[-1] < target_dur - 0.3:
        end_snapped = snap_to_grid(target_dur, beat_times,
                                   boundaries[-1] + beat_dur * 0.5, fallback=target_dur)
        boundaries.append(end_snapped if end_snapped <= target_dur + 0.2 else target_dur)

    if verbose:
        print(f"  切点 {len(boundaries)} 个 / 片段 {len(boundaries) - 1} 段")
    return boundaries


def assign_clips(n_segs, n_clips, weights=None):
    """把片段分配给素材：按权重比例分配，且尽量不连续重复同一素材。

    weights=None 时等权（等价于均衡轮转）。
    """
    if n_clips <= 0:
        raise ValueError("素材数量为 0，无法分配片段")
    if n_clips == 1:
        return [0] * n_segs
    if weights is None:
        weights = [1.0] * n_clips
    if len(weights) != n_clips:
        raise ValueError(f"weights 长度 {len(weights)} 与素材数 {n_clips} 不一致")
    weights = [max(1e-6, float(w)) for w in weights]
    total_w = sum(weights)
    share = [w / total_w for w in weights]

    seq, counts, last = [], [0] * n_clips, -1
    for i in range(n_segs):
        deficits = [share[c] * (i + 1) - counts[c] for c in range(n_clips)]
        order = sorted(range(n_clips), key=lambda c: (-deficits[c], c))
        pick = order[0]
        if pick == last:
            pick = order[1]
        seq.append(pick)
        counts[pick] += 1
        last = pick
    return seq


def _wrap_offset(clip):
    """回绕复用时的起点（跳过素材开头一点点）。"""
    return min(clip.duration * 0.02, max(0.0, clip.duration - 1e-3))


def plan_segments(boundaries, clip_assign, video_clips,
                  beat_dur, clip_names=None, min_shot=MIN_SHOT,
                  min_speed=MIN_SPEED, max_speed=MAX_SPEED,
                  allow_reuse=None, transition=None):
    """生成分段计划：每段的源入点/出点/倍速/时间轴区间。

    allow_reuse=None 时由 SHORT_SEGMENT_STRATEGY 决定（loop=复用 / error=报错）。
    """
    if allow_reuse is None:
        if SHORT_SEGMENT_STRATEGY not in ("loop", "error"):
            raise ValueError(
                f"SHORT_SEGMENT_STRATEGY 只能是 loop 或 error，当前为 {SHORT_SEGMENT_STRATEGY!r}"
            )
        allow_reuse = (SHORT_SEGMENT_STRATEGY != "error")
    if transition is None:
        transition = TRANSITION_MODE
    n_segs = len(boundaries) - 1
    if clip_names is None:
        clip_names = [f"clip{i}" for i in range(len(video_clips))]
    positions = [_wrap_offset(c) for c in video_clips]
    segments = []
    timeline_cursor = 0.0
    issues = []

    for i in range(n_segs):
        target = boundaries[i + 1] - boundaries[i]
        if target < min_shot - 1e-6:
            issues.append(f"片段{i} 目标时长 {target:.3f}s 低于最短镜头 {min_shot}s，已跳过")
            continue

        ci = clip_assign[i]
        clip = video_clips[ci]
        pos = positions[ci]
        reused = False

        if pos >= clip.duration - 1e-3:
            if not allow_reuse:
                raise RuntimeError(
                    f"素材 {clip_names[ci]} 已用尽，且 SHORT_SEGMENT_STRATEGY='error'"
                )
            pos = _wrap_offset(clip)
            reused = True

        avail = clip.duration - pos
        desired = min(AVG_SPEED_BASE + (i % 5) * AVG_SPEED_STEP, max_speed)

        if avail < target * min_speed and allow_reuse and not reused:
            pos = _wrap_offset(clip)
            reused = True
            avail = clip.duration - pos

        source_dur, speed, note = resolve_segment_timing(
            avail, target, desired, min_speed, max_speed
        )
        if note:
            issues.append(
                f"片段{i}: 素材可用 {avail:.3f}s 不足以填满 {target:.3f}s（最慢 {min_speed}x）"
            )

        out_dur = timeline_duration_for(source_dur, speed)
        is_fast = (target / beat_dur) < 0.8
        seg = EditSegment(
            index=i,
            clip_index=ci,
            clip_name=clip_names[ci],
            label="",
            timeline_start=timeline_cursor,
            timeline_end=timeline_cursor + out_dur,
            duration=out_dur,
            source_in=pos,
            source_out=pos + source_dur,
            speed=speed,
            is_fast=is_fast,
            reused=reused,
            transition=("cut" if is_fast or transition == "cut" else transition),
            note=note,
        )
        positions[ci] = pos + source_dur
        timeline_cursor = seg.timeline_end
        segments.append(seg)

    return segments, issues


def apply_labels(segments, pattern):
    for seg in segments:
        if seg.index < len(pattern):
            seg.label = pattern[seg.index][1]
        else:
            seg.label = "?"
    return segments


# ==========================================================================
# 特效（原算法保持不变，仅把硬编码 BPM 改为真实 beat_dur）
# ==========================================================================
def fisheye_geom(h, w):
    """鱼眼重映射中与强度无关的几何项，可缓存复用。

    返回 (xs, ys, rn, rbase)：
      xs 形状 (w,)、ys 形状 (h,) 为像素相对中心的偏移（可广播，省内存）
      rn / rbase 形状 (h, w)，使得 scale = rn + strength * rbase
      （由 rr*(1+s*rr²)/rr_safe 展开而来，逐帧只剩 1 次乘法 + 1 次加法）
    """
    xs = np.arange(w, dtype=np.float32) - w / 2.0
    ys = np.arange(h, dtype=np.float32) - h / 2.0
    rr = np.sqrt((xs[None, :] / (w / 2.0)) ** 2 + (ys[:, None] / (h / 2.0)) ** 2)
    rr = np.clip(rr, 0.0, 1.0)
    rn = rr / np.maximum(rr, 0.001)
    return xs, ys, rn, rn * rr * rr


def fisheye_apply(img, strength, geom):
    """用给定几何项做鱼眼重映射（批量渲染时复用缓存的 geom），输出尺寸不变。"""
    h, w = img.shape[:2]
    xs, ys, rn, rbase = geom
    scale = rn + strength * rbase
    src_x = np.clip(w / 2.0 + xs[None, :] * scale, 0, w - 1).astype(np.int32)
    src_y = np.clip(h / 2.0 + ys[:, None] * scale, 0, h - 1).astype(np.int32)
    return img[src_y, src_x]


def fisheye(img, strength):
    """鱼眼畸变，输出尺寸不变"""
    h, w = img.shape[:2]
    return fisheye_apply(img, strength, fisheye_geom(h, w))


def pixelate(img, block_size):
    """像素化：每个像素取所属方块左上角的颜色，输出尺寸与输入完全一致。

    修复：旧实现用 repeat+截断，当 h/w 不能被 block_size 整除时输出会变小
    （如 960x720、block=13 -> 949x715），导致该帧右下出现黑边。改用取整除
    得到方块索引，保证输出尺寸不变。
    """
    h, w = img.shape[:2]
    if block_size < 2:
        return img
    sh = max(1, h // block_size)
    sw = max(1, w // block_size)
    yi = (np.arange(sh) * block_size).astype(int)
    xi = (np.arange(sw) * block_size).astype(int)
    small = img[np.ix_(yi, xi)]
    yr = np.minimum(np.arange(h) // block_size, sh - 1)
    xr = np.minimum(np.arange(w) // block_size, sw - 1)
    return small[yr[:, None], xr[None, :]]


FLARE_CH = np.array([1.0, 0.85, 0.6], dtype=np.float32)   # 光晕的通道权重


def flare_kernel(h, w):
    """光晕中与强度无关的高斯核，可缓存复用。返回 (e1, e2)，形状均为 (h, w)。"""
    cx, cy = w * 0.5, h * 0.35
    y_arr, x_arr = np.ogrid[:h, :w]
    dist = np.sqrt((x_arr - cx) ** 2 * 1.5 + (y_arr - cy) ** 2)
    e1 = np.exp(-dist ** 2 / (2 * 100 ** 2)).astype(np.float32)
    e2 = np.exp(-((x_arr - cx * 0.7) ** 2 + (y_arr - cy * 1.3) ** 2) / (2 * 45 ** 2)).astype(np.float32)
    return e1, e2


def flare_base(h, w, kernel=None):
    """光晕「单位强度」的逐像素加性基 (h, w, 3)。

    (e1*I + e2*I*0.4)*CH*255 可展开为 (e1 + 0.4*e2)*CH*255 * I，
    故只需按尺寸预计算一次基，逐帧仅剩一次标量乘法。
    """
    e1, e2 = flare_kernel(h, w) if kernel is None else kernel
    return (e1 + 0.4 * e2)[:, :, None] * FLARE_CH * 255.0


def flare_offsets(h, w, intensity, kernel=None):
    """光晕的逐像素加性偏移 (h, w, 3)。kernel 为 flare_kernel 的缓存结果。"""
    return flare_base(h, w, kernel) * intensity


def add_flare(img, intensity):
    h, w = img.shape[:2]
    off = flare_offsets(h, w, intensity)
    return np.clip(img.astype(np.float32) + off, 0, 255).astype(np.uint8)


def add_effects(seg, seg_start_time, beat_times, seg_index, is_fast, beat_dur):
    """逐帧特效。beat_dur 由真实 tempo 求得（P0-6），不再硬编码 172.3BPM。

    性能（720x960 单帧实测约 238ms -> 约 140ms）：
    - 与强度无关的几何量按尺寸预计算并缓存：暗角遮罩、鱼眼几何、光晕高斯核
    - 胶片颗粒改为「按段预生成基准噪声 + 逐帧随机窗口切片」，避免每帧生成
      2.07M 个正态随机数（约 35ms/帧 -> 约 2ms/帧）
    - uint8 转换改为惰性（仅在鱼眼/像素化真正触发时计算，省约 3ms/帧）
    - 运动模糊与暗角乘算改为原地运算，减少大数组分配
    """
    seg_dur = seg.duration
    beat_dur_local = beat_dur
    rng = np.random.RandomState(777 + seg_index * 19)
    tw, th = TARGET_RESOLUTION
    vignette_cache = {}
    grain_cache = {}
    fisheye_cache = {}
    flare_cache = {}
    grain_pad = 24

    def get_vignette(h, w):
        mask = vignette_cache.get((h, w))
        if mask is None:
            yv, xv = np.ogrid[:h, :w]
            dv = np.sqrt((xv - w / 2) ** 2 + (yv - h / 2) ** 2)
            mv = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
            mask = (1.0 - (dv / mv) ** 2 * FILTER_VIGNETTE).astype(np.float32)
            vignette_cache[(h, w)] = mask
        return mask

    def get_grain(h, w):
        """按段预生成的胶片颗粒基准（比目标尺寸大 grain_pad，逐帧随机窗口切片）。"""
        tile = grain_cache.get((h, w))
        if tile is None:
            tile = rng.normal(0, FILTER_FILM_GRAIN * 255,
                              (h + grain_pad, w + grain_pad, 3)).astype(np.float32)
            grain_cache[(h, w)] = tile
        return tile

    def get_fisheye_geom(h, w):
        geom = fisheye_cache.get((h, w))
        if geom is None:
            geom = fisheye_geom(h, w)
            fisheye_cache[(h, w)] = geom
        return geom

    def get_flare_base(h, w):
        base = flare_cache.get((h, w))
        if base is None:
            base = flare_base(h, w).astype(np.float32)
            flare_cache[(h, w)] = base
        return base

    def to_uint8(f):
        return np.clip(f, 0, 255).astype(np.uint8)

    def process_frame(get_frame, t):
        frame = get_frame(t)
        f = frame.astype(np.float32)
        global_t = seg_start_time + t

        # 节拍信息
        nearest_beat_dist = np.min(np.abs(beat_times - global_t))
        is_near_beat = nearest_beat_dist < beat_dur_local * 0.3
        pulse_phase = (nearest_beat_dist / beat_dur_local) * np.pi
        pulse_amount = np.cos(pulse_phase) * 0.5 + 0.5

        fh, fw = f.shape[:2]

        # ===== 1. 缩放脉冲 + 推镜 + 震动 =====
        pulse_zoom = 1.0 + pulse_amount * PULSE_INTENSITY
        if not is_fast and seg_dur > 0.5:
            push_zoom = 1.0 + (t / seg_dur) * ZOOM_SPEED * min(seg_dur, 3.0)
        else:
            push_zoom = 1.0
        total_zoom = pulse_zoom * push_zoom

        shake_x = shake_y = 0.0
        if is_fast and is_near_beat:
            strength = SHAKE_INTENSITY * (1 - nearest_beat_dist / (beat_dur_local * 0.3))
            angle = rng.uniform(0, 2 * np.pi)
            shake_x = np.cos(angle) * strength
            shake_y = np.sin(angle) * strength

        if abs(total_zoom - 1.0) > 0.001 or abs(shake_x) > 0.1 or abs(shake_y) > 0.1:
            crop_w = int(fw / total_zoom)
            crop_h = int(fh / total_zoom)
            cx_pix = fw / 2 + shake_x * total_zoom
            cy_pix = fh / 2 + shake_y * total_zoom
            x1 = int(max(0, min(cx_pix - crop_w / 2, fw - crop_w)))
            y1 = int(max(0, min(cy_pix - crop_h / 2, fh - crop_h)))
            x2, y2 = x1 + crop_w, y1 + crop_h
            cropped = f[y1:y2, x1:x2, :]
            rows = np.linspace(0, crop_h - 1, fh).astype(int)
            cols = np.linspace(0, crop_w - 1, fw).astype(int)
            f = cropped[rows[:, np.newaxis], cols[np.newaxis, :], :]

        # ===== 2. 鱼眼脉冲（复用缓存的几何项；uint8 惰性转换）=====
        if FISHEYE_ON and pulse_amount > 0.3:
            fe_str = pulse_amount * 0.12
            f = fisheye_apply(to_uint8(f), fe_str, get_fisheye_geom(fh, fw)).astype(np.float32)

        # ===== 3. 像素化故障 =====
        if PIXEL_ON and is_fast and rng.random() < 0.15:
            block = rng.randint(6, 15)
            f = pixelate(to_uint8(f), block).astype(np.float32)

        # ===== 4. Glitch色差 =====
        if is_fast and is_near_beat:
            ga = GLITCH_INTENSITY * (1 - nearest_beat_dist / (beat_dur_local * 0.3))
            if ga > 0.5:
                off = int(ga)
                f[:, :, 0] = np.roll(f[:, :, 0], -off, axis=1)
                f[:, :, 2] = np.roll(f[:, :, 2], off, axis=1)
                go = rng.randint(-max(1, off // 2), max(2, off // 2 + 1))
                f[:, :, 1] = np.roll(f[:, :, 1], go, axis=0)

        # ===== 5. 色相偏移 =====
        if is_fast:
            hs = HUE_SHIFT_AMOUNT * pulse_amount * (1 if seg_index % 2 == 0 else -1)
            if abs(hs) > 1.0:
                sr = np.radians(hs)
                cr = np.cos(sr)
                sh = np.sin(sr)
                r, g, b = f[:, :, 0].copy(), f[:, :, 1].copy(), f[:, :, 2].copy()
                f[:, :, 0] = r * (0.8 + 0.2 * cr) + g * 0.1 * sh + b * 0.1 * -sh
                f[:, :, 1] = r * 0.1 * -sh + g * (0.8 + 0.2 * cr) + b * 0.1 * sh
                f[:, :, 2] = r * 0.1 * sh + g * 0.1 * -sh + b * (0.8 + 0.2 * cr)

        # ===== 6. 运动模糊（原地增量，等价于 blurred = blurred*(1-k) + shifted*k）=====
        if is_fast and t < seg_dur * 0.4:
            mb = MOTION_BLUR_INTENSITY * (1 - t / (seg_dur * 0.4))
            bs = int(mb * 10)
            if bs > 1:
                blurred = f.copy()
                for i in range(1, bs + 1):
                    w_i = 1.0 / (i + 1)
                    shifted = np.roll(f, i, axis=1)
                    blurred += (shifted - blurred) * (w_i * 0.3)
                f = blurred

        # ===== 7. 调色 =====
        f = f * FILTER_BRIGHTNESS
        f = 128.0 + (f - 128.0) * FILTER_CONTRAST

        if FILTER_TEAL_ORANGE:
            lum = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
            hl = np.clip((lum - 100) / 100, 0, 1)
            sh_mask = 1.0 - hl
            f[:, :, 0] += hl * 20
            f[:, :, 1] += hl * 10
            f[:, :, 2] -= hl * 14
            f[:, :, 0] -= sh_mask * 9.6
            f[:, :, 1] += sh_mask * 6.4
            f[:, :, 2] += sh_mask * 16

        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        for c in range(3):
            f[:, :, c] = gray + (f[:, :, c] - gray) * FILTER_SATURATION

        # ===== 8. 镜头光晕（复用缓存的单位强度基，强度仅做标量乘）=====
        if FLARE_ON and pulse_amount > 0.5:
            ch, cw = f.shape[:2]
            np.add(f, get_flare_base(ch, cw) * (pulse_amount * 0.6), out=f)
            np.clip(f, 0, 255, out=f)

        # ===== 9. 胶片颗粒（预生成基准 + 逐帧随机窗口切片，原地叠加）=====
        if FILTER_FILM_GRAIN > 0:
            gh, gw = f.shape[:2]
            tile = get_grain(gh, gw)
            oy = rng.randint(0, grain_pad + 1)
            ox = rng.randint(0, grain_pad + 1)
            f += tile[oy:oy + gh, ox:ox + gw, :]

        # ===== 10. 撕裂转场 =====
        td = 0.05
        fh2, fw2 = f.shape[:2]
        if t < td and is_fast:
            ta = int(15 * (1 - t / td))
            mid = fh2 // 2
            f[:mid, :, :] = np.roll(f[:mid, :, :], ta, axis=1)
            f[mid:, :, :] = np.roll(f[mid:, :, :], -ta, axis=1)
        if t > seg_dur - td and is_fast:
            ta = int(15 * (1 - (seg_dur - t) / td))
            mid = fh2 // 2
            f[:mid, :, :] = np.roll(f[:mid, :, :], -ta, axis=1)
            f[mid:, :, :] = np.roll(f[mid:, :, :], ta, axis=1)

        # 暗角（遮罩已按尺寸预计算并缓存，原地乘算）
        vign = get_vignette(fh2, fw2)
        f *= vign[:, :, np.newaxis]

        np.clip(f, 0, 255, out=f)

        # 确保输出尺寸精确
        result = f.astype(np.uint8)
        if result.shape[0] != th or result.shape[1] != tw:
            new_result = np.zeros((th, tw, 3), dtype=np.uint8)
            sh2 = min(result.shape[0], th)
            sw2 = min(result.shape[1], tw)
            new_result[:sh2, :sw2, :] = result[:sh2, :sw2, :]
            result = new_result

        return result

    return seg.transform(process_frame, apply_to=["video"])


# ==========================================================================
# 渲染
# ==========================================================================
def render_segments(segments, video_clips, beat_grid):
    """按计划执行：切源 -> 变速 -> 加特效 -> 拼接。"""
    beat_dur = beat_grid.beat_duration
    renders = []
    for seg in segments:
        clip = video_clips[seg.clip_index]
        sub = clip.subclipped(seg.source_in, seg.source_out)
        # P0-2 修复：with_speed_scaled(speed) 使结果时长 = 源时长 / speed
        sped = sub.with_speed_scaled(seg.speed)
        fx = add_effects(sped, seg.timeline_start, beat_grid.beats,
                         seg.index, seg.is_fast, beat_dur)
        # 转场：dip 只作用于非快切片段，淡入淡出黑场不改变片段时长（时序安全）
        if seg.transition == "dip":
            fade = min(TRANSITION_DUR, max(0.0, seg.duration / 3.0))
            if fade > 0:
                fx = fx.with_effects([vfx.FadeIn(fade), vfx.FadeOut(fade)])
        renders.append(fx)
        mark = "⚡" if seg.is_fast else "  "
        reuse = " [复用]" if seg.reused else ""
        trans = "" if seg.transition == "cut" else f" <{seg.transition}>"
        print(f"    {mark} 片段{seg.index + 1:2d} "
              f"[{seg.clip_name}|{seg.label:8s}] "
              f"源 {seg.source_in:.2f}~{seg.source_out:.2f}s "
              f"-> {seg.duration:.3f}s @ {seg.speed:.2f}x{reuse}{trans}")
    final = concatenate_videoclips(renders, method="compose")
    print(f"\n  拼接后: {final.duration:.3f}s")
    return final


# ==========================================================================
# 剪辑计划（EDL）与执行前校验（P2 第六项）
# ==========================================================================
def build_edl(segments, audio_info, clip_durations,
              target_resolution=TARGET_RESOLUTION, target_duration=AUDIO_DURATION):
    return {
        "version": 1,
        "created_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_resolution": list(target_resolution),
        "target_duration": float(target_duration),
        "audio": audio_info,
        "clip_durations": clip_durations,
        "segments": [asdict(s) for s in segments],
    }


def validate_edl(edl, min_speed=MIN_SPEED, max_speed=MAX_SPEED, tol=1e-3):
    """执行前校验剪辑计划，返回问题列表（critical 直接抛错，其余为警告）。"""
    problems, critical = [], []
    segs = edl.get("segments") or []
    if not segs:
        critical.append("剪辑计划为空")
        return problems, critical

    clip_durations = edl.get("clip_durations") or {}
    cursor = 0.0
    for s in segs:
        if abs(s["timeline_start"] - cursor) > tol:
            problems.append(
                f"片段{s['index']} 时间轴不连续: start={s['timeline_start']:.4f}s，应为 {cursor:.4f}s"
            )
        if abs((s["timeline_end"] - s["timeline_start"]) - s["duration"]) > tol:
            problems.append(f"片段{s['index']} duration 与区间不一致")
        if s["source_in"] < -tol:
            critical.append(f"片段{s['index']} 源入点为负: {s['source_in']:.4f}s")
        if s["source_out"] - s["source_in"] <= 0:
            critical.append(f"片段{s['index']} 源区间为空")
        dur = clip_durations.get(s["clip_name"])
        if dur is not None and s["source_out"] > dur + 0.05:
            critical.append(
                f"片段{s['index']} 源出点 {s['source_out']:.3f}s 超出素材 {s['clip_name']} 时长 {dur:.3f}s"
            )
        if not (min_speed - 1e-6 <= s["speed"] <= max_speed + 1e-6):
            problems.append(f"片段{s['index']} 倍速 {s['speed']:.3f} 超出 [{min_speed}, {max_speed}]")
        trans = s.get("transition", "cut")
        if trans not in ("cut", "dip"):
            problems.append(f"片段{s['index']} 未知转场类型: {trans}")
        cursor = s["timeline_end"]

    audio = edl.get("audio") or {}
    want = audio.get("duration")
    if want:
        drift = abs(cursor - float(want))
        if drift > 0.2:
            problems.append(f"成片总时长 {cursor:.3f}s 与音频区间 {want:.3f}s 偏差 {drift:.3f}s")
    return problems, critical


def save_edl(edl, path=EDL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(edl, fp, ensure_ascii=False, indent=2)
    return path


# ==========================================================================
# 主流程
# ==========================================================================
def build_edit_plan(folder=FOLDER, audio_file=AUDIO_FILE, duration=AUDIO_DURATION,
                    audio_start=AUDIO_START, auto_select=AUTO_SELECT_AUDIO,
                    clip_order=None, strategy=EDIT_STRATEGY,
                    target_resolution=TARGET_RESOLUTION, verbose=True):
    """构建剪辑计划（不渲染）。返回 (plan, video_clips, beat_grid, status)。

    plan 是可直接落盘的 dict（含 EDL + 素材分析 + 选段理由）。
    """
    clip_order = CLIP_ORDER if clip_order is None else clip_order
    status = {"ok": True, "messages": []}

    # ---- 素材 ----
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"素材文件夹不存在: {folder}")
    video_files, missing = collect_video_files(folder, clip_order)
    if missing:
        msg = f"CLIP_ORDER 中以下素材缺失: {', '.join(missing)}"
        print(f"  警告: {msg}")
        status["messages"].append(msg)
    if not video_files:
        raise FileNotFoundError(f"在 {folder} 中未找到任何视频素材（支持 {VIDEO_EXTS}）")
    if verbose:
        print(f"\n素材: {len(video_files)} 个 -> {[os.path.basename(v) for v in video_files]}")
    if len(video_files) < 4:
        hint = ("素材少于 4 个，可能不足以支撑密集快切。三种策略："
                "loop=回绕复用已用素材 / error=素材不足直接报错 / 增加素材")
        print(f"  提示: {hint}")
        status["messages"].append(hint)

    # ---- 音乐 ----
    audio_path = os.path.join(folder, audio_file)
    if not os.path.exists(audio_path):
        mp3s = sorted([f for f in os.listdir(folder) if f.lower().endswith(".mp3")],
                      key=natural_sort_key)
        if not mp3s:
            raise FileNotFoundError(f"未找到音频（{audio_file} 及任何 .mp3）于 {folder}")
        audio_path = os.path.join(folder, mp3s[0])
        print(f"  提示: 指定音频不存在，改用 {os.path.basename(audio_path)}")

    total_audio = probe_audio_duration(audio_path)
    selection = None
    if auto_select or audio_start is None:
        selection = select_audio_segment(audio_path, duration, verbose=verbose)
        start, _end = validate_audio_window(audio_path, selection["start"], duration, total_audio)
        status["messages"].extend(selection["reasons"])
    else:
        start, _end = validate_audio_window(audio_path, audio_start, duration, total_audio)

    if verbose:
        print(f"\n分析音乐（{start:.2f}s 起，取 {duration:.2f}s）...")
    beat_grid = detect_beats(audio_path, start, duration + 2)
    if beat_grid.status != "ok":
        status["messages"].append(f"拍点状态: {beat_grid.status}")

    # ---- 素材加载与分析 ----
    if verbose:
        print("\n加载素材...")
    video_clips, clip_names, errors = prepare_video_clips(video_files, target_resolution)
    if errors:
        names = ", ".join(n for n, _ in errors)
        status["messages"].append(f"以下素材损坏/无法读取: {names}")
    if not video_clips:
        raise RuntimeError(f"所有素材均无法读取: {', '.join(n for n, _ in errors)}")
    if errors:
        print(f"  警告: {len(errors)} 个素材被跳过: {', '.join(n for n, _ in errors)}")

    assets = analyze_assets(video_files, target_duration=duration)
    clip_durations = {name: clip.duration for name, clip in zip(clip_names, video_clips)}

    if verbose:
        print("\n素材质量评估（相对候选评分）:")
        for a in assets:
            flag = "  ⚠低分" if a.quality_score < QUALITY_MIN_WARN else ""
            print(f"  {a.name}: 质量 {a.quality_score:.2f} "
                  f"(清晰度 {a.sharpness:.2f} / 运动 {a.motion:.2f} / 曝光 {a.exposure_score:.2f}) "
                  f"候选 {a.candidate_score:.2f}{flag}")
    low = [a.name for a in assets if a.quality_score < QUALITY_MIN_WARN and a.width]
    if len(assets) > 1 and len(low) == len(assets):
        status["messages"].append("所有素材质量分都偏低（可能模糊/欠曝/过曝），建议更换素材")

    # ---- 切点与分段 ----
    pattern = PATTERN_PRESETS.get(strategy, PATTERN_PRESETS["normal"])
    boundaries = build_boundaries(beat_grid, duration, pattern, MIN_SHOT, verbose=verbose)
    n_segs = len(boundaries) - 1
    weights = quality_weights(assets, clip_names)
    clip_assign = assign_clips(n_segs, len(video_clips), weights)
    segments, issues = plan_segments(boundaries, clip_assign, video_clips,
                                     beat_grid.beat_duration, clip_names=clip_names,
                                     transition=TRANSITION_MODE)
    for msg in issues:
        print(f"  警告: {msg}")
    status["messages"].extend(issues)
    apply_labels(segments, pattern)

    audio_info = {
        "path": os.path.basename(audio_path),
        "start": float(start),
        "duration": float(duration),
        "tempo": float(beat_grid.tempo),
        "beat_status": beat_grid.status,
        "auto_selected": bool(selection is not None),
    }
    if selection:
        audio_info["selection_score"] = selection["score"]
        audio_info["selection_reasons"] = selection["reasons"]

    edl = build_edl(segments, audio_info, clip_durations, target_resolution, duration)
    problems, critical = validate_edl(edl)
    for p in problems:
        print(f"  校验警告: {p}")
        status["messages"].append(p)
    if critical:
        raise RuntimeError("剪辑计划校验失败: " + "; ".join(critical))

    plan = {
        "edl": edl,
        "assets": [asdict(a) for a in assets],
        "audio_selection": selection,
        "strategy": str(strategy),
        "segments": segments,
        "beat_grid": beat_grid,
        "audio_path": audio_path,
    }
    return plan, video_clips, beat_grid, status


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="游戏CG自动卡点剪辑（auto beat edit）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--plan-only", action="store_true",
                   help="只分析并输出剪辑计划(EDL)+素材质量报告，不渲染")
    p.add_argument("--folder", default=FOLDER, help="素材文件夹")
    p.add_argument("--audio", default=AUDIO_FILE, help="音乐文件名")
    p.add_argument("--duration", type=float, default=AUDIO_DURATION, help="输出时长（秒）")
    p.add_argument("--audio-start", type=float, default=AUDIO_START,
                   help="音乐起始秒数；设为负数表示启用自动选段")
    p.add_argument("--strategy", default=EDIT_STRATEGY,
                   choices=sorted(PATTERN_PRESETS), help="节奏模板")
    p.add_argument("--edl", default=EDL_PATH, help="剪辑计划输出路径")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 60)
    print("  游戏CG自动卡点剪辑（auto beat edit）")
    print("=" * 60)

    audio_start = None if args.audio_start is not None and args.audio_start < 0 else args.audio_start
    try:
        plan, video_clips, beat_grid, status = build_edit_plan(
            folder=args.folder, audio_file=args.audio, duration=args.duration,
            audio_start=audio_start, strategy=args.strategy,
        )
    except Exception as exc:  # noqa: BLE001 - 顶层给出明确失败原因
        print(f"\n[失败] {exc}")
        return 1

    edl = plan["edl"]
    if EXPORT_EDL:
        path = save_edl(edl, args.edl)
        print(f"\n剪辑计划已保存: {path}")

    if args.plan_only:
        print("\n[--plan-only] 已跳过渲染。片段概览:")
        for s in edl["segments"][:12]:
            print(f"    #{s['index']:2d} {s['clip_name']:>12s} | {s['label']:8s} | "
                  f"{s['timeline_start']:6.2f}~{s['timeline_end']:6.2f}s | "
                  f"{s['speed']:.2f}x | {s['transition']}")
        if len(edl["segments"]) > 12:
            print(f"    ... 共 {len(edl['segments'])} 段")
        print(f"  总时长 {edl['segments'][-1]['timeline_end']:.2f}s / "
              f"目标 {edl['target_duration']:.2f}s")
        for c in video_clips:
            c.close()
        return 0

    print("\n渲染中...")
    final = render_segments(plan["segments"], video_clips, beat_grid)
    target_dur = edl["target_duration"]
    if final.duration > target_dur + 0.05:
        final = final.subclipped(0, target_dur)

    print("\n加音乐...")
    full_audio = AudioFileClip(plan["audio_path"])
    video_dur = final.duration
    audio_end = min(edl["audio"]["start"] + video_dur, full_audio.duration)
    bgm = full_audio.subclipped(edl["audio"]["start"], audio_end)
    if video_dur > bgm.duration:
        final = final.subclipped(0, bgm.duration)
    final = final.with_audio(bgm)

    print("\n导出中...")
    final.write_videofile(
        OUTPUT_FILE, fps=30, codec="libx264",
        audio_codec="aac", bitrate=("2000k" if FAST_EXPORT else "12000k"),
        preset="medium", threads=4,
    )

    final.close()
    full_audio.close()
    for c in video_clips:
        c.close()

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print("\n✅ 完成！")
    print(f"   文件: {OUTPUT_FILE}")
    print(f"   大小: {size_mb:.2f} MB")
    if status["messages"]:
        print("   提示:")
        for m in status["messages"]:
            print(f"     - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
