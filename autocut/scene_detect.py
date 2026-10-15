"""素材分析：视频镜头/关键帧检测 + 运动强度 + 画面情绪特征。

视频：按固定间隔抽帧，HSV 直方图差异检测镜头切换；
      帧差均值估计运动强度曲线；每个镜头内运动峰值即关键帧。
图片：记录亮度/色彩丰富度，供情绪匹配使用。
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAMPLE_INTERVAL = 0.2          # 抽帧间隔（秒）
SCENE_THRESHOLD = 0.45         # 直方图相关性差异阈值（越小越灵敏）
MIN_SHOT_LEN = 0.8             # 最短镜头时长（秒）


@dataclass
class Shot:
    start: float
    end: float
    keyframe: float            # 镜头内运动峰值时间（最佳卡点）
    motion: float              # 镜头平均运动强度 0~1
    brightness: float = 0.5    # 镜头平均亮度 0~1
    colorfulness: float = 0.5  # 镜头色彩丰富度 0~1
    motion_dir: float = 0.0    # 横向运动趋势：<0 内容左移，>0 右移，≈0 无方向
    subject_center: float = 0.0  # 主体居中度 0~1（中心区域边缘密度占比，>0.55 构图主体明显）
    shot_type: str = "medium"  # 景别："close" 特写 / "medium" 中景 / "wide" 全景
    is_static: bool = False    # 近乎静止镜头（可容纳更多花样，画面不会花）

    @property
    def length(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class Asset:
    path: Path
    kind: str                  # "video" | "image"
    duration: float
    brightness: float          # 0~1
    colorfulness: float        # 0~1
    energy: float              # 综合情绪能量 0~1
    shots: list[Shot] = field(default_factory=list)
    motion_times: np.ndarray | None = None
    motion_values: np.ndarray | None = None


def _classify_shot(subject_center: float, motion: float,
                   colorfulness: float) -> str:
    """景别判定：主体居中且信息密集 → 特写；空旷少动 → 全景；其余中景。

    这不是严格的人脸/景别识别，而是「构图信息量」的近似：
    特写画面的边缘能量集中在中间窄带，全景则分散且画面平稳。
    """
    if subject_center >= 0.56:
        return "close"
    if subject_center <= 0.44 and motion <= 0.40:
        return "wide"
    return "medium"


def _frame_stats(bgr: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[..., 2].mean()) / 255.0
    # Hasler-Süsstrunk 色彩丰富度
    b, g, r = bgr[..., 0].astype(float), bgr[..., 1].astype(float), bgr[..., 2].astype(float)
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = min(1.0, (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                             + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 128.0)
    return brightness, colorfulness


def analyze_video(path: Path) -> Asset | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total / fps if fps > 0 and total > 0 else 0.0
    if duration < 0.5:
        cap.release()
        return None

    step = max(1, int(fps * SAMPLE_INTERVAL))
    times, hists, motion_raw = [], [], []
    dir_raw, b_raw, c_raw, subj_raw = [], [], [], []
    prev_hist, prev_gray = None, None

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        times.append(t)
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # 主体居中度：中心 1/3 宽带的边缘密度相对全图
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        w = edges.shape[1]
        cd = edges[:, w // 3: 2 * w // 3].mean()
        gd = edges.mean()
        subj_raw.append(min(1.0, cd / max(gd, 1e-6) / 2.0))   # 均匀分布≈0.5，集中≈1
        # 明暗与色彩（直接用 small，速度快）
        b_raw.append(float(hsv[..., 2].mean()) / 255.0)
        c_raw.append(float(hsv[..., 1].mean()) / 255.0)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - max(0.0, corr)          # 0=同画面 1=完全不同
            m = float(np.abs(gray - prev_gray).mean()) / 255.0
            motion_raw.append(min(1.0, m * 6.0 + diff * 0.5))
            # 横向运动趋势：哪一侧变化大说明主体/镜头朝那边动
            lw, rw = w // 2, w - w // 2
            d = (np.abs(gray[:, w // 2:] - prev_gray[:, w // 2:]).mean()
                 - np.abs(gray[:, :w // 2] - prev_gray[:, :w // 2]).mean())
            dir_raw.append(float(np.clip(d / 20.0, -1.0, 1.0)))
        else:
            motion_raw.append(0.0)
            dir_raw.append(0.0)
        prev_hist, prev_gray = hist, gray
        idx += step
    cap.release()

    if len(times) < 2:
        return None

    times = np.array(times)
    motion_values = np.array(motion_raw)
    duration = float(max(duration, times[-1]))

    # ---- 镜头切分 ----
    boundaries = [0.0]
    for i in range(1, len(hists)):
        corr = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        if (1.0 - max(0.0, corr)) > SCENE_THRESHOLD:
            t = float(times[i])
            if t - boundaries[-1] >= MIN_SHOT_LEN:
                boundaries.append(t)
    if duration - boundaries[-1] < MIN_SHOT_LEN and len(boundaries) > 1:
        boundaries.pop()                        # 尾部太短并入上一镜头
    boundaries.append(duration)

    shots: list[Shot] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        mask = (times >= a) & (times <= b)
        if not mask.any():
            continue
        seg_motion = motion_values[mask]
        seg_times = times[mask]
        ki = int(np.argmax(seg_motion))
        subj = float(np.array(subj_raw)[mask].mean())
        m_mean = float(seg_motion.mean())
        shots.append(Shot(start=a, end=b,
                          keyframe=float(seg_times[ki]),
                          motion=m_mean,
                          brightness=float(np.array(b_raw)[mask].mean()),
                          colorfulness=float(np.array(c_raw)[mask].mean()),
                          motion_dir=float(np.array(dir_raw)[mask].mean()),
                          subject_center=subj,
                          shot_type=_classify_shot(
                              subj, m_mean, float(np.array(c_raw)[mask].mean())),
                          is_static=bool(m_mean < 0.22)))
    if not shots:
        subj = float(np.mean(subj_raw))
        m_mean = float(motion_values.mean())
        shots.append(Shot(0.0, duration, duration / 2, m_mean,
                          brightness=float(np.mean(b_raw)),
                          colorfulness=float(np.mean(c_raw)),
                          motion_dir=0.0, subject_center=subj,
                          shot_type=_classify_shot(subj, m_mean, float(np.mean(c_raw))),
                          is_static=bool(m_mean < 0.22)))

    energy = float(np.clip(np.percentile(motion_values, 75) * 1.5, 0, 1))
    return Asset(path=path, kind="video", duration=duration,
                 brightness=float(np.mean(b_raw)),
                 colorfulness=float(np.mean(c_raw)),
                 energy=energy, shots=shots,
                 motion_times=times, motion_values=motion_values)


def analyze_image(path: Path) -> Asset | None:
    # cv2.imread 在 Windows 非 ASCII 路径下会失败，改用 imdecode
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    b, c = _frame_stats(img)
    energy = float(np.clip(0.3 * b + 0.7 * c, 0, 1))   # 图片情绪≈色彩丰富度
    return Asset(path=path, kind="image", duration=0.0,
                 brightness=b, colorfulness=c, energy=energy)


def scan_materials(material_dir: Path) -> list[Asset]:
    """扫描素材目录，返回全部可用素材（视频按镜头、图片按张）。"""
    assets: list[Asset] = []
    for p in sorted(material_dir.iterdir()):
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            a = analyze_video(p)
        elif ext in IMAGE_EXTS:
            a = analyze_image(p)
        else:
            continue
        if a is not None:
            assets.append(a)
    return assets
