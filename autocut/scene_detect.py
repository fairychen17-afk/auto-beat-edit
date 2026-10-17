"""素材分析：镜头检测 / 画质评估 / 运动分析 三层分离。

能力边界（诚实声明，避免把启发式当成语义理解）：
- **镜头检测**：HSV 直方图差异切镜头 —— 可靠，是成熟的经典方法；
- **画质评估**：Laplacian 方差（清晰度）+ 曝光裁剪率（死黑/死白占比），
  只回答「这帧拍得好不好」，不回答「拍的是什么」；
- **运动分析**：灰度帧差均值（运动强度）+ 左右半区帧差不对称（运动方向），
  只反映像素变化，无法区分「镜头在动」还是「主体在动」；
- **语义理解**：本系统【没有】接入任何视觉语义模型。subject_center / shot_type
  只是「构图边缘集中度」的启发式估计：中心边缘密度高 ≈ 画面信息集中在中间
  （常见于特写，但也可能是居中文字、对称构图）。所有下游决策都应把它当作
  概率性提示，而不是确定性结论。

关键帧策略：镜头内【清晰度最高】且远离镜头边界的帧（运动峰值帧常是抖动/
横穿瞬间，不适合当卡点帧——这是对旧策略的修正）。
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
EDGE_MARGIN = 0.15             # 关键帧避让镜头边界的比例（防抖/切换残影）


@dataclass
class Shot:
    start: float
    end: float
    keyframe: float            # 镜头内清晰度峰值时间（最佳卡点/封面帧）
    motion: float              # 镜头平均运动强度 0~1（帧差启发式）
    brightness: float = 0.5    # 镜头平均亮度 0~1
    colorfulness: float = 0.5  # 镜头色彩丰富度 0~1
    motion_dir: float = 0.0    # 横向像素变化趋势：<0 左侧变化大，>0 右侧变化大
    subject_center: float = 0.0  # 构图边缘集中度 0~1（启发式，非语义主体识别）
    shot_type: str = "medium"  # 由构图集中度推断的近似景别（启发式，见模块 docstring）
    is_static: bool = False    # 近乎静止镜头（帧差很小）
    sharpness: float = 0.0     # 清晰度（Laplacian 方差，按素材内 min-max 归一）
    exposure_ok: float = 1.0   # 曝光正常度 0~1（1 - 死黑/死白像素占比）
    quality: float = 0.5       # 画质综合分 0~1（清晰度×曝光，供窗口选择排序）

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
    """近似景别推断：构图边缘集中度 + 运动平稳度的启发式分类。

    注意：这不是人脸/语义景别识别。中心边缘密集常对应特写构图，
    分散且平稳常对应全景——但对称图案、居中文字都会误判为"特写"。
    下游只用它做「手法适用性」的软性提示（如疑似特写则少用破坏性特效），
    不做硬性结论。
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


def _quality_stats(gray: np.ndarray) -> tuple[float, float]:
    """画质评估：Laplacian 方差（清晰度）+ 曝光正常度（1-死黑死白占比）。"""
    lap = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    u8 = gray.astype(np.uint8)
    clipped = float(((u8 < 8) | (u8 > 247)).mean())
    return lap, max(0.0, 1.0 - clipped * 1.5)   # 裁剪 33% 以上 → 0


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
    lap_raw, expo_raw = [], []
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
        # 构图边缘集中度：中心 1/3 宽带的边缘密度相对全图（启发式构图指标）
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        w = edges.shape[1]
        cd = edges[:, w // 3: 2 * w // 3].mean()
        gd = edges.mean()
        subj_raw.append(min(1.0, cd / max(gd, 1e-6) / 2.0))   # 均匀分布≈0.5，集中≈1
        # 画质：清晰度 + 曝光
        lap, expo = _quality_stats(gray)
        lap_raw.append(lap)
        expo_raw.append(expo)
        # 明暗与色彩（直接用 small，速度快）
        b_raw.append(float(hsv[..., 2].mean()) / 255.0)
        c_raw.append(float(hsv[..., 1].mean()) / 255.0)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - max(0.0, corr)          # 0=同画面 1=完全不同
            m = float(np.abs(gray - prev_gray).mean()) / 255.0
            motion_raw.append(min(1.0, m * 6.0 + diff * 0.5))
            # 横向像素变化不对称：哪一侧变化大（像素级，不等于主体运动方向）
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

    # 清晰度按本素材 min-max 归一（绝对值随分辨率/编码差异大，相对比较才有意义）
    lap_arr = np.array(lap_raw)
    lap_lo, lap_hi = float(lap_arr.min()), float(lap_arr.max())
    sharp_norm = ((lap_arr - lap_lo) / max(lap_hi - lap_lo, 1e-9)
                  if lap_hi > lap_lo else np.full_like(lap_arr, 0.5))

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

    def _keyframe(a: float, b: float, mask: np.ndarray) -> float:
        """关键帧 = 镜头内清晰度最高、且避开首尾 EDGE_MARGIN 区域的帧。

        旧策略取运动峰值帧——那是抖动/横穿瞬间，往往是最糊的帧。
        """
        seg_times = times[mask]
        seg_sharp = sharp_norm[mask]
        span = b - a
        inner = ((seg_times >= a + span * EDGE_MARGIN)
                 & (seg_times <= b - span * EDGE_MARGIN))
        cand = np.where(inner)[0]
        if len(cand) == 0:                      # 镜头太短，退化为取中点
            return float((a + b) / 2)
        ki = int(cand[int(np.argmax(seg_sharp[cand]))])
        return float(seg_times[ki])

    shots: list[Shot] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        mask = (times >= a) & (times <= b)
        if not mask.any():
            continue
        seg_motion = motion_values[mask]
        subj = float(np.array(subj_raw)[mask].mean())
        m_mean = float(seg_motion.mean())
        cf = float(np.array(c_raw)[mask].mean())
        sharp = float(sharp_norm[mask].mean())
        expo = float(np.array(expo_raw)[mask].mean())
        shots.append(Shot(start=a, end=b,
                          keyframe=_keyframe(a, b, mask),
                          motion=m_mean,
                          brightness=float(np.array(b_raw)[mask].mean()),
                          colorfulness=cf,
                          motion_dir=float(np.array(dir_raw)[mask].mean()),
                          subject_center=subj,
                          shot_type=_classify_shot(subj, m_mean, cf),
                          is_static=bool(m_mean < 0.22),
                          sharpness=sharp,
                          exposure_ok=expo,
                          quality=float(np.clip(0.65 * sharp + 0.35 * expo,
                                                0.0, 1.0))))
    if not shots:
        subj = float(np.mean(subj_raw))
        m_mean = float(motion_values.mean())
        cf = float(np.mean(c_raw))
        sharp = float(sharp_norm.mean())
        expo = float(np.mean(expo_raw))
        shots.append(Shot(0.0, duration, duration / 2, m_mean,
                          brightness=float(np.mean(b_raw)), colorfulness=cf,
                          motion_dir=0.0, subject_center=subj,
                          shot_type=_classify_shot(subj, m_mean, cf),
                          is_static=bool(m_mean < 0.22), sharpness=sharp,
                          exposure_ok=expo,
                          quality=float(np.clip(0.65 * sharp + 0.35 * expo,
                                                0.0, 1.0))))

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
    # 图片也做画质评估（清晰/曝光），与视频同一评分体系
    gray = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY)
    lap, expo = _quality_stats(gray)
    sharp = min(1.0, lap / 400.0)                # 单张图无相对比较，用绝对尺度
    return Asset(path=path, kind="image", duration=0.0,
                 brightness=b, colorfulness=c, energy=energy,
                 shots=[Shot(0.0, 0.0, 0.0, 0.0, brightness=b, colorfulness=c,
                             sharpness=sharp, exposure_ok=expo,
                             quality=float(np.clip(0.65 * sharp + 0.35 * expo,
                                                   0.0, 1.0)))])


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
