"""生成合成测试素材：3 段带镜头切换的运动视频 + 4 张图片 + 1 首 128BPM 测试 BGM。"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "material"
BGM = ROOT / "bgm"
FPS = 30
W, H = 1280, 720


def make_video(path: Path, scenes: list[tuple[float, tuple[int, int, int], float]]):
    """scenes: (时长, 颜色, 运动速度)"""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for dur, color, speed in scenes:
        n = int(dur * FPS)
        for f in range(n):
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = color
            x = int((f * speed) % (W - 200))
            y = int(H / 2 + 120 * np.sin(f / 15.0))
            cv2.circle(img, (x + 100, y), 80, (255, 255, 255), -1)
            vw.write(img)
    vw.release()


def make_images():
    for i, color in enumerate([(200, 80, 40), (40, 160, 200), (90, 200, 90), (200, 200, 60)]):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = color
        cv2.circle(img, (W // 2, H // 2), 150 + i * 20, (255, 255, 255), 3)
        # cv2.imwrite 在 Windows 非 ASCII 路径下会静默失败，改用 imencode+tofile
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        buf.tofile(str(MAT / f"img_{i}.jpg"))


def make_bgm(path: Path, bpm=128.0, dur=40.0, sr=22050):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = 0.15 * np.sin(2 * np.pi * 220 * t)          # 低音持续音
    beat_int = 60.0 / bpm
    beats = np.arange(0.5, dur - 0.5, beat_int)
    for i, b in enumerate(beats):
        start = int(b * sr)
        length = int(0.08 * sr)
        env = np.exp(-np.linspace(0, 6, length))
        freq = 60 if i % 4 == 0 else 200            # 每 4 拍一个重底鼓（强拍）
        click = env * np.sin(2 * np.pi * freq * np.arange(length) / sr)
        amp = 1.0 if i % 4 == 0 else 0.6
        # 高潮（60% 处）能量加强
        amp *= 1.0 + 1.2 * np.exp(-((b / dur - 0.6) ** 2) / 0.01)
        y[start:start + length] += amp * click
    y = np.clip(y, -1, 1)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def main():
    MAT.mkdir(exist_ok=True)
    BGM.mkdir(exist_ok=True)
    make_video(MAT / "clip_a.mp4", [
        (3.0, (180, 60, 30), 18), (2.5, (30, 60, 180), 26), (3.0, (60, 180, 60), 8)])
    make_video(MAT / "clip_b.mp4", [
        (2.5, (150, 30, 120), 30), (3.0, (20, 140, 140), 12), (2.0, (90, 90, 30), 22)])
    make_video(MAT / "clip_c.mp4", [
        (3.0, (40, 40, 160), 15), (2.0, (160, 160, 40), 34), (2.5, (120, 40, 40), 20)])
    make_images()
    make_bgm(BGM / "test_beat.wav")
    make_bgm(BGM / "test_slow.wav", bpm=72.0, dur=40.0)
    print("测试素材已生成:", MAT, BGM)


if __name__ == "__main__":
    main()
