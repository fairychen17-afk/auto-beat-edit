"""端到端测试：真实素材生成 → 完整 CLI → 真实 ffmpeg 渲染 → 输出验证。

与单元测试的区别：这里跑的是完整流水线（含真实渲染），验证
时间线数学、素材裁切、音频同步、转场链、输出规格与异常处理。

运行：python -m unittest tests.test_e2e -v
（渲染需要数秒，属正常；素材在每个用例的临时目录内独立生成）
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocut.cli import main as cli_main  # noqa: E402
from autocut.renderer import ffmpeg_path  # noqa: E402

W, H, FPS = 320, 180, 30          # 低分辨率，让 e2e 足够快


# ------------------------------------------------------------ 素材生成
def _make_video(path: Path, scenes, fps=FPS, size=(W, H)):
    """scenes: [(时长, 颜色BGR, 圆点速度)]，镜头切换靠颜色突变。"""
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, size)
    for dur, color, speed in scenes:
        for f in range(int(dur * fps)):
            img = np.zeros((size[1], size[0], 3), np.uint8)
            img[:] = color
            x = int((f * speed) % (size[0] - 40))
            cv2.circle(img, (x + 20, size[1] // 2), 20, (255, 255, 255), -1)
            vw.write(img)
    vw.release()


def _make_bgm(path: Path, bpm=120.0, dur=20.0, sr=22050):
    """带强拍结构的测试 BGM：每 4 拍一个重底鼓，60% 处能量加强。"""
    import wave
    n = int(dur * sr)
    y = 0.10 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)
    for i, b in enumerate(np.arange(0.3, dur - 0.3, 60.0 / bpm)):
        s = int(b * sr)
        ln = int(0.08 * sr)
        env = np.exp(-np.linspace(0, 6, ln))
        freq = 60 if i % 4 == 0 else 200
        amp = (1.0 if i % 4 == 0 else 0.6)
        amp *= 1.0 + 1.2 * np.exp(-((b / dur - 0.6) ** 2) / 0.01)
        y[s:s + ln] += amp * env * np.sin(2 * np.pi * freq
                                          * np.arange(ln) / sr)
    y = np.clip(y, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def _probe_container(path: Path) -> dict:
    """解析 ffmpeg -i 的 stderr：时长 / 分辨率 / fps / 是否含音视频流。"""
    r = subprocess.run([ffmpeg_path(), "-i", str(path)],
                       capture_output=True, text=True, errors="ignore")
    info = {"video": "Video:" in r.stderr, "audio": "Audio:" in r.stderr,
            "duration": None, "width": None, "height": None, "fps": None}
    for ln in r.stderr.splitlines():
        ln = ln.strip()
        if ln.startswith("Duration:"):
            hms = ln.split()[1].rstrip(",")          # 00:00:06.07
            h, m, s = hms.split(":")
            info["duration"] = int(h) * 3600 + int(m) * 60 + float(s)
        if "Video:" in ln:
            import re
            mm = re.search(r"(\d{2,5})x(\d{2,5})", ln)
            if mm:
                info["width"], info["height"] = int(mm.group(1)), int(mm.group(2))
            fm = re.search(r"([\d.]+) fps", ln)
            if fm:
                info["fps"] = float(fm.group(1))
    return info


def _decoded_duration(path: Path, stream: str) -> float:
    """解码整条流取真实时长（验证音视频同步，而不是只看容器头）。"""
    r = subprocess.run([ffmpeg_path(), "-i", str(path), "-map", f"0:{stream}",
                        "-f", "null", "-"],
                       capture_output=True, text=True, errors="ignore")
    last = 0.0
    for ln in r.stderr.splitlines():
        if "time=" in ln:
            ts = ln.split("time=")[-1].split()[0]
            try:
                h, m, s = ts.split(":")
                last = int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                pass
    return last


# ------------------------------------------------------------ 测试用例
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        root = Path(cls.td.name)
        cls.mat = root / "material"
        cls.bgm_dir = root / "bgm"
        cls.mat.mkdir()
        cls.bgm_dir.mkdir()
        # 3 段视频（各 2 个镜头，颜色突变触发镜头检测），共约 13s 可用素材
        _make_video(cls.mat / "clip_a.mp4",
                    [(2.0, (180, 60, 30), 8), (2.0, (30, 60, 180), 14)])
        _make_video(cls.mat / "clip_b.mp4",
                    [(2.5, (150, 30, 120), 18), (2.0, (20, 140, 140), 6)])
        _make_video(cls.mat / "clip_c.mp4",
                    [(2.0, (40, 40, 160), 10), (2.5, (160, 160, 40), 20)])
        _make_bgm(cls.bgm_dir / "test_beat.wav", bpm=120.0, dur=20.0)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_dry_run_produces_edl(self):
        out = Path(self.td.name) / "dry.mp4"
        rc = cli_main(["--material", str(self.mat), "--bgm", str(self.bgm_dir),
                       "--output", str(out), "--duration", "5.0",
                       "--resolution", f"{W}x{H}", "--fps", str(FPS),
                       "--seed", "42", "--dry-run"])
        self.assertEqual(rc, 0)
        edl = out.with_suffix(".edl.json")
        self.assertTrue(edl.exists())
        data = json.loads(edl.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["segments"]), 2)
        self.assertGreater(data["output"]["duration"], 1.0)
        # 每段都有决策理由与卡点位置
        for seg in data["segments"]:
            self.assertIn("cut_at", seg)
            self.assertIn("music_tension", seg)
        self.assertFalse(out.exists())             # dry-run 不渲染

    def test_seed_reproducible(self):
        outs = []
        for tag in ("r1", "r2"):
            out = Path(self.td.name) / f"{tag}.mp4"
            rc = cli_main(["--material", str(self.mat), "--bgm",
                           str(self.bgm_dir), "--output", str(out),
                           "--duration", "5.0", "--resolution", f"{W}x{H}",
                           "--seed", "7", "--dry-run"])
            self.assertEqual(rc, 0)
            outs.append(json.loads(out.with_suffix(".edl.json")
                                   .read_text(encoding="utf-8"))["segments"])
        self.assertEqual(outs[0], outs[1])         # 同 seed → 同一套剪辑

    def test_full_render_and_probe(self):
        """完整渲染 + 容器探针 + EDL 一致性 + 音视频同步。"""
        out = Path(self.td.name) / "final.mp4"
        edl = Path(self.td.name) / "final.edl.json"
        rc = cli_main(["--material", str(self.mat), "--bgm", str(self.bgm_dir),
                       "--output", str(out), "--duration", "5.0",
                       "--resolution", f"{W}x{H}", "--fps", str(FPS),
                       "--seed", "42", "--edl", str(edl)])
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 20_000)   # 非空壳文件

        info = _probe_container(out)
        self.assertTrue(info["video"], "输出缺少视频流")
        self.assertTrue(info["audio"], "输出缺少音频流")
        self.assertEqual((info["width"], info["height"]), (W, H))
        self.assertAlmostEqual(info["fps"], FPS, delta=0.5)

        # 输出时长 ≈ EDL 声明的成片时长（时间线数学 ↔ 渲染结果一致）
        edl_total = json.loads(edl.read_text(encoding="utf-8"))[
            "output"]["duration"]
        self.assertAlmostEqual(info["duration"], edl_total, delta=0.4)

        # 音视频同步：分别解码两条流，真实时长差 < 0.35s
        dv = _decoded_duration(out, "v:0")
        da = _decoded_duration(out, "a:0")
        self.assertGreater(dv, 1.0)
        self.assertAlmostEqual(dv, da, delta=0.35)

    def test_full_render_with_camera_and_fx(self):
        """开全运镜 + 特效渲染一遍（手法链路真实可走通）。"""
        out = Path(self.td.name) / "full.mp4"
        rc = cli_main(["--material", str(self.mat), "--bgm", str(self.bgm_dir),
                       "--output", str(out), "--duration", "4.0",
                       "--resolution", f"{W}x{H}", "--fps", str(FPS),
                       "--seed", "3", "--camera-motion", "full",
                       "--style", "dynamic"])
        self.assertEqual(rc, 0)
        info = _probe_container(out)
        self.assertTrue(info["video"] and info["audio"])
        self.assertGreater(info["duration"], 1.0)

    def test_preserve_order_and_bgm_start(self):
        """顺序剪辑 + 指定 BGM 起点：EDL 中素材顺序与 BGM 截取正确。"""
        out = Path(self.td.name) / "ord.mp4"
        edl = Path(self.td.name) / "ord.edl.json"
        rc = cli_main(["--material", str(self.mat), "--bgm", str(self.bgm_dir),
                       "--output", str(out), "--duration", "4.0",
                       "--resolution", f"{W}x{H}", "--seed", "5",
                       "--preserve-order", "--bgm-start", "8.0",
                       "--dry-run", "--edl", str(edl)])
        self.assertEqual(rc, 0)
        data = json.loads(edl.read_text(encoding="utf-8"))
        self.assertAlmostEqual(data["bgm"]["clip_start"], 8.0, delta=0.6)
        names = [s["source"] for s in data["segments"]]
        first_seen = []
        for n in names:
            if n not in first_seen:
                first_seen.append(n)
        self.assertEqual(first_seen, sorted(first_seen))   # 文件序保持

    def test_bad_inputs_handled(self):
        """异常路径：空目录 / 不存在的 BGM / 损坏视频，全部优雅失败。"""
        empty = Path(self.td.name) / "empty"
        empty.mkdir()
        rc = cli_main(["--material", str(empty), "--bgm", str(self.bgm_dir),
                       "--output", str(Path(self.td.name) / "x.mp4")])
        self.assertEqual(rc, 1)                          # 空素材目录

        rc = cli_main(["--material", str(self.mat),
                       "--bgm", str(Path(self.td.name) / "nope.mp3"),
                       "--output", str(Path(self.td.name) / "x.mp4")])
        self.assertEqual(rc, 1)                          # BGM 不存在

        broken_dir = Path(self.td.name) / "broken"
        broken_dir.mkdir()
        (broken_dir / "bad.mp4").write_bytes(b"\x00\x01\x02 not a video")
        rc = cli_main(["--material", str(broken_dir), "--bgm",
                       str(self.bgm_dir),
                       "--output", str(Path(self.td.name) / "x.mp4")])
        self.assertEqual(rc, 1)                          # 损坏视频被跳过

        corrupt_bgm = Path(self.td.name) / "corrupt.wav"
        corrupt_bgm.write_bytes(b"RIFF\x00\x00 garbage")
        rc = cli_main(["--material", str(self.mat), "--bgm", str(corrupt_bgm),
                       "--output", str(Path(self.td.name) / "x.mp4")])
        self.assertEqual(rc, 1)                          # 损坏 BGM 报错退出


if __name__ == "__main__":
    unittest.main(verbosity=2)
