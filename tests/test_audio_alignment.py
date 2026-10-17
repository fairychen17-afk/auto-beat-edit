"""音频对齐专项测试：验证「逐帧包络 → 逐拍/逐小节/乐句」的时间戳映射正确性。

背景（回归测试的由来）：旧版 _detect_sections 直接用节拍序号 k*4 去切
逐帧 onset 包络（frames ≠ beats），导致整首歌的"结构段落"实际只反映了
开头几秒的包络帧。以下测试在旧实现上会失败，在新实现上必须通过。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocut.audio_analyzer import (  # noqa: E402
    BgmInfo, _align_downbeat_phase, _detect_sections, _tension_curve,
    beat_energies)


def make_env_step(duration=40.0, fps=10.0, split=20.0,
                  lo=0.1, hi=0.9) -> np.ndarray:
    """前半低能量、后半高能量的逐帧包络（帧率为 fps）。"""
    n = int(duration * fps)
    env = np.full(n, lo)
    env[int(split * fps):] = hi
    return env


class TestBeatEnergyMapping(unittest.TestCase):
    """逐帧包络 → 逐拍能量：必须按时间戳取"节拍领地"的均值。"""

    def test_territory_mean(self):
        fps, dur = 10.0, 10.0
        env = np.arange(int(dur * fps)) * 0.1        # env[i] = t 的线性斜坡
        beats = np.array([1.0, 2.0, 3.0])
        e = beat_energies(env, fps, beats, dur)
        # 领地：第0拍 [0,1.5) 第1拍 [1.5,2.5) 第2拍 [2.5,10]
        self.assertAlmostEqual(e[0], np.mean(env[0:15]), places=6)
        self.assertAlmostEqual(e[1], np.mean(env[15:25]), places=6)
        self.assertAlmostEqual(e[2], np.mean(env[25:100]), places=6)

    def test_beat_count_matches(self):
        env = make_env_step()
        beats = np.arange(0.0, 40.0, 0.5)
        e = beat_energies(env, 10.0, beats, 40.0)
        self.assertEqual(len(e), len(beats))         # 一拍一份能量，不多不少

    def test_step_signal_lands_on_right_beats(self):
        """能量阶跃（20s 处）必须映射到 20s 之后的拍，而不是开头几帧。"""
        env = make_env_step()
        beats = np.arange(0.0, 40.0, 0.5)
        e = beat_energies(env, 10.0, beats, 40.0)
        early = e[beats < 18.0]
        late = e[beats > 22.0]
        self.assertLess(float(early.mean()), 0.2)    # 前半拍能量低
        self.assertGreater(float(late.mean()), 0.8)  # 后半拍能量高
        # 旧 bug 的判据：若误按帧索引切片，读到的全是开头低能量帧
        self.assertGreater(float(late.mean()) - float(early.mean()), 0.5)


class TestDownbeatAlignment(unittest.TestCase):
    """小节相位对齐：downbeat 应落在 onset 最强的相位上，而不是盲目从第 0 拍起。"""

    def test_phase_detection(self):
        beat_e = np.tile(np.array([0.2, 0.9, 0.3, 0.3]), 10)  # 相位 1 最强
        self.assertEqual(_align_downbeat_phase(beat_e), 1)

    def test_phase_zero_when_strong_first(self):
        beat_e = np.tile(np.array([0.9, 0.3, 0.5, 0.3]), 10)
        self.assertEqual(_align_downbeat_phase(beat_e), 0)

    def test_short_input_safe(self):
        self.assertEqual(_align_downbeat_phase(np.array([0.5, 0.4])), 0)


class TestSectionDetection(unittest.TestCase):
    """结构段落：必须基于小节能量（bar_e），时间边界取小节头时间戳。"""

    def _bar_grid(self, duration=40.0, bpm=120.0, split=20.0):
        beat_int = 60.0 / bpm                          # 0.5s
        beats = np.arange(0.0, duration, beat_int)
        env = make_env_step(duration, fps=10.0, split=split)
        beat_e = beat_energies(env, 10.0, beats, duration)
        bar_times = beats[::4]                         # 相位 0
        bar_e = np.array([beat_e[i:i + 4].mean()
                          for i in range(0, len(beats), 4)])
        return beats, beat_e, bar_times, bar_e

    def test_two_halves_become_low_then_high(self):
        beats, beat_e, bar_times, bar_e = self._bar_grid()
        secs = _detect_sections(beats, beat_e, bar_times, bar_e, 40.0)
        levels = [lv for _, _, lv in secs]
        self.assertEqual(levels[0], "low")             # 前半低
        self.assertEqual(levels[-1], "high")           # 后半高
        self.assertGreaterEqual(len(secs), 2)          # 确实分出了结构
        # 段落边界必须是小节头时间（20s 附近，过渡小节并入前段属正常），
        # 而不是帧号错位导致的开头几秒
        boundary = secs[1][0]
        self.assertAlmostEqual(boundary, 20.0, delta=2.5)
        self.assertGreater(boundary, 10.0)   # 旧 bug 下边界会落在前几秒

    def test_section_time_bounds_are_bar_aligned(self):
        beats, beat_e, bar_times, bar_e = self._bar_grid()
        secs = _detect_sections(beats, beat_e, bar_times, bar_e, 40.0)
        bar_set = set(np.round(bar_times, 3))
        for s, e, _ in secs:
            self.assertIn(round(s, 3), bar_set | {0.0})
        # 覆盖全曲
        self.assertAlmostEqual(secs[0][0], 0.0, places=3)
        self.assertAlmostEqual(secs[-1][1], 40.0, places=3)

    def test_flat_energy_single_section(self):
        beats = np.arange(0.0, 40.0, 0.5)
        env = np.full(400, 0.5)
        beat_e = beat_energies(env, 10.0, beats, 40.0)
        bar_times = beats[::4]
        bar_e = np.array([beat_e[i:i + 4].mean()
                          for i in range(0, len(beats), 4)])
        secs = _detect_sections(beats, beat_e, bar_times, bar_e, 40.0)
        self.assertEqual(len(secs), 1)                 # 平坦能量 → 单一中段
        self.assertEqual(secs[0][2], "mid")


class TestTensionCurve(unittest.TestCase):
    """张力曲线：由小节能量平滑生成，峰值位置、梯度符号可验证。"""

    def test_peak_follows_energy_peak(self):
        bar_times = np.arange(0.0, 40.0, 2.0)          # 20 小节
        bar_e = np.exp(-((np.arange(20) - 12) ** 2) / 18.0)  # 峰值在第 12 小节
        t, v = _tension_curve(bar_times, bar_e, 40.0)
        peak_t = t[int(np.argmax(v))]
        self.assertAlmostEqual(peak_t, bar_times[12] + 1.0, delta=2.0)

    def test_gradient_sign(self):
        from autocut.audio_analyzer import BgmInfo as _B
        bar_times = np.arange(0.0, 40.0, 2.0)
        bar_e = np.linspace(0.1, 0.9, 20)              # 单调爬升
        t, v = _tension_curve(bar_times, bar_e, 40.0)
        bgm = _B(path=Path("x.mp3"), duration=40.0, bpm=120.0,
                 beats=np.arange(0, 40, 0.5),
                 strong_beats=np.arange(0, 40, 2.0), climax=30.0,
                 energy_mean=0.5, mood="chill",
                 tension_t=t, tension_v=v)
        self.assertGreater(bgm.tension_grad_at(10.0), 0.0)   # 爬升段梯度为正
        self.assertGreater(bgm.tension_at(30.0),
                           bgm.tension_at(5.0))              # 后段张力更高

    def test_phrase_tail_dip(self):
        """乐句末小节张力轻微衰减（回合感）。"""
        bar_times = np.arange(0.0, 32.0, 2.0)          # 16 小节 = 4 乐句
        bar_e = np.full(16, 0.8)
        _, v = _tension_curve(bar_times, bar_e, 32.0)
        # 端点拼接后取中间小节比较：乐句末（3,7,11,15）应低于乐句中
        mid = v[3]      # bar index 2 (v 多了首尾端点 → +1 偏移)
        tail = v[4]     # bar index 3（乐句末）
        self.assertLess(tail, mid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
