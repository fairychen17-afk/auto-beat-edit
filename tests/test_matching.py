"""需求-供给匹配机制测试：证明剪辑决策是评分驱动的，而非随机采样式排雷。

核心断言逻辑：
- 同一输入 → 同一输出（确定性），换 seed 只在精确平局时才有差异；
- 高张力槽位必须拿到高质量素材窗口（匹配而不是随机穿插）；
- 段长决策跟随张力/张力梯度，且切点优先落在小节头；
- 手法选择跟随需求画像（张力、乐句角色、明暗、画质），且 why 有证据。
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocut.audio_analyzer import BgmInfo  # noqa: E402
from autocut.moves import GENTLE_MOVES, plan_shots  # noqa: E402
from autocut.scene_detect import Asset, Shot  # noqa: E402
from autocut.timeline import Unit, _beat_counts, _match_units, _target_beats  # noqa: E402


def make_bgm(bpm=120.0, dur=40.0, tension_fn=None) -> BgmInfo:
    """带真实张力曲线的 BgmInfo（不依赖音频文件）。"""
    beats = np.arange(0.0, dur, 60.0 / bpm)
    bar_times = beats[::4]
    if tension_fn is None:
        tension_fn = lambda t: 0.5                    # noqa: E731
    ct = np.linspace(0.0, dur, 200)
    cv = np.array([tension_fn(x) for x in ct])
    return BgmInfo(path=Path("fake.mp3"), duration=dur, bpm=bpm, beats=beats,
                   strong_beats=bar_times, climax=dur * 0.6, energy_mean=0.6,
                   mood="energetic",
                   onset_env=np.full(int(dur * 10), 0.5, dtype=np.float32),
                   onset_fps=10.0,
                   sections=[(0.0, dur, "mid")],
                   bar_times=bar_times, tension_t=ct, tension_v=cv,
                   downbeat_phase=0)


def feat(**kw) -> dict:
    base = dict(energy=0.5, tension=0.5, tension_grad=0.0, phrase_start=False,
                brightness=0.5, colorfulness=0.5, subject_center=0.3,
                quality=0.5, motion=0.4, motion_dir=0.0, shot_type="medium",
                section="mid", boundary=False, jump=False, kind="video")
    base.update(kw)
    return base


class TestDeterminism(unittest.TestCase):
    """评分制的根本性质：输入确定 → 输出确定。"""

    def test_same_input_same_plan(self):
        feats = [feat(tension=0.8, section="high", brightness=0.3),
                 feat(tension=0.3, section="low", brightness=0.7),
                 feat(tension=0.6, motion=0.8, quality=0.9)]
        p1 = plan_shots(feats, "energetic", "dynamic", random.Random(1),
                        camera_motion="full")
        p2 = plan_shots(feats, "energetic", "dynamic", random.Random(1),
                        camera_motion="full")
        self.assertEqual(p1, p2)

    def test_seed_only_breaks_exact_ties(self):
        """分数有区分度时，换 seed 不得改变结果。"""
        feats = [feat(tension=0.85, section="high", brightness=0.2),
                 feat(tension=0.2, section="low")]
        p1 = plan_shots(feats, "energetic", "dynamic", random.Random(1),
                        camera_motion="full")
        p2 = plan_shots(feats, "energetic", "dynamic", random.Random(999),
                        camera_motion="full")
        self.assertEqual([p["move"] for p in p1], [p["move"] for p in p2])
        self.assertEqual([p["fx"] for p in p1], [p["fx"] for p in p2])
        self.assertEqual([p["transition"] for p in p1],
                         [p["transition"] for p in p2])

    def test_input_change_changes_plan(self):
        """素材/音乐特征变了，方案必须变（否则就是伪感知）。"""
        base = [feat(tension=0.8, section="high", brightness=0.3, motion=0.3)]
        changed = [feat(tension=0.8, section="high", brightness=0.3,
                        motion=0.3, shot_type="close", subject_center=0.8)]
        p1 = plan_shots(base, "energetic", "dynamic", random.Random(1),
                        camera_motion="full")[0]
        p2 = plan_shots(changed, "energetic", "dynamic", random.Random(1),
                        camera_motion="full")[0]
        # 疑似特写 → 破坏性特效被禁，运镜收敛到安全集
        self.assertNotEqual((p1["move"], p1["fx"]), (p2["move"], p2["fx"]))


class TestTensionDrivenCounts(unittest.TestCase):
    """段长决策：张力驱动 + 小节对齐。"""

    def test_target_beats_follow_tension(self):
        self.assertEqual(_target_beats(0.9, 0.0, "energetic"), 1)
        self.assertEqual(_target_beats(0.1, 0.0, "energetic"), 4)
        # 张力上升再压一档（build 渐密），回落放长一档
        self.assertEqual(_target_beats(0.6, 0.10, "energetic"), 1)
        self.assertEqual(_target_beats(0.6, -0.10, "energetic"), 3)

    def test_cut_points_prefer_bar_heads(self):
        """均匀张力下，切点应优先落在小节头（(累计拍-相位) % 4 == 0）。"""
        bgm = make_bgm(tension_fn=lambda t: 0.5)      # 平坦张力
        counts = _beat_counts(bgm, 0, 10, "chill")
        cum = np.cumsum([0] + counts)
        on_bar = sum(1 for c in cum[1:] if c % 4 == 0)
        # 超过一半的切点落在小节头（随机或纯阈值策略做不到这一点）
        self.assertGreaterEqual(on_bar / max(len(counts), 1), 0.5,
                                f"counts={counts}")

    def test_rising_tension_densifies(self):
        """张力持续爬升段，段长应单调不增（越切越密）。"""
        bgm = make_bgm(tension_fn=lambda t: 0.2 + 0.6 * (t / 40.0))
        counts = _beat_counts(bgm, 0, 10, "energetic")
        # 后半段平均拍数应不大于前半段
        half = len(counts) // 2
        self.assertLessEqual(float(np.mean(counts[half:])),
                             float(np.mean(counts[:half])) + 0.01,
                             f"counts={counts}")


class TestUnitMatching(unittest.TestCase):
    """需求-供给匹配：高张力槽位拿高质量窗口。"""

    def _unit(self, quality, motion=0.4, length=2.0, tag=0) -> Unit:
        sh = Shot(0.0, length, length / 2, motion, quality=quality)
        a = Asset(path=Path(f"u{tag}.mp4"), kind="video", duration=length,
                  brightness=0.5, colorfulness=0.5, energy=0.5, shots=[sh])
        return Unit(a, sh, 0.0, length, length / 2, motion, quality)

    def test_high_tension_slot_gets_best_unit(self):
        good = self._unit(0.95, tag=1)
        poor = self._unit(0.10, tag=2)
        tensions = [0.9, 0.1]                        # 槽位0 张力高
        wants = [2.0, 2.0]
        slots = _match_units(tensions, wants, [poor, good], allow_reuse=False)
        self.assertIs(slots[0], good)              # 高张力段拿到好素材
        self.assertIs(slots[1], poor)

    def test_no_reuse_by_default(self):
        us = [self._unit(0.9 - i * 0.1, tag=i) for i in range(4)]
        slots = _match_units([0.9, 0.7, 0.5, 0.3], [2.0] * 4, us,
                             allow_reuse=False)
        self.assertEqual(len({id(u) for u in slots}), 4)   # 0 重复

    def test_allow_reuse_wraps(self):
        us = [self._unit(0.8, tag=1)]
        slots = _match_units([0.9, 0.5], [2.0, 2.0], us, allow_reuse=True)
        self.assertEqual(len(slots), 2)


class TestScoreDrivenTechniques(unittest.TestCase):
    """手法选择跟随需求画像（而不是随机池）。"""

    def test_high_tension_prefers_impact_camera(self):
        p = plan_shots([feat(tension=0.9, phrase_start=True)],
                       "energetic", "dynamic", random.Random(1),
                       camera_motion="full")[0]
        self.assertIn(p["move"], ("crash_zoom", "zoom_punch", "heartbeat",
                                  "punch_pan"))

    def test_low_tension_prefers_gentle_camera(self):
        p = plan_shots([feat(tension=0.15, section="low")],
                       "calm", "cinematic", random.Random(1),
                       camera_motion="full")[0]
        self.assertIn(p["move"], ("breath", "drift", "zoom_out", "ken_diag",
                                  "zoom_in", "pan_up", "pan_down", "static"))
        self.assertNotIn(p["move"], ("shake", "punch_pan", "zoom_punch"))

    def test_dark_bright_contrast_rule(self):
        dark = plan_shots([feat(tension=0.85, section="high", brightness=0.2)],
                          "energetic", "dynamic", random.Random(1))[0]
        self.assertIn(dark["fx"], ("beat_flash", "flash_white"))
        self.assertIn("暗画面配亮闪", dark["why"])

    def test_tension_gradient_drives_ramp(self):
        up = plan_shots([feat(tension=0.5, tension_grad=0.10)],
                        "chill", "balanced", random.Random(1))[0]
        self.assertEqual(up["ramp"], (0.6, 1.7))
        down = plan_shots([feat(tension=0.7, tension_grad=-0.10)],
                          "chill", "balanced", random.Random(1))[0]
        self.assertEqual(down["ramp"], (1.7, 0.6))
        flat = plan_shots([feat(tension=0.5, tension_grad=0.0)],
                          "chill", "balanced", random.Random(1))[0]
        self.assertIsNone(flat["ramp"])

    def test_slowmo_only_for_good_footage(self):
        """慢动作展示只给画质好的窗口（画质差慢放=暴露缺陷）。"""
        good = plan_shots([feat(tension=0.3, quality=0.9)],
                          "calm", "cinematic", random.Random(1))[0]
        poor = plan_shots([feat(tension=0.3, quality=0.2)],
                          "calm", "cinematic", random.Random(1))[0]
        self.assertEqual(good["speed"], 0.62)
        self.assertEqual(poor["speed"], 1.0)

    def test_why_is_auditable(self):
        p = plan_shots([feat(tension=0.9, section="high", brightness=0.2,
                             phrase_start=True)],
                       "energetic", "dynamic", random.Random(1),
                       camera_motion="full")[0]
        self.assertTrue(p["why"])                    # 决策理由非空
        self.assertNotEqual(p["why"], "；")

    def test_camera_off_still_stable(self):
        plan = plan_shots([feat(shot_type="wide", tension=0.9),
                           feat(kind="image", shot_type="image", tension=0.9)],
                          "energetic", "dynamic", random.Random(1),
                          camera_motion="off")
        self.assertEqual(plan[0]["move"], "static")
        self.assertEqual(plan[1]["move"], "breath")

    def test_subtle_stays_gentle(self):
        feats = [feat(tension=0.9, motion=0.8, motion_dir=0.6, boundary=True,
                      shot_type="wide")] * 6
        plan = plan_shots(feats, "energetic", "dynamic", random.Random(1),
                          camera_motion="subtle")
        self.assertTrue(all(p["move"] in GENTLE_MOVES for p in plan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
