"""单元测试：核心编排逻辑（不需要素材文件，纯数据驱动）。

运行：python -m unittest discover -s tests -v
或   python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocut import (Asset, BgmInfo, Segment, Shot, Timeline,  # noqa: E402
                     auto_style, export_edl, material_profile, pick_bgm)
from autocut.moves import (CAMERA_POOL, FX_POOL, GRADE_LOW_SAT,  # noqa: E402
                           GRADE_POOL, TRANS_TIERS, camera_chain, fx_chain,
                           plan_shots, resolve_style)
from autocut.timeline import _beat_counts, _structure_flags  # noqa: E402
import random  # noqa: E402


def make_bgm(bpm: float = 120.0, dur: float = 30.0, mood: str = "energetic",
             energy: float = 0.7, sections=None) -> BgmInfo:
    beat = 60.0 / bpm
    beats = np.arange(0.0, dur, beat)
    env_fps = 100.0
    env = np.clip(0.5 + 0.5 * np.sin(np.linspace(0, 12, int(dur * env_fps))), 0, 1)
    return BgmInfo(path=Path("fake.mp3"), duration=dur, bpm=bpm, beats=beats,
                   strong_beats=beats[::4], climax=dur * 0.6,
                   energy_mean=energy, mood=mood,
                   onset_env=env.astype(np.float32), onset_fps=env_fps,
                   sections=sections or [(0.0, dur, "high")])


def make_shot(start=0.0, end=2.0, keyframe=1.0, motion=0.5, brightness=0.5,
              colorfulness=0.5, motion_dir=0.0, subject_center=0.3,
              shot_type="medium", is_static=False) -> Shot:
    return Shot(start=start, end=end, keyframe=keyframe, motion=motion,
                brightness=brightness, colorfulness=colorfulness,
                motion_dir=motion_dir, subject_center=subject_center,
                shot_type=shot_type, is_static=is_static)


def make_asset(name="v.mp4", shots=None, energy=0.7, brightness=0.5,
               colorfulness=0.5) -> Asset:
    return Asset(path=Path(name), kind="video", duration=6.0,
                 brightness=brightness, colorfulness=colorfulness,
                 energy=energy, shots=shots or [make_shot()])


class TestShotClassification(unittest.TestCase):
    def test_types(self):
        """景别分类是纯函数，直接验证阈值行为。"""
        from autocut.scene_detect import _classify_shot
        self.assertEqual(_classify_shot(0.70, 0.5, 0.5), "close")
        self.assertEqual(_classify_shot(0.40, 0.20, 0.5), "wide")
        self.assertEqual(_classify_shot(0.50, 0.60, 0.5), "medium")

    def test_shot_length_property(self):
        self.assertAlmostEqual(make_shot(start=1.0, end=2.5).length, 1.5)


class TestBeatGrid(unittest.TestCase):
    def test_counts_are_beat_multiples_and_energy_driven(self):
        bgm = make_bgm(bpm=172.0, dur=40.0)
        counts = _beat_counts(bgm, 0, 8, "energetic")
        self.assertTrue(counts)
        self.assertTrue(all(1 <= c <= 4 for c in counts))
        self.assertLessEqual(sum(counts), len(bgm.beats))

    def test_low_energy_gives_longer_segments(self):
        bgm = make_bgm(bpm=120.0, dur=40.0, energy=0.2)
        bgm.onset_env = np.full_like(bgm.onset_env, 0.1)
        counts = _beat_counts(bgm, 0, 6, "energetic")
        self.assertTrue(all(c >= 4 for c in counts), counts)

    def test_structure_flags_from_sections(self):
        """结构边界标记直接来自修正后的 sections（10s 处低→高）。"""
        bgm = make_bgm(bpm=120.0, dur=30.0,
                       sections=[(0.0, 10.0, "low"), (10.0, 30.0, "high")])
        flags = _structure_flags(bgm, 0, [4] * 6)   # 段起点 0/2/4/6/8/10s
        self.assertFalse(flags[0])
        self.assertTrue(flags[5])                   # 10s 处命中边界
        self.assertFalse(any(flags[1:5]))           # 其余不误报


class TestPlanShotsRules(unittest.TestCase):
    def _plan(self, feats, style="dynamic", camera="off"):
        return plan_shots(feats, "energetic", style, random.Random(1),
                          camera_motion=camera)

    def test_close_up_protected(self):
        feats = [dict(energy=0.8, brightness=0.3, colorfulness=0.4,
                      subject_center=0.75, motion=0.5, motion_dir=0.0,
                      shot_type="close", kind="video", boundary=False,
                      section="high")]
        p = self._plan(feats)[0]
        self.assertNotIn(p["fx"], ("split_mirror", "hflip", "strobe", "chroma",
                                   "glitch", "pip_card", "flash_freeze"))
        self.assertIn("close", "close")          # 规则命中说明
        self.assertIn("特写", p["why"])

    def test_dark_scene_gets_bright_flash(self):
        feats = [dict(energy=0.85, brightness=0.2, colorfulness=0.5,
                      subject_center=0.3, motion=0.4, motion_dir=0.0,
                      shot_type="medium", kind="video", boundary=False,
                      section="high")]
        p = self._plan(feats)[0]
        if p["fx"] in ("flash_white", "beat_flash"):
            self.assertIn("亮闪", p["why"])

    def test_jump_cut_uses_hard_cut_duration(self):
        feats = [dict(energy=0.8, brightness=0.6, colorfulness=0.5,
                      subject_center=0.3, motion=0.4, motion_dir=0.0,
                      shot_type="medium", kind="video", boundary=False,
                      section="high", jump=True)]
        p = self._plan(feats)[0]
        self.assertLessEqual(p["tdur"], 0.06)
        self.assertIn("跳切", p["why"])

    def test_directional_transition_matches_motion(self):
        feats = [dict(energy=0.8, brightness=0.6, colorfulness=0.5,
                      subject_center=0.3, motion=0.5, motion_dir=0.8,
                      shot_type="medium", kind="video", boundary=False,
                      section="high")]
        p = self._plan(feats)[0]
        from autocut.moves import DIRECTIONAL_TRANS
        self.assertIn(p["transition"], DIRECTIONAL_TRANS["right"])

    def test_camera_off_yields_static(self):
        feats = [dict(energy=0.7, brightness=0.5, colorfulness=0.5,
                      subject_center=0.3, motion=0.5, motion_dir=0.0,
                      shot_type="medium", kind="video", boundary=False,
                      section="high"),
                 dict(energy=0.7, brightness=0.5, colorfulness=0.5,
                      subject_center=0.0, motion=0.0, motion_dir=0.0,
                      shot_type="image", kind="image", boundary=False,
                      section="high")]
        plan = self._plan(feats, camera="off")
        self.assertEqual(plan[0]["move"], "static")
        self.assertEqual(plan[1]["move"], "breath")   # 图片保留极轻微呼吸

    def test_subtle_camera_only_gentle(self):
        from autocut.moves import GENTLE_MOVES
        feats = [dict(energy=0.9, brightness=0.5, colorfulness=0.5,
                      subject_center=0.3, motion=0.7, motion_dir=0.6,
                      shot_type="wide", kind="video", boundary=True,
                      section="high")] * 6
        plan = self._plan(feats, camera="subtle")
        self.assertTrue(all(p["move"] in GENTLE_MOVES for p in plan),
                        [p["move"] for p in plan])

    def test_high_tier_gets_shorter_transition(self):
        f_high = dict(energy=0.95, brightness=0.5, colorfulness=0.5,
                      subject_center=0.3, motion=0.5, motion_dir=0.0,
                      shot_type="medium", kind="video", boundary=False,
                      section="high")
        f_low = dict(f_high, energy=0.1, section="low")
        p_high = self._plan([f_high])[0]
        p_low = self._plan([f_low])[0]
        self.assertLess(p_high["tdur"], p_low["tdur"])


class TestFilterChains(unittest.TestCase):
    def test_every_fx_chain_builds(self):
        """所有特效都必须能生成滤镜链（曾因 f-string 占位符写错而崩溃）。"""
        names = sorted({n for pool in FX_POOL.values() for n in pool})
        for n in names:
            with self.subTest(fx=n):
                chain = fx_chain(n, 1.0, 1920, 1080, 128.0)
                self.assertIsInstance(chain, str)
                self.assertNotIn("NameError", chain)

    def test_structural_fx_has_placeholder(self):
        for n, expect in [("split_mirror", True), ("pip_card", True),
                          ("grain", False)]:
            with self.subTest(fx=n):
                chain = fx_chain(n, 1.0, 1920, 1080, 128.0)
                self.assertEqual("{I}" in chain, expect)
                # 占位符替换后不应残留未替换的名字
                self.assertNotIn("{I}", chain.replace("{I}", "x"))

    def test_every_camera_chain_builds(self):
        for n in sorted({m for pool in CAMERA_POOL.values() for m in pool}):
            with self.subTest(move=n):
                chain = camera_chain(n, 1920, 1080, 1.5, 128.0, 1.0, 30)
                self.assertIsInstance(chain, str)
        self.assertEqual(camera_chain("static", 1920, 1080, 1.0, 120, 1.0, 30), "")

    def test_pools_nonempty(self):
        for style, pool in CAMERA_POOL.items():
            self.assertTrue(pool, style)
        for style, pool in FX_POOL.items():
            self.assertTrue(pool, style)
        for tier, pool in TRANS_TIERS.items():
            self.assertTrue(pool, tier)
        self.assertTrue(GRADE_POOL["energetic"] and GRADE_LOW_SAT)


class TestTimelineMath(unittest.TestCase):
    def test_total_duration_uses_per_boundary_transitions(self):
        a = make_asset()
        segs = [Segment(asset=a, src_in=0.0, duration=2.0, transition="fade",
                        grade="", tdur=0.4),
                Segment(asset=a, src_in=2.0, duration=2.0, transition="fade",
                        grade="", tdur=0.05),
                Segment(asset=a, src_in=4.0, duration=2.0, transition="fade",
                        grade="")]
        tl = Timeline(segments=segs, bgm=make_bgm(), bgm_start=0.0, bgm_end=6.0,
                      tempo_ratio=1.0, trans_dur=0.5, mood="energetic")
        self.assertEqual(tl.boundary_durs, [0.4, 0.05])
        self.assertAlmostEqual(tl.total_duration, 6.0 - 0.45, places=6)

    def test_default_transition_used_when_tdur_missing(self):
        a = make_asset()
        segs = [Segment(asset=a, src_in=0.0, duration=1.0, transition="fade", grade=""),
                Segment(asset=a, src_in=1.0, duration=1.0, transition="fade", grade="")]
        tl = Timeline(segments=segs, bgm=make_bgm(), bgm_start=0.0, bgm_end=2.0,
                      tempo_ratio=1.0, trans_dur=0.3, mood="chill")
        self.assertEqual(tl.boundary_durs, [0.3])


class TestBgmMatching(unittest.TestCase):
    def test_mood_and_bpm_win(self):
        good = make_bgm(bpm=140, mood="energetic", energy=0.75)
        bad = make_bgm(bpm=80, mood="calm", energy=0.3)
        profile = material_profile([make_asset(energy=0.8)])
        self.assertIs(pick_bgm([bad, good], "energetic", profile=profile), good)

    def test_section_score_prefers_track_with_drop(self):
        with_drop = make_bgm(mood="energetic", sections=[(0, 5, "low"), (5, 20, "high")])
        flat = make_bgm(mood="energetic", sections=[(0, 30, "low")])
        profile = material_profile([make_asset(energy=0.75)])
        self.assertIs(pick_bgm([flat, with_drop], "energetic", profile=profile),
                      with_drop)

    def test_auto_style_mapping(self):
        self.assertEqual(auto_style({"energy": 0.9, "wide_ratio": 0.6,
                                     "close_ratio": 0.1, "motion": 0.7,
                                     "mood": "energetic"}), "dynamic")
        self.assertEqual(auto_style({"energy": 0.7, "wide_ratio": 0.1,
                                     "close_ratio": 0.8, "motion": 0.5,
                                     "mood": "energetic"}), "balanced")
        self.assertEqual(auto_style({"energy": 0.2, "wide_ratio": 0.2,
                                     "close_ratio": 0.2, "motion": 0.2,
                                     "mood": "calm"}), "cinematic")

    def test_resolve_style_validates(self):
        self.assertEqual(resolve_style("dynamic", "calm"), "dynamic")
        self.assertEqual(resolve_style("auto", "energetic"), "dynamic")
        self.assertEqual(resolve_style("auto", "calm"), "cinematic")


class TestEdlExport(unittest.TestCase):
    def test_export_roundtrip(self):
        shot = make_shot(subject_center=0.7, shot_type="close")
        a = make_asset(shots=[shot])
        segs = [Segment(asset=a, src_in=0.0, duration=1.5, transition="fade",
                        grade="eq=saturation=1.1", move="static", fx="beat_pulse",
                        tdur=0.2, why="特写不用破坏性特效"),
                Segment(asset=a, src_in=1.5, duration=1.5, transition="fade",
                        grade="", move="static", fx="none", ramp=(0.6, 1.7))]
        tl = Timeline(segments=segs, bgm=make_bgm(), bgm_start=1.0, bgm_end=5.0,
                      tempo_ratio=0.98, trans_dur=0.3, mood="energetic",
                      style="dynamic", amp=1.0, bgm_sections=[(0, 5, "high")])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "a.edl.json"
            export_edl(tl, out, 1920, 1080, 30)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["output"]["width"], 1920)
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["segments"][0]["fx"], "beat_pulse")
        self.assertEqual(data["segments"][1]["speed_ramp"], [0.6, 1.7])
        self.assertIn("why", data["segments"][0])
        self.assertEqual(data["bgm"]["clip_start"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
