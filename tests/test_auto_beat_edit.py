"""auto_beat_edit 的可复现测试（P2 第七项）。

全部使用合成的音频/视频，不依赖用户桌面的真实素材。
运行：python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest
import soundfile as sf
from moviepy import ColorClip, ImageSequenceClip, VideoFileClip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auto_beat_edit as abe  # noqa: E402


# ==========================================================================
# 合成素材
# ==========================================================================
def synth_click_track(path, sr=22050, total=12.0, bpm=120.0,
                      active=(0.0, 12.0), amp=0.9, seed=0):
    """生成带打击感点击声的音频（衰减噪声脉冲），active 之外为近似静音。"""
    n = int(sr * total)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.RandomState(seed)
    period = 60.0 / bpm
    click_len = max(1, int(0.02 * sr))
    env = np.exp(-np.arange(click_len) / (0.004 * sr)).astype(np.float32)
    t = active[0]
    while t < min(active[1], total):
        i = int(t * sr)
        seg = min(click_len, n - i)
        if seg > 0:
            y[i:i + seg] += amp * env[:seg] * rng.randn(seg).astype(np.float32)
        t += period
    y += 1e-6 * rng.randn(n).astype(np.float32)
    sf.write(path, y, sr)
    return path


@pytest.fixture
def click_audio(tmp_path):
    path = tmp_path / "click_120bpm.wav"
    synth_click_track(str(path), total=12.0, bpm=120.0, active=(0.0, 12.0))
    return str(path)


@pytest.fixture
def tiny_videos(tmp_path):
    """三个 2 秒的小视频，命名故意用 1 / 2 / 10 以验证自然排序。"""
    files = []
    for name, color in [("1.mp4", (200, 40, 40)), ("2.mp4", (40, 200, 40)), ("10.mp4", (40, 40, 200))]:
        p = tmp_path / name
        clip = ColorClip(size=(64, 80), color=color, duration=2.0)
        clip.write_videofile(str(p), fps=12, codec="libx264", audio=False, logger=None)
        clip.close()
        files.append(str(p))
    return str(tmp_path), files


def make_pattern_video(path, size=(64, 80), fps=12, n=14):
    """高对比 + 逐帧移动的素材（清晰度高、运动量大）。"""
    rng = np.random.RandomState(3)
    base = rng.randint(0, 255, (size[1], size[0], 3)).astype(np.uint8)
    frames = [np.roll(base, (i * 7) % size[0], axis=1) for i in range(n)]
    c = ImageSequenceClip(frames, fps=fps)
    c.write_videofile(str(path), fps=fps, codec="libx264", audio=False, logger=None)
    c.close()


def make_flat_video(path, size=(64, 80), fps=12, duration=2.0):
    """纯色素材（清晰度≈0、运动≈0）。"""
    c = ColorClip(size=size, color=(128, 128, 128), duration=duration)
    c.write_videofile(str(path), fps=fps, codec="libx264", audio=False, logger=None)
    c.close()


# ==========================================================================
# 时间换算 / 变速语义（P0-2）
# ==========================================================================
def test_speed_duration_relationship_matches_moviepy():
    """documents: with_speed_scaled(f) => duration = source / f"""
    base = ColorClip(size=(8, 8), color=(1, 2, 3), duration=2.0)
    assert abs(base.with_speed_scaled(0.5).duration - 4.0) < 1e-6
    assert abs(base.with_speed_scaled(1.0).duration - 2.0) < 1e-6
    assert abs(base.with_speed_scaled(2.0).duration - 1.0) < 1e-6


def test_speed_for_and_durations():
    assert abs(abe.speed_for(1.2, 1.0) - 1.2) < 1e-9
    assert abs(abe.source_duration_for(1.0, 1.06) - 1.06) < 1e-9
    assert abs(abe.timeline_duration_for(1.06, 1.06) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        abe.speed_for(1.0, 0.0)


def test_resolve_segment_timing_fills_target_exactly():
    src, spd, note = abe.resolve_segment_timing(available=5.0, target_duration=1.0, desired_speed=1.06)
    assert note == ""
    assert abs(abe.timeline_duration_for(src, spd) - 1.0) < 1e-9


def test_resolve_segment_timing_clamps_to_max_speed():
    src, spd, note = abe.resolve_segment_timing(available=5.0, target_duration=1.0, desired_speed=9.9)
    assert spd <= abe.MAX_SPEED + 1e-9
    assert abs(abe.timeline_duration_for(src, spd) - 1.0) < 1e-9


def test_resolve_segment_timing_insufficient_footage():
    src, spd, note = abe.resolve_segment_timing(available=0.1, target_duration=1.0, desired_speed=1.06)
    assert note == "footage_short"
    assert src <= 0.1 + 1e-9
    assert spd == abe.MIN_SPEED


def test_clamp_speed():
    assert abe.clamp_speed(9.0) == abe.MAX_SPEED
    assert abe.clamp_speed(0.1) == abe.MIN_SPEED
    assert abe.clamp_speed(1.0) == 1.0


# ==========================================================================
# 音频窗口校验（P0-4）
# ==========================================================================
def test_validate_audio_window_out_of_range(click_audio):
    with pytest.raises(ValueError):
        abe.validate_audio_window(click_audio, start=99.0, duration=2.0)
    with pytest.raises(ValueError):
        abe.validate_audio_window(click_audio, start=-1.0, duration=2.0)
    with pytest.raises(ValueError):
        abe.validate_audio_window(click_audio, start=0.0, duration=0.0)


def test_validate_audio_window_shortens_at_tail(click_audio):
    start, end = abe.validate_audio_window(click_audio, start=10.0, duration=5.0)
    assert start == 10.0
    assert end <= 12.0 + 1e-6


# ==========================================================================
# 拍点检测与回退（P0-5）
# ==========================================================================
def test_detect_beats_on_click_track(click_audio):
    grid = abe.detect_beats(click_audio, 0, None)
    assert grid.status == "ok"
    assert abs(grid.tempo - 120.0) < 8.0
    assert grid.beats.size >= 15
    assert grid.half_beats.size >= grid.beats.size


def test_detect_beats_silent_audio_no_crash(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(22050 * 5, dtype=np.float32), 22050)
    grid = abe.detect_beats(str(path), 0, None)
    assert grid.status != "ok"
    assert grid.beats.size > 0          # 回退合成网格
    assert grid.half_beats.size > 0


def test_build_half_beats_boundaries():
    assert abe.build_half_beats(np.array([])).size == 0
    assert abe.build_half_beats(np.array([1.0])).size == 1
    hb = abe.build_half_beats(np.array([0.0, 1.0, 2.0]))
    assert np.allclose(hb, [0.0, 0.5, 1.0, 1.5, 2.0])


# ==========================================================================
# 自动选段（P1 第一项）
# ==========================================================================
def test_select_audio_segment_prefers_active_section(tmp_path):
    path = tmp_path / "half_quiet.wav"
    # 0~10s 近静音，10~18s 有稳定节拍
    synth_click_track(str(path), total=20.0, bpm=128.0, active=(10.0, 18.0))
    result = abe.select_audio_segment(str(path), duration=5.0, verbose=False)
    assert result["start"] >= 8.0, result
    assert result["score"] > 0.0
    assert len(result["reasons"]) >= 3
    assert result["candidates"]


def test_select_audio_segment_short_audio(tmp_path):
    path = tmp_path / "short.wav"
    synth_click_track(str(path), total=3.0, bpm=120.0, active=(0.0, 3.0))
    result = abe.select_audio_segment(str(path), duration=10.0, verbose=False)
    assert result["start"] == 0.0
    assert result["duration"] == pytest.approx(3.0, abs=1e-3)


# ==========================================================================
# 素材扫描 / 自然排序 / 报错（P0-3 / P1 第三项）
# ==========================================================================
def test_natural_sort_key_orders_numbers():
    names = ["10.mp4", "2.mp4", "1.mp4", "20.mp4"]
    assert sorted(names, key=abe.natural_sort_key) == ["1.mp4", "2.mp4", "10.mp4", "20.mp4"]


def test_collect_video_files_auto_scan_natural_order(tmp_path):
    for n in ["10.mp4", "1.mp4", "2.mp4"]:
        (tmp_path / n).write_bytes(b"")
    found, missing = abe.collect_video_files(str(tmp_path), clip_order=None)
    assert [os.path.basename(f) for f in found] == ["1.mp4", "2.mp4", "10.mp4"]
    assert missing == []


def test_collect_video_files_reports_missing_names(tmp_path):
    (tmp_path / "1.mp4").write_bytes(b"")
    (tmp_path / "2.mp4").write_bytes(b"")
    found, missing = abe.collect_video_files(str(tmp_path), clip_order=["1.mp4", "2.mp4", "9.mp4"])
    assert [os.path.basename(f) for f in found] == ["1.mp4", "2.mp4"]
    assert missing == ["9.mp4"]


def test_prepare_video_clips_reports_broken(tmp_path, tiny_videos):
    folder, files = tiny_videos
    bad = os.path.join(folder, "broken.mp4")
    with open(bad, "wb") as fp:
        fp.write(b"not a video")
    clips, names, errors = abe.prepare_video_clips([files[0], bad], target_resolution=None)
    assert names == ["1.mp4"]
    assert errors and errors[0][0] == "broken.mp4"
    for c in clips:
        c.close()


# ==========================================================================
# 网格吸附 / 分段计划
# ==========================================================================
def test_snap_to_grid_respects_min_gap():
    grid = np.array([0.0, 0.5, 1.0, 1.5])
    assert abe.snap_to_grid(0.4, grid, min_gap=0.0) == pytest.approx(0.5)
    assert abe.snap_to_grid(0.4, grid, min_gap=0.6) == pytest.approx(1.0)
    assert abe.snap_to_grid(0.4, np.array([]), fallback=7.0) == pytest.approx(7.0)


def test_assign_clips_avoids_immediate_repeat_and_balances():
    seq = abe.assign_clips(7, 3)
    assert len(seq) == 7
    assert all(a != b for a, b in zip(seq, seq[1:]))
    counts = [seq.count(i) for i in range(3)]
    assert max(counts) - min(counts) <= 2


def test_assign_clips_single_clip_ok():
    assert abe.assign_clips(5, 1) == [0] * 5
    with pytest.raises(ValueError):
        abe.assign_clips(5, 0)


def test_assign_clips_skips_clip_with_insufficient_footage():
    """长片段不能被分给过短的素材，否则会 footage_short、成片总时长漂移。"""
    # 可用时长 1.343 / 3.097 / 4.645 / 3.205；首段需要 2.11s
    demands = [2.11, 0.35, 0.35, 0.35]
    capacities = [1.343, 3.097, 4.645, 3.205]
    seq = abe.assign_clips(4, 4, demands=demands, capacities=capacities)
    assert capacities[seq[0]] >= demands[0] * abe.MIN_SPEED - 1e-9, seq
    assert seq[0] != 0, seq          # 1.343s 的素材填不满 2.11s
    # 未提供约束时行为不变（纯权重轮转）
    assert abe.assign_clips(4, 4) == abe.assign_clips(4, 4, demands=None, capacities=None)


def test_plan_segments_single_clip_no_indexerror(tiny_videos):
    """原版在素材少于 4 个时会 IndexError，这里必须安全。"""
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips([files[0]], target_resolution=None)
    grid = abe.BeatGrid(beats=np.arange(0.0, 6.0, 0.5),
                        half_beats=np.arange(0.0, 6.0, 0.25), tempo=120.0)
    boundaries = abe.build_boundaries(grid, 3.0, abe.PATTERN_PRESETS["normal"], verbose=False)
    assign = abe.assign_clips(len(boundaries) - 1, len(clips))
    segs, issues = abe.plan_segments(boundaries, assign, clips, grid.beat_duration,
                                     clip_names=names)
    assert segs
    for s in segs:
        assert s.source_out <= clips[0].duration + 1e-6
        assert s.clip_name == "1.mp4"
    for c in clips:
        c.close()


def test_plan_segments_reuse_or_error(tiny_videos):
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips([files[0]], target_resolution=None)
    grid = abe.BeatGrid(beats=np.arange(0.0, 6.0, 0.5),
                        half_beats=np.arange(0.0, 6.0, 0.25), tempo=120.0)
    # 目标 8s 远大于素材 2s，禁止复用时必须报错
    boundaries = [0.0, 2.0, 4.0, 6.0, 8.0]
    assign = abe.assign_clips(4, 1)
    with pytest.raises(RuntimeError):
        abe.plan_segments(boundaries, assign, clips, grid.beat_duration,
                          clip_names=names, allow_reuse=False)
    segs, _ = abe.plan_segments(boundaries, assign, clips, grid.beat_duration,
                                clip_names=names, allow_reuse=True)
    assert any(s.reused for s in segs)
    for c in clips:
        c.close()


# ==========================================================================
# EDL 与校验（P2 第六项）
# ==========================================================================
def test_build_boundaries_degrades_half_beat_when_below_min_shot():
    """半拍短于 MIN_SHOT 时必须降级为整拍，切点全部落在整拍网格上。

    240BPM 半拍 = 0.125s < MIN_SHOT 0.15s（否则会出现低于最短镜头约束的切点）。
    """
    bpm = 240.0
    bd = 60.0 / bpm
    beats = np.arange(0.0, 9.6, bd)                       # 末尾刻意早于窗口结束
    grid = abe.BeatGrid(beats=beats, half_beats=abe.build_half_beats(beats), tempo=bpm)
    labels = []
    boundaries = abe.build_boundaries(grid, 10.0, abe.PATTERN_PRESETS["normal"],
                                      verbose=False, label_sink=labels)

    durs = np.diff(boundaries)
    assert (durs >= abe.MIN_SHOT - 1e-6).all(), durs
    assert len(labels) == len(boundaries) - 1
    # 降级后不应出现半拍级别（0.125s）的切点
    assert (durs < bd - 1e-6).sum() == 0, durs
    # 标签必须真实：长片段的标签不应是「闪 / 快切」
    for d, lb in zip(durs, labels):
        if d > 1.0:
            assert "闪" not in lb and "快切" not in lb, (d, lb)
    # 网格补齐后总时长应贴近目标（容差 0.2s）
    assert abs(boundaries[-1] - 10.0) < 0.2, boundaries[-1]
    # 每个切点都落在拍点网格上（含按 tempo 补齐的末端网格）
    full_grid = np.concatenate([beats, np.arange(beats[-1] + bd, 10.0 + bd, bd)])
    for b in boundaries[1:]:
        assert np.min(np.abs(full_grid - b)) < 0.06, b


def test_build_boundaries_uses_half_beats_at_172bpm_without_collapsing_longs():
    """172.3BPM 半拍 ≈ 0.174s ≥ MIN_SHOT，必须真正使用半拍快切，且长镜头不塌缩。

    复现真实素材问题：MIN_SHOT=0.25 时半拍被判过短、整体降级为整拍，快切段被
    拉长、结尾长镜头被挤到只剩 1 拍（实测「结尾长」由 1.59s 塌缩为 0.35s）。
    """
    bpm = 172.3
    bd = 60.0 / bpm
    beats = np.arange(0.0, 9.6, bd)
    grid = abe.BeatGrid(beats=beats, half_beats=abe.build_half_beats(beats), tempo=bpm)
    labels = []
    boundaries = abe.build_boundaries(grid, 10.0, abe.PATTERN_PRESETS["normal"],
                                      verbose=False, label_sink=labels)

    durs = np.diff(boundaries)
    assert (durs >= abe.MIN_SHOT - 1e-6).all(), durs
    # 必须存在半拍级别的快切（而不是整体降级为整拍）
    assert (durs < bd - 1e-6).sum() >= 5, durs
    # 「结尾长」不能被挤塌缩成 1 拍
    long_tail = [d for d, lb in zip(durs, labels) if lb == "结尾长"]
    assert long_tail and max(long_tail) > 1.0, list(zip([round(d, 3) for d in durs], labels))
    assert abs(boundaries[-1] - 10.0) < 0.2, boundaries[-1]


def test_build_boundaries_keeps_half_beats_at_moderate_tempo():
    """120BPM 半拍=0.25s 恰好满足 MIN_SHOT，不应降级。"""
    grid = abe.BeatGrid(beats=np.arange(0.0, 10.0, 0.5),
                        half_beats=np.arange(0.0, 10.0, 0.25), tempo=120.0)
    boundaries = abe.build_boundaries(grid, 8.0, abe.PATTERN_PRESETS["normal"], verbose=False)
    durs = np.diff(boundaries)
    assert (durs >= abe.MIN_SHOT - 1e-6).all()
    assert (durs < 0.5 - 1e-6).any()          # 存在半拍（0.25s）级别的快切


def test_edl_timeline_is_contiguous_and_valid(tiny_videos):
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips(files, target_resolution=None)
    grid = abe.BeatGrid(beats=np.arange(0.0, 6.0, 0.5),
                        half_beats=np.arange(0.0, 6.0, 0.25), tempo=120.0)
    boundaries = abe.build_boundaries(grid, 3.0, abe.PATTERN_PRESETS["normal"], verbose=False)
    assign = abe.assign_clips(len(boundaries) - 1, len(clips))
    segs, _ = abe.plan_segments(boundaries, assign, clips, grid.beat_duration, clip_names=names)
    abe.apply_labels(segs, [p[1] for p in abe.PATTERN_PRESETS["normal"]])
    durations = {n: c.duration for n, c in zip(names, clips)}
    edl = abe.build_edl(segs, {"start": 0.0, "duration": 3.0}, durations,
                        target_resolution=(64, 80), target_duration=3.0)
    problems, critical = abe.validate_edl(edl)
    assert critical == []
    assert problems == []
    # 时间轴连续
    for a, b in zip(edl["segments"], edl["segments"][1:]):
        assert abs(a["timeline_end"] - b["timeline_start"]) < 1e-6
    for c in clips:
        c.close()


def test_validate_edl_flags_source_out_of_range():
    seg = abe.EditSegment(index=0, clip_index=0, clip_name="a.mp4", label="x",
                          timeline_start=0.0, timeline_end=1.0, duration=1.0,
                          source_in=0.0, source_out=5.0, speed=1.0, is_fast=False)
    edl = abe.build_edl([seg], {"start": 0.0, "duration": 1.0}, {"a.mp4": 2.0})
    problems, critical = abe.validate_edl(edl)
    assert critical and "超出素材" in critical[0]


def test_save_and_reload_edl(tmp_path):
    seg = abe.EditSegment(index=0, clip_index=0, clip_name="a.mp4", label="x",
                          timeline_start=0.0, timeline_end=1.0, duration=1.0,
                          source_in=0.0, source_out=1.0, speed=1.0, is_fast=True)
    edl = abe.build_edl([seg], {"start": 1.0, "duration": 1.0}, {"a.mp4": 2.0})
    path = abe.save_edl(edl, str(tmp_path / "plan.json"))
    import json
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    assert data["segments"][0]["clip_name"] == "a.mp4"
    assert data["segments"][0]["is_fast"] is True


# ==========================================================================
# 端到端：构建计划 -> 渲染 -> 导出（小分辨率）
# ==========================================================================
def test_end_to_end_render(tmp_path, monkeypatch, tiny_videos):
    folder, files = tiny_videos
    audio = os.path.join(folder, "track.wav")
    synth_click_track(audio, total=12.0, bpm=120.0, active=(0.0, 12.0))

    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))

    plan, clips, grid, status = abe.build_edit_plan(
        folder=folder, audio_file="track.wav", duration=3.0,
        audio_start=1.0, auto_select=False, clip_order=["1.mp4", "2.mp4", "10.mp4"],
        target_resolution=(96, 128), verbose=False,
    )
    assert plan["edl"]["segments"]
    assert plan["edl"]["audio"]["start"] == 1.0

    final = abe.render_segments(plan["segments"], clips, grid)
    assert final.duration > 1.0

    # 复现 main() 的加音乐 -> 导出路径
    from moviepy import AudioFileClip
    full_audio = AudioFileClip(audio)
    audio_end = min(plan["edl"]["audio"]["start"] + final.duration, full_audio.duration)
    bgm = full_audio.subclipped(plan["edl"]["audio"]["start"], audio_end)
    if final.duration > bgm.duration:
        final = final.subclipped(0, bgm.duration)
    final = final.with_audio(bgm)

    out = str(tmp_path / "out.mp4")
    final.write_videofile(out, fps=12, codec="libx264", audio_codec="aac",
                          logger=None)
    final.close()
    full_audio.close()
    for c in clips:
        c.close()
    assert os.path.getsize(out) > 1000
    from moviepy import VideoFileClip
    probe = VideoFileClip(out)
    assert probe.audio is not None, "导出文件应包含音轨"
    probe.close()


def test_build_edit_plan_missing_audio_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        abe.build_edit_plan(folder=str(folder), audio_file="none.mp3",
                            duration=2.0, verbose=False)


# ==========================================================================
# 素材质量评分（第五项）
# ==========================================================================
def test_analyze_assets_quality_metrics(tmp_path):
    sharp = str(tmp_path / "sharp.mp4")
    flat = str(tmp_path / "flat.mp4")
    make_pattern_video(sharp)
    make_flat_video(flat)

    infos = abe.analyze_assets([sharp, flat], target_duration=2.0)
    by = {i.name: i for i in infos}
    assert by["flat.mp4"].sharpness_raw == pytest.approx(0.0, abs=25.0)
    assert by["sharp.mp4"].sharpness_raw > by["flat.mp4"].sharpness_raw + 100.0
    # 归一化后 sharp 应为 1.0，flat 为 0.0
    assert by["sharp.mp4"].sharpness == pytest.approx(1.0, abs=1e-6)
    assert by["flat.mp4"].sharpness == pytest.approx(0.0, abs=1e-6)
    assert by["sharp.mp4"].motion > by["flat.mp4"].motion
    assert 0.0 <= by["flat.mp4"].exposure_score <= 1.0
    assert by["sharp.mp4"].quality_score > by["flat.mp4"].quality_score
    for i in infos:
        assert 0.0 <= i.quality_score <= 1.0
        assert 0.0 <= i.candidate_score <= 1.0
        assert 0.0 <= i.usable_ratio <= 1.0


def test_quality_weights_aligns_to_clip_names(tmp_path):
    sharp = str(tmp_path / "a.mp4")
    flat = str(tmp_path / "b.mp4")
    make_pattern_video(sharp)
    make_flat_video(flat)
    infos = abe.analyze_assets([sharp, flat], target_duration=2.0)
    weights = abe.quality_weights(infos, ["b.mp4", "a.mp4", "missing.mp4"])
    assert weights is not None and len(weights) == 3
    assert weights[1] > weights[0]          # a.mp4 质量更高 -> 权重更大
    assert weights[2] == pytest.approx(1.0)  # 缺失素材兜底
    assert abe.quality_weights(infos, None, prefer_quality=False) is None


def test_assign_clips_weighted_bias():
    seq = abe.assign_clips(20, 3, weights=[3.0, 1.0, 1.0])
    counts = [seq.count(i) for i in range(3)]
    assert counts[0] == max(counts)
    assert counts[0] > counts[1]
    assert all(a != b for a, b in zip(seq, seq[1:]))
    with pytest.raises(ValueError):
        abe.assign_clips(5, 2, weights=[1.0])


# ==========================================================================
# 转场语义（cut / dip）
# ==========================================================================
def test_plan_segments_default_transition_is_cut(tiny_videos):
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips(files, target_resolution=None)
    grid = abe.BeatGrid(beats=np.arange(0.0, 6.0, 0.5),
                        half_beats=np.arange(0.0, 6.0, 0.25), tempo=120.0)
    boundaries = abe.build_boundaries(grid, 3.0, abe.PATTERN_PRESETS["normal"], verbose=False)
    assign = abe.assign_clips(len(boundaries) - 1, len(clips))
    segs, _ = abe.plan_segments(boundaries, assign, clips, grid.beat_duration, clip_names=names)
    assert all(s.transition == "cut" for s in segs)
    for c in clips:
        c.close()


def test_plan_segments_honours_short_segment_strategy(tiny_videos, monkeypatch):
    """allow_reuse 未显式给出时，应由 SHORT_SEGMENT_STRATEGY 统一驱动。"""
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips([files[0]], target_resolution=None)
    boundaries = [0.0, 2.0, 4.0, 6.0, 8.0]
    assign = abe.assign_clips(4, 1)

    monkeypatch.setattr(abe, "SHORT_SEGMENT_STRATEGY", "error")
    with pytest.raises(RuntimeError):
        abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names)

    monkeypatch.setattr(abe, "SHORT_SEGMENT_STRATEGY", "loop")
    segs, _ = abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names)
    assert any(s.reused for s in segs)

    monkeypatch.setattr(abe, "SHORT_SEGMENT_STRATEGY", "bogus")
    with pytest.raises(ValueError):
        abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names)

    for c in clips:
        c.close()


def test_parse_args_plan_only_and_auto_start():
    args = abe.parse_args(["--plan-only", "--duration", "6.0",
                           "--audio-start", "-1", "--strategy", "slow"])
    assert args.plan_only is True
    assert args.duration == pytest.approx(6.0)
    assert args.audio_start == pytest.approx(-1.0)
    assert args.strategy == "slow"

    default = abe.parse_args([])
    assert default.plan_only is False
    with pytest.raises(SystemExit):
        abe.parse_args(["--strategy", "nope"])


# ==========================================================================
# 特效完整性 / 性能回归（帧尺寸必须保持不变）
# ==========================================================================
def test_pixelate_preserves_shape_for_all_blocks():
    """旧实现在 h/w 不能被 block 整除时会缩小帧（导致黑边），必须修复。"""
    img = np.random.RandomState(0).randint(0, 255, (128, 96, 3)).astype(np.uint8)
    for block in range(2, 20):
        out = abe.pixelate(img, block)
        assert out.shape == img.shape, (block, out.shape)
        assert out.dtype == np.uint8
    # 分块正确：同一块内颜色相同（block=8 时 128/96 均可整除）
    out = abe.pixelate(img, 8)
    assert np.array_equal(out[0:8, 0:8], np.broadcast_to(out[0, 0], (8, 8, 3)))


def test_fisheye_and_flare_preserve_shape():
    img = np.random.RandomState(1).randint(0, 255, (128, 96, 3)).astype(np.uint8)
    assert abe.fisheye(img, 0.12).shape == img.shape
    assert abe.fisheye(img, 0.0).shape == img.shape
    assert abe.add_flare(img, 0.5).shape == img.shape


def test_add_effects_preserves_frame_size_and_dtype(monkeypatch):
    """整条特效链必须保持帧尺寸与 uint8，且分析头/慢两种路径都不崩。"""
    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))
    clip = ColorClip(size=(96, 128), color=(120, 60, 200), duration=1.0)
    beats = np.arange(0.0, 2.0, 0.5)
    # 首个采样点上 pulse_amount≈1.0，会触发鱼眼 + 光晕
    for is_fast in (True, False):
        for idx in range(4):
            fx = abe.add_effects(clip, 0.0, beats, idx, is_fast, 0.5)
            frame = fx.get_frame(0.0)
            assert frame.shape == (128, 96, 3), frame.shape
            assert frame.dtype == np.uint8
            fx.close()
    clip.close()


# ==========================================================================
# 重拍/弱拍强度分级（P1 第二项「区分不同强度节拍」）
# ==========================================================================
def test_normalize_strengths_uniform_stays_neutral():
    """强弱差异过小时统一记 0.5（中等），避免把噪声放大成虚假重拍。"""
    assert np.allclose(abe._normalize_strengths(np.full(8, 3.7)), 0.5)
    assert np.allclose(abe._normalize_strengths(np.array([])), np.array([]))


def test_normalize_strengths_spreads_variation_to_unit_range():
    raw = np.array([10.0, 10.1, 10.2, 40.0, 90.0, 95.0])
    s = abe._normalize_strengths(raw)
    assert s.min() == pytest.approx(0.0) and s.max() == pytest.approx(1.0)
    assert (np.diff(s[np.argsort(raw)]) >= -1e-9).all()      # 单调


def test_beatgrid_strengths_default_to_neutral():
    grid = abe.BeatGrid(beats=np.arange(4.0), half_beats=np.arange(4.0), tempo=120.0)
    assert np.allclose(grid.strengths, 0.5)
    # 长度不匹配时也不能错位，退回中性
    bad = abe.BeatGrid(beats=np.arange(4.0), half_beats=np.arange(4.0), tempo=120.0,
                       strengths=np.array([1.0, 2.0]))
    assert np.allclose(bad.strengths, 0.5)


def test_strength_scale_neutral_reproduces_old_behaviour():
    assert abe.strength_scale(0.5) == pytest.approx(1.0)
    assert abe.strength_scale(1.0) == pytest.approx(1.0 + abe.STRENGTH_CONTRAST)
    assert abe.strength_scale(0.0) == pytest.approx(1.0 - abe.STRENGTH_CONTRAST)
    assert abe.strength_scale(5.0) == abe.strength_scale(1.0)   # 越界钳制


def test_strength_at_picks_nearest_beat():
    grid = abe.BeatGrid(beats=np.array([0.0, 1.0, 2.0]),
                        half_beats=np.array([0.5, 1.5]), tempo=60.0,
                        strengths=np.array([0.1, 0.9, 0.4]))
    # 取最近拍点的强度（值本身会被归一化，故与 grid.strengths 对齐比较）
    assert abe.strength_at(grid, 0.9) == pytest.approx(grid.strengths[1])
    assert abe.strength_at(grid, 2.4) == pytest.approx(grid.strengths[2])
    assert grid.strengths[1] > grid.strengths[2]        # 原始强弱关系保留
    empty = abe.BeatGrid(beats=np.array([]), half_beats=np.array([]), tempo=120.0)
    assert abe.strength_at(empty, 1.0) == 1.0


def test_detect_beats_returns_strengths_aligned(click_audio):
    grid = abe.detect_beats(click_audio, 0, None)
    assert grid.strengths.size == grid.beats.size
    assert grid.strengths.min() >= 0.0 and grid.strengths.max() <= 1.0


def test_plan_segments_records_beat_strength(tiny_videos):
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips(files, None)
    boundaries = [0.0, 0.5, 1.0, 1.5]
    assign = [0, 1, 2]
    segs, _ = abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names,
                                seg_strengths=[0.2, 0.9, 0.5])
    assert [s.beat_strength for s in segs] == pytest.approx([0.2, 0.9, 0.5])
    # 未提供时不报错，按中性 1.0 处理
    segs2, _ = abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names)
    assert all(s.beat_strength == 1.0 for s in segs2)
    for c in clips:
        c.close()


def _noise_clip(path, size=(64, 80), fps=12, n=14):
    rng = np.random.RandomState(5)
    frames = [rng.randint(0, 255, (size[1], size[0], 3)).astype(np.uint8) for _ in range(n)]
    c = ImageSequenceClip(frames, fps=fps)
    c.write_videofile(str(path), fps=fps, codec="libx264", audio=False, logger=None)
    c.close()


def test_add_effects_scales_with_beat_strength(tmp_path, monkeypatch):
    """重拍帧必须比弱拍帧「更冲」：同一帧下不同强度应产生不同像素结果。"""
    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))
    p = tmp_path / "noise.mp4"
    _noise_clip(str(p))
    clip = VideoFileClip(str(p))
    beats = np.array([0.0, 0.5, 1.0])
    weak = abe.add_effects(clip, 0.0, beats, 0, True, 0.5,
                           np.array([0.0, 0.5, 0.5])).get_frame(0.0)
    neutral = abe.add_effects(clip, 0.0, beats, 0, True, 0.5,
                              np.array([0.5, 0.5, 0.5])).get_frame(0.0)
    strong = abe.add_effects(clip, 0.0, beats, 0, True, 0.5,
                             np.array([1.0, 0.5, 0.5])).get_frame(0.0)
    for fr in (weak, neutral, strong):
        assert fr.shape == (128, 96, 3) and fr.dtype == np.uint8
    assert np.abs(strong.astype(int) - weak.astype(int)).mean() > 3.0
    # 中性强度应与不传强度（旧行为）一致
    legacy = abe.add_effects(clip, 0.0, beats, 0, True, 0.5).get_frame(0.0)
    assert np.abs(neutral.astype(int) - legacy.astype(int)).mean() == pytest.approx(0.0, abs=1e-6)
    clip.close()


def test_main_plan_only_writes_edl_without_rendering(tmp_path, monkeypatch, tiny_videos):
    folder, files = tiny_videos
    audio = os.path.join(folder, "track.wav")
    synth_click_track(audio, total=12.0, bpm=120.0, active=(0.0, 12.0))
    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))

    edl_out = str(tmp_path / "plan.json")
    code = abe.main(["--plan-only", "--folder", folder, "--audio", "track.wav",
                     "--duration", "3.0", "--audio-start", "1.0", "--edl", edl_out])
    assert code == 0
    assert os.path.exists(edl_out)


def test_dip_transition_is_timing_safe(tmp_path, monkeypatch, tiny_videos):
    folder, files = tiny_videos
    audio = os.path.join(folder, "track.wav")
    synth_click_track(audio, total=12.0, bpm=120.0, active=(0.0, 12.0))

    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))
    monkeypatch.setattr(abe, "TRANSITION_MODE", "dip")

    plan, clips, grid, _ = abe.build_edit_plan(
        folder=folder, audio_file="track.wav", duration=3.0,
        audio_start=1.0, auto_select=False,
        clip_order=["1.mp4", "2.mp4", "10.mp4"],
        target_resolution=(96, 128), verbose=False,
    )
    segs = plan["segments"]
    # 非快切片段使用 dip，快切片段仍为 cut
    assert any(s.transition == "dip" for s in segs)
    assert all(s.transition == "cut" for s in segs if s.is_fast)
    # EDL 中记录转场类型且校验通过
    assert all(s["transition"] in ("cut", "dip") for s in plan["edl"]["segments"])
    problems, critical = abe.validate_edl(plan["edl"])
    assert critical == [] and problems == []

    final = abe.render_segments(segs, clips, grid)
    # dip 是段内淡入淡出，不改变任何片段时长 -> 总时长不变
    assert final.duration == pytest.approx(sum(s.duration for s in segs), abs=0.05)
    final.close()
    for c in clips:
        c.close()


# ==========================================================================
# 素材内窗口优选：让质量分析真正作用于「取素材的哪一段画面」
# ==========================================================================
def _fake_asset(name, dur, times, scores):
    a = abe.AssetInfo(name=name, duration=dur, width=64, height=80, aspect=0.8, has_audio=False)
    a.window_times = list(times)
    a.window_score = list(scores)
    return a


def test_window_profile_marks_black_head_as_low_score():
    """片头黑帧必须拿低分：清晰度/运动量/曝光都接近 0。"""
    frames = [np.zeros((32, 24, 3), dtype=np.uint8) for _ in range(6)]
    rng = np.random.RandomState(1)
    base = rng.randint(0, 255, (32, 24, 3)).astype(np.uint8)
    frames += [np.roll(base, i * 5, axis=1) for i in range(6)]
    clip = ImageSequenceClip(frames, fps=12)
    times, scores = abe._window_profile(clip, n=12)
    assert len(times) == len(scores) == 12
    half = len(scores) // 2
    assert max(scores[:half]) < max(scores[half:])
    clip.close()


def test_window_profile_empty_for_zero_duration():
    assert abe._window_profile(ColorClip(size=(8, 8), color=(0, 0, 0), duration=0.0)) == ([], [])


def test_choose_window_start_skips_edge_guard():
    a = _fake_asset("a.mp4", 4.0, [0.0, 1.5], [1.0, 0.3])
    start, score = abe.choose_window_start(a, 4.0, 1.0, [], 0.0)
    assert start == pytest.approx(1.5)          # 0.0 落在片头保护区内被排除
    assert score is not None


def test_choose_window_start_penalises_used_windows():
    a = _fake_asset("a.mp4", 4.0, [0.5, 1.5, 2.5], [0.9, 0.85, 0.2])
    first, _ = abe.choose_window_start(a, 4.0, 1.0, [], 0.0)
    assert first == pytest.approx(0.5)
    second, _ = abe.choose_window_start(a, 4.0, 1.0, [(0.5, 1.5)], 0.0)
    assert second == pytest.approx(1.5)         # 0.5 被重叠扣分挤掉


def test_choose_window_start_falls_back_when_nothing_fits():
    a = _fake_asset("a.mp4", 2.0, [0.1, 0.5], [0.9, 0.8])
    start, score = abe.choose_window_start(a, 2.0, 3.0, [], 0.7)
    assert start == pytest.approx(0.7) and score is None


def test_capacity_offset_reserves_window_guard(monkeypatch):
    clip = ColorClip(size=(8, 8), color=(0, 0, 0), duration=4.0)
    monkeypatch.setattr(abe, "WINDOW_SELECT", True)
    assert abe._capacity_offset(clip) == pytest.approx(abe.WINDOW_EDGE_GUARD)
    monkeypatch.setattr(abe, "WINDOW_SELECT", False)
    assert abe._capacity_offset(clip) == pytest.approx(abe._wrap_offset(clip))
    clip.close()


def test_plan_segments_prefers_high_scoring_window(tiny_videos):
    folder, files = tiny_videos
    clips, names, _ = abe.prepare_video_clips(files, None)
    boundaries = [0.0, 0.5, 1.0]
    assign = [0, 0]
    asset = _fake_asset(names[0], clips[0].duration, [0.3, 1.0, 1.6], [0.2, 0.95, 0.3])
    segs, _ = abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names,
                                assets=[asset], window_select=True)
    # 第一段应落到 score 最高的 1.0s，而不是顺序游标的 2% 处（≈0.04s）
    assert segs[0].source_in == pytest.approx(1.0)
    # 窗口优选只改源入点，不改时长：时间轴严格 0 -> 0.5 -> 1.0
    assert [s.timeline_end for s in segs] == pytest.approx([0.5, 1.0])
    # 关闭窗口优选时应退回原顺序游标
    segs2, _ = abe.plan_segments(boundaries, assign, clips, 0.5, clip_names=names,
                                 assets=[asset], window_select=False)
    assert segs2[0].source_in < 0.1
    for c in clips:
        c.close()


def test_build_edit_plan_window_select_keeps_edl_valid(tmp_path, monkeypatch, tiny_videos):
    folder, files = tiny_videos
    synth_click_track(os.path.join(folder, "track.wav"), total=12.0,
                      bpm=120.0, active=(0.0, 12.0))
    monkeypatch.setattr(abe, "TARGET_RESOLUTION", (96, 128))
    plan, clips, grid, _ = abe.build_edit_plan(
        folder=folder, audio_file="track.wav", duration=3.0,
        audio_start=1.0, auto_select=False,
        clip_order=["1.mp4", "2.mp4", "10.mp4"],
        target_resolution=(96, 128), verbose=False,
    )
    problems, critical = abe.validate_edl(plan["edl"])
    assert critical == [] and problems == []
    durs = plan["edl"]["clip_durations"]
    for s in plan["edl"]["segments"]:
        assert s["source_in"] >= 0.0
        assert s["source_out"] <= durs[s["clip_name"]] + 0.05
    # 每个素材都产出了候选窗口分（用于素材内优选）
    assert any(a["window_score"] for a in plan["assets"])
    for a in plan["assets"]:
        assert a["window_score"] is None or len(a["window_score"]) in (0, abe.WINDOW_CANDIDATES)
    for c in clips:
        c.close()

