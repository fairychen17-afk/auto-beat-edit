"""ffmpeg 渲染器：统一规格 → 镜头运动 → 卡点特效 → 调色 → xfade 转场链 → 混音 → 成品。

xfade 链时序模型（转场中点即卡点；每个边界可用各自的转场时长 d_k）：
    L_1 = c_1 + d_1/2
    L_i = c_i + (d_{i-1} + d_i)/2      （中间段）
    L_n = c_n + d_{n-1}/2
    O_k = sum(L_1..L_k) - sum(d_1..d_k)   （第 k 次 xfade 的 offset）
    总时长 = sum(c_i) = sum(L_i) - sum(d_k)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

from .moves import camera_chain, fx_chain
from .timeline import Timeline


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_has_audio(path: Path) -> bool:
    """用 ffmpeg -i 的 stderr 判断是否含音频流。"""
    r = subprocess.run([ffmpeg_path(), "-i", str(path)],
                       capture_output=True, text=True, errors="ignore")
    return "Audio:" in r.stderr


def _seg_filter(idx: int, seg, W: int, H: int, fps: int,
                bpm: float, amp: float) -> list[str]:
    """构造一个片段的滤镜链，返回若干条以 ';' 分隔的 filterchain。

    输出标签统一为 [v{idx}]，与转场链对接。
    段内变速（ramp）用 split + 两段不同速度 + concat 实现，避免依赖
    setpts 的时间单位约定，行为可预期。
    """
    dur = seg.duration
    avail = getattr(seg, "_avail", dur)
    tag = f"_{idx}"
    chains: list[str] = []
    cur = f"[{idx}:v]"

    # ---- 1. 取源 + 变速 ----
    ramp = getattr(seg, "ramp", None) if seg.asset.kind == "video" else None
    if ramp:
        a, b = float(ramp[0]), float(ramp[1])
        need = dur * 2 * a * b / (a + b)               # 段内变速所需源时长
        need = min(need, max(avail - 0.02, 0.2))
        half = max(need / 2, 0.05)
        chains.append(f"{cur}split=2[ra{tag}][rb{tag}]")
        chains.append(f"[ra{tag}]trim=start=0:duration={half:.3f},"
                      f"setpts=(PTS-STARTPTS)/{a:.4f}[raa{tag}]")
        chains.append(f"[rb{tag}]trim=start={half:.3f}:duration={half:.3f},"
                      f"setpts=(PTS-STARTPTS)/{b:.4f}[rbb{tag}]")
        chains.append(f"[raa{tag}][rbb{tag}]concat=n=2:v=1:a=0[src{tag}]")
        speed = need / max(dur, 1e-3)                  # 等效速度（用于补帧估算）
        produced = (need / 2) / a + (need / 2) / b     # 两段不同速度拼起来的实际时长
        cur = f"[src{tag}]"
    elif seg.asset.kind == "video":
        speed = max(float(getattr(seg, "speed", 1.0)), 0.05)
        src_need = dur * speed
        chains.append(f"{cur}trim=start=0:duration={src_need:.3f},"
                      f"setpts={1.0 / speed:.5f}*(PTS-STARTPTS)[src{tag}]")
        produced = min(src_need, max(avail, 0.0)) / speed
        cur = f"[src{tag}]"
    else:
        speed = 1.0
        chains.append(f"{cur}trim=duration={dur:.3f},"
                      f"setpts=PTS-STARTPTS[src{tag}]")
        produced = dur
        cur = f"[src{tag}]"

    # ---- 2. 镜头运动（超采 → 推拉/平移/脉冲 → 回到输出尺寸）----
    cam = camera_chain(getattr(seg, "move", "static"), W, H, dur, bpm, amp, fps)
    base = cam if cam else (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                            f"crop={W}:{H}")

    # ---- 3. 卡点特效 ----
    fx = fx_chain(getattr(seg, "fx", "none"), amp, W, H, bpm)

    # ---- 4. 调色 + 规格归一 + 补帧 ----
    tail_parts = [getattr(seg, "grade", "") or "null"]
    shortfall = dur - produced                     # 素材不足：克隆末帧补足
    if seg.asset.kind == "video" and shortfall > 0.01:
        tail_parts.append(f"tpad=stop_mode=clone:stop={shortfall:.3f}")
    tail_parts.append(f"fps={fps},setsar=1,format=yuv420p")
    tail = ",".join([p for p in tail_parts if p])

    if fx and "{I}" in fx:
        # 结构型特效（镜像分屏 / 画中画）需要 split / overlay，用独立链段拼接
        fx = fx.replace("{I}", tag)
        chains.append(f"{cur}{base}[pre{tag}]")
        chains.append(f"[pre{tag}]{fx}[post{tag}]")
        chains.append(f"[post{tag}]{tail}[v{idx}]")
    else:
        parts = [base, fx, tail]
        chains.append(cur + ",".join([p for p in parts if p]) + f"[v{idx}]")
    return chains


def render(tl: Timeline, output: Path, width: int = 1920, height: int = 1080,
           fps: int = 30, mix_original: bool = False) -> Path:
    ff = ffmpeg_path()
    segs = tl.segments
    n = len(segs)
    total = max(tl.total_duration, 0.3)
    bpm = float(tl.bgm.bpm)
    amp = float(getattr(tl, "amp", 0.85))

    # ---- 输入 ----
    args = [ff, "-y"]
    for seg in segs:
        if seg.asset.kind == "video":
            args += ["-ss", f"{seg.src_in:.3f}", "-i", str(seg.asset.path)]
        else:
            args += ["-loop", "1", "-framerate", str(fps),
                     "-t", f"{seg.duration:.3f}", "-i", str(seg.asset.path)]
    bgm_idx = n
    args += ["-ss", f"{tl.bgm_start:.3f}", "-t", f"{tl.bgm_end - tl.bgm_start + 1.0:.3f}",
             "-i", str(tl.bgm.path)]

    # ---- 视频链 ----
    fc: list[str] = []
    for i, seg in enumerate(segs):
        fc += _seg_filter(i, seg, width, height, fps, bpm, amp)

    # xfade 转场链：每个边界可用各自的转场时长（跳切≈硬切、硬档更短促）
    bdurs = tl.boundary_durs
    prev = "v0"
    cum = segs[0].duration
    for i in range(1, n):
        d_i = bdurs[i - 1]
        offset = cum - d_i
        out = f"x{i}" if i < n - 1 else "vout"
        fc.append(f"[{prev}][v{i}]xfade=transition={segs[i - 1].transition}"
                  f":duration={d_i:.3f}:offset={offset:.3f}[{out}]")
        prev = out
        cum = cum + segs[i].duration - d_i
    if n == 1:
        fc.append("[v0]copy[vout]")

    # 片头/片尾黑场，让成片有完整的起止
    fade_in_d = min(0.25, total * 0.1)
    fade_out_d = min(0.6, total * 0.2)
    fade_out_st = max(0.0, total - fade_out_d)
    fc.append(f"[vout]format=yuv420p,"
              f"fade=t=in:st=0:d={fade_in_d:.3f},"
              f"fade=t=out:st={fade_out_st:.3f}:d={fade_out_d:.3f}[vfinal]")

    # ---- 音频链 ----
    ratio = tl.tempo_ratio
    atempo = ""
    r = ratio
    while r > 2.0:      # atempo 单段限 [0.5, 2.0]
        atempo += "atempo=2.0,"
        r /= 2.0
    while r < 0.5:
        atempo += "atempo=0.5,"
        r /= 0.5
    atempo += f"atempo={r:.4f}"
    fade_out_start = max(0.0, total - 1.5)
    bgm_chain = (f"[{bgm_idx}:a]aresample=44100,{atempo},"
                 f"atrim=duration={total:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out_start:.3f}:d=1.5,"
                 f"volume=1.0[bgm]")

    audio_srcs = []
    if mix_original:
        for i, seg in enumerate(segs):
            if seg.asset.kind == "video" and probe_has_audio(seg.asset.path):
                fc.append(f"[{i}:a]aresample=44100,volume=0.12[a{i}]")
                audio_srcs.append(f"[a{i}]")
    fc.append(bgm_chain)
    if audio_srcs:
        mix_in = "[bgm]" + "".join(audio_srcs)
        fc.append(f"{mix_in}amix=inputs={1 + len(audio_srcs)}:"
                  f"duration=first:normalize=0[aout]")
    else:
        fc.append("[bgm]anull[aout]")

    args += ["-filter_complex", ";".join(fc),
             "-map", "[vfinal]", "-map", "[aout]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", "-t", f"{total:.3f}",
             str(output)]
    proc = subprocess.run(args, capture_output=True, text=True, errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 渲染失败:\n" + proc.stderr[-3000:])
    return output
