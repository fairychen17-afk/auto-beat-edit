"""命令行入口：python main.py --material 素材目录 --bgm BGM目录/文件 --output out.mp4"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .audio_analyzer import analyze_bgm, scan_bgm_library, AUDIO_EXTS
from .bgm_matcher import auto_style, material_mood, material_profile, pick_bgm
from .scene_detect import scan_materials
from .timeline import build_timeline
from .edl import export_edl
from .renderer import render


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autocut",
                                 description="自动卡点剪辑：关键帧检测 + BGM 匹配 + 卡点转场成片")
    ap.add_argument("--material", required=True, help="素材目录（视频/图片混放）")
    ap.add_argument("--bgm", required=True, help="BGM 目录（自动挑选）或单个音频文件")
    ap.add_argument("--output", default="output/final.mp4", help="输出视频路径")
    ap.add_argument("--duration", type=float, default=None, help="目标成片时长（秒），默认按素材量自动")
    ap.add_argument("--resolution", default="1920x1080", help="如 1920x1080 / 1080x1920")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mix-original", action="store_true", help="保留素材原声（压低混入）")
    ap.add_argument("--allow-reuse", action="store_true",
                    help="允许素材窗口重复使用（默认每段只用一次，素材用完即止）")
    ap.add_argument("--preserve-order", action="store_true",
                    help="按素材文件名/时间顺序排列片段（默认按能量与高潮匹配穿插）")
    ap.add_argument("--bgm-start", type=float, default=None,
                    help="BGM 从第几秒开始剪辑（默认自动定位高潮段）")
    ap.add_argument("--style", default="auto", choices=["auto", "dynamic", "balanced", "cinematic"],
                    help="剪辑风格：dynamic 动感快切 / balanced 均衡 / cinematic 电影感慢镜（默认按情绪自动）")
    ap.add_argument("--intensity", type=float, default=1.0,
                    help="运动与特效强度倍率（0.3~1.5，默认 1.0）")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；同一素材+种子可复现同一版剪辑")
    ap.add_argument("--camera-motion", default="off", choices=["off", "subtle", "full"],
                    help="镜头运动：off 完全稳定（默认，只切镜/转场/调色/节拍特效）"
                         " / subtle 仅温和缓推缓拉 / full 完整运镜（推拉/平移/抖动/脉冲）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做分析+编排，导出剪辑决策表（EDL JSON）并打印报告，不渲染视频")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="打印逐段剪辑决策与理由")
    ap.add_argument("--edl", default=None,
                    help="把剪辑决策表导出为 JSON 文件（可与 --dry-run 一起用）")
    args = ap.parse_args(argv)

    t0 = time.time()
    material_dir = Path(args.material)
    bgm_path = Path(args.bgm)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = parse_size(args.resolution)

    print(f"[1/4] 扫描素材: {material_dir}")
    assets = scan_materials(material_dir)
    if not assets:
        print("未找到可用素材（支持 mp4/mov/jpg/png 等）")
        return 1
    n_shots = sum(len(a.shots) for a in assets if a.kind == "video")
    n_imgs = sum(1 for a in assets if a.kind == "image")
    print(f"      视频 {sum(1 for a in assets if a.kind == 'video')} 个（{n_shots} 个镜头），图片 {n_imgs} 张")

    print(f"[2/4] 分析 BGM: {bgm_path}")
    if bgm_path.is_dir():
        candidates = scan_bgm_library(bgm_path)
    else:
        candidates = [analyze_bgm(bgm_path)]
    if not candidates:
        print("未找到可用 BGM")
        return 1
    profile = material_profile(assets)
    mood = profile["mood"]
    bgm = pick_bgm(candidates, mood, min_duration=args.duration or 15.0,
                   profile=profile)
    print(f"      素材画像: 情绪={mood} 能量={profile['energy']:.2f} "
          f"镜头均长={profile['avg_shot_len']:.1f}s 期望BPM={profile['desired_bpm']:.0f} "
          f"明暗={profile['brightness']:.2f} 色彩={profile['colorfulness']:.2f}")
    print(f"      景别分布: 特写 {profile['close_ratio']:.0%} / 全景 {profile['wide_ratio']:.0%}"
          f" | 镜头运动均值 {profile['motion']:.2f} 静止镜头 {profile['static_ratio']:.0%}")
    style = args.style
    if style == "auto":
        style = auto_style(profile)
        print(f"      按素材自动选风格 → {style}")
    else:
        print(f"      风格（手动指定）→ {style}")
    print(f"      → 选中 BGM《{bgm.path.name}》"
          f" bpm={bgm.bpm:.0f} mood={bgm.mood} 时长={bgm.duration:.1f}s 高潮={bgm.climax:.1f}s")

    print("[3/4] 编排时间线（卡点吸附）")
    tl = build_timeline(assets, bgm, mood, target_duration=args.duration,
                        allow_reuse=args.allow_reuse,
                        preserve_order=args.preserve_order,
                        bgm_start=args.bgm_start, style=style,
                        intensity=args.intensity, seed=args.seed,
                        camera_motion=args.camera_motion)
    uniq = len({(s.asset.path, round(s.src_in, 2)) for s in tl.segments})
    print(f"      素材窗口去重后仍使用 {uniq}/{len(tl.segments)} 段（不重复）"
          if uniq == len(tl.segments) else
          f"      警告：{len(tl.segments) - uniq} 段重复使用了素材")
    print(f"      {len(tl.segments)} 段，成片约 {tl.total_duration:.1f}s，"
          f"BGM 截取 {tl.bgm_start:.1f}~{tl.bgm_end:.1f}s，变速比 {tl.tempo_ratio:.3f}")
    if tl.bgm_sections:
        secs = " ".join(f"{s:.0f}-{e:.0f}s:{lv}" for s, e, lv in tl.bgm_sections[:6])
        print(f"      BGM 结构：{secs}（low=前奏/间奏，high=副歌/drop）")

    # 手法统计：让"剪得怎么样"可被看见
    durs = [s.duration for s in tl.segments]
    tds = tl.boundary_durs or [tl.trans_dur]
    print(f"      风格={tl.style} 强度={tl.amp:.2f} | 段长 {min(durs):.2f}~{max(durs):.2f}s"
          f"（随音乐能量起伏）| 转场 {min(tds):.2f}~{max(tds):.2f}s")
    print({"off": "      镜头: 稳定模式（无运镜，画面不晃动）",
           "subtle": "      镜头: 温和模式（仅缓推缓拉/呼吸）",
           "full": "      镜头: 全运镜模式（推拉/平移/抖动等）"}[args.camera_motion])
    mv, fx_, tr, gr = {}, {}, {}, {}
    for s in tl.segments:
        mv[s.move] = mv.get(s.move, 0) + 1
        fx_[s.fx] = fx_.get(s.fx, 0) + 1
        tr[s.transition] = tr.get(s.transition, 0) + 1
        gr[s.grade] = gr.get(s.grade, 0) + 1
    print(f"      镜头运动: {', '.join(f'{k}×{v}' for k, v in mv.items())}")
    print(f"      卡点特效: {', '.join(f'{k}×{v}' for k, v in fx_.items())}")
    print(f"      转场手法: {len(tr)} 种 | 调色预设: {len(gr)} 种（相邻不重复）")
    sp = [f"{s.speed:.2f}x" for s in tl.segments if abs(s.speed - 1.0) > 0.02]
    rp = [f"{s.ramp[0]:.1f}→{s.ramp[1]:.1f}" for s in tl.segments if s.ramp]
    if sp:
        print(f"      整体变速段: {', '.join(sp)}")
    if rp:
        print(f"      段内变速: {', '.join(rp)}")

    # 决策明细：每段用了什么、为什么（--verbose 或 --dry-run 时打印）
    if args.verbose or args.dry_run:
        print("      ── 逐段决策 ──")
        for i, s in enumerate(tl.segments):
            print(f"      {i:>2}. {s.asset.path.name} 入点{s.src_in:.2f}s "
                  f"{s.duration:.2f}s | {s.move}/{s.fx}/{s.transition}@"
                  f"{(s.tdur or tl.trans_dur):.2f}s"
                  + (f" | {s.why}" if s.why else ""))

    edl_path = args.edl
    if edl_path is None and args.dry_run:
        edl_path = str(output.with_suffix(".edl.json"))
    if edl_path:
        export_edl(tl, Path(edl_path), width, height, args.fps)
        print(f"      剪辑决策表已导出: {edl_path}")

    if args.dry_run:
        print("完成（dry-run：未渲染视频）")
        return 0

    print("[4/4] 渲染输出")
    render(tl, output, width, height, args.fps, mix_original=args.mix_original)
    print(f"完成: {output}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
