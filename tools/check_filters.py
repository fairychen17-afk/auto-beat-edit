"""开发自检：逐个渲染每一种手法，确认 ffmpeg 滤镜链都成立。

用法：python tools/check_filters.py [素材.mp4]
输出每种手法的渲染结果（OK / FAIL），用于在改动手法库后快速回归。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocut.moves import FX_POOL, CAMERA_POOL, GRADE_POOL, GRADE_LOW_SAT  # noqa: E402
from autocut.renderer import _seg_filter, ffmpeg_path  # noqa: E402
from autocut.scene_detect import analyze_video  # noqa: E402
from autocut.timeline import Segment  # noqa: E402

W, H, FPS, BPM = 320, 180, 30, 120.0
DUR = 1.5


def check(seg: Segment, label: str, tmp: Path) -> bool:
    chains = _seg_filter(0, seg, W, H, FPS, BPM, 1.0)
    args = [ffmpeg_path(), "-y", "-ss", "0.5", "-i", str(seg.asset.path),
            "-filter_complex", ";".join(chains), "-map", "[v0]",
            "-c:v", "libx264", "-preset", "ultrafast", "-t", f"{DUR}",
            str(tmp / "out.mp4")]
    p = subprocess.run(args, capture_output=True, text=True, errors="ignore")
    if p.returncode != 0:
        tail = [ln for ln in p.stderr.splitlines() if "Error" in ln or "error" in ln]
        print(f"  FAIL {label}: {tail[-1][:160] if tail else p.stderr[-160:]}")
        return False
    print(f"  OK   {label}")
    return True


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run_material/1.mp4")
    if not src.exists():
        print(f"找不到测试视频 {src}，请传入一个 mp4 路径")
        return 1
    asset = analyze_video(src)
    if asset is None:
        print("视频无法解析")
        return 1

    def seg(**kw) -> Segment:
        kw.setdefault("grade", "")
        s = Segment(asset=asset, src_in=0.0, duration=DUR,
                    transition="fade", **kw)
        s._avail = 3.0            # noqa: SLF001
        return s

    ok = 0
    total = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("== 卡点特效 ==")
        for fx in sorted({f for pool in FX_POOL.values() for f in pool}):
            total += 1
            ok += check(seg(fx=fx), fx, tmp)
        print("== 镜头运动（static 即关闭运镜） ==")
        for mv in sorted({m for pool in CAMERA_POOL.values() for m in pool}):
            total += 1
            ok += check(seg(move=mv), mv, tmp)
        print("== 调色 ==")
        for g in sorted(set(sum(GRADE_POOL.values(), [])) | set(GRADE_LOW_SAT)):
            total += 1
            ok += check(seg(grade=g), g[:40], tmp)
        print("== 段内变速 / 跳切 / 组合 ==")
        for label, kw in [("ramp 0.6→1.7", dict(ramp=(0.6, 1.7))),
                          ("ramp 1.7→0.6", dict(ramp=(1.7, 0.6))),
                          ("speed 1.35", dict(speed=1.35)),
                          ("speed 0.62", dict(speed=0.62)),
                          ("pip+ramp", dict(fx="pip_card", ramp=(1.5, 0.7))),
                          ("mirror+grade", dict(fx="split_mirror",
                                                grade="eq=saturation=1.2"))]:
            total += 1
            ok += check(seg(**kw), label, tmp)
    print(f"\n{ok}/{total} 通过")
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())
