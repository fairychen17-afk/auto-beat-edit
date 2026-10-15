"""剪辑手法库（shot language）+ 素材感知编排。

手法分四层：
  - 镜头运动 camera：推拉 / 平移 / 漂移 / 摇摆 / 呼吸 / 冲击缩放 / 对角 Ken Burns /
                    心跳脉冲 / 缩放脉冲 / 抖动（超采 + zoompan 逐帧驱动）
  - 卡点特效 fx：白闪 / 黑闪 / 红闪 / 每拍亮度脉冲 / 饱和度脉冲 / 频闪 /
                 镜像分屏 / 色差 / 颗粒 / 翻转 / 电影遮幅 / 复古胶片
  - 调色 grade：按情绪多套预设 + 低饱和平衡组（艳丽素材自动降饱和）
  - 转场 transition：soft / mid / hard 三档，按段落能量选取

编排是「素材感知」的（plan_shots）：
  - 主体居中明显 → 禁用分屏/翻转/抖动等破坏性手法，优先温和推拉
  - 暗段配白闪、亮段配黑闪（对比原则，闪光才看得见）
  - 素材本身艳丽（饱和度高）→ 自动换低饱和调色组避免过艳
  - 原镜头有横向运动趋势 → 平移手法顺势借力
  - 音乐结构边界（能量阶跃）→ 强制高能转场
所有周期性运动的频率都由 BGM 的 BPM 推导，画面与音乐共用同一时钟。
"""
from __future__ import annotations

import random

# 超采系数：先把画面放大到 OVERSCAN 倍，再在其中做推拉/平移，
# 这样放大时不会露出黑边，且最终 crop 回输出分辨率只损失一次重采样。
OVERSCAN = 1.5


# ---------------------------------------------------------------- 镜头运动
CAMERA_POOL = {
    "dynamic":   ["zoom_punch", "punch_pan", "zoom_in", "shake", "pan_left",
                  "pan_right", "crash_zoom", "heartbeat", "sway", "zoom_out",
                  "ken_diag", "static"],
    "balanced":  ["zoom_in", "zoom_out", "pan_left", "pan_right", "drift",
                  "ken_diag", "breath", "zoom_punch", "sway", "static"],
    "cinematic": ["zoom_in", "zoom_out", "drift", "pan_up", "pan_down",
                  "breath", "ken_diag", "static"],
}

# 图片素材不适合的运动
_IMAGE_SOFT = {"zoom_in", "zoom_out", "drift", "pan_up", "pan_down",
               "pan_left", "pan_right", "zoom_punch", "ken_diag",
               "breath", "static"}

# 主体居中明显的镜头（人脸/主体特写）只许用温和手法
_SUBJECT_SAFE = {"zoom_in", "zoom_out", "drift", "breath", "ken_diag",
                 "static", "crash_zoom"}
# 对主体破坏性的特效
_SUBJECT_BAD_FX = {"split_mirror", "hflip", "strobe", "chroma"}

# subtle 档允许的温和运动（缓推缓拉/呼吸/漂移，不抖动、不平移、不脉冲）
GENTLE_MOVES = {"zoom_in", "zoom_out", "breath", "drift", "ken_diag", "static"}


def camera_chain(move: str, W: int, H: int, dur: float,
                 bpm: float, amp: float, fps: int = 30) -> str:
    """生成镜头运动的滤镜链；static 返回空串（调用方走直出路径）。

    实现：先超采到 OVERSCAN 倍（留出推拉/平移空间），再用 zoompan 逐帧计算
    缩放与位移。zoompan 的 z/x/y 是逐帧求值的表达式，时间用输出帧号 on 推导
    （on/fps = 秒），因此可以做出与鼓点同相位的缩放脉冲与抖动。
    """
    if move == "static":
        return ""

    Df = max(dur * fps, 1.0)                       # 段总帧数
    Tf = max((60.0 / max(bpm, 40.0)) * fps, 1.0)   # 一拍多少帧
    OW, OH = int(round(W * OVERSCAN)), int(round(H * OVERSCAN))
    up = (f"scale={OW}:{OH}:force_original_aspect_ratio=increase,"
          f"crop={OW}:{OH},fps={fps}")
    P = f"(1-cos(2*PI*on/{Tf:.4f}))/2"             # 每拍脉冲 0→1→0
    u = f"on/{Df:.4f}"                             # 0~1 段内进度
    sp = "(iw-iw/zoom)"                            # 可平移空间

    def zp(z: str, offx: str = "0", offy: str = "0") -> str:
        x = f"iw/2-(iw/zoom/2)+({offx})"
        y = f"ih/2-(ih/zoom/2)+({offy})"
        return (f"zoompan=z='{z}':x='{x}':y='{y}'"
                f":d=1:s={W}x{H}:fps={fps}")

    if move == "zoom_in":
        a = 0.30 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*(1+{a:.4f}*{u})')}"

    if move == "zoom_out":
        a = 0.30 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*({1 + a:.4f}-{a:.4f}*{u})')}"

    # ---- 平移类 ----
    zc = f"{OVERSCAN:.3f}*{1.0 + 0.18 * amp:.4f}"
    if move == "pan_left":
        return f"{up},{zp(zc, f'{sp}*(0.5-{u})')}"
    if move == "pan_right":
        return f"{up},{zp(zc, f'{sp}*({u}-0.5)')}"
    if move == "pan_up":
        return f"{up},{zp(zc, '0', f'{sp}*(0.5-{u})')}"
    if move == "pan_down":
        return f"{up},{zp(zc, '0', f'{sp}*({u}-0.5)')}"

    if move == "drift":
        return f"{up},{zp(zc, f'{sp}*0.5*sin(2*PI*{u})', f'{sp}*0.35*cos(2*PI*{u})')}"

    if move == "ken_diag":
        # 对角 Ken Burns：缩放 + 斜向平移同时进行
        a = 0.22 * amp
        z = f"{OVERSCAN:.3f}*(1+{a:.4f}*{u})"
        return f"{up},{zp(z, f'{sp}*({u}-0.5)', f'{sp}*({u}-0.5)')}"

    if move == "breath":
        # 呼吸：两拍一个周期的缓慢起伏，适合长镜头/人像
        a = 0.07 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*(1+{a:.4f}*(1-cos(PI*on/{Tf:.4f}))/2)')}"

    if move == "crash_zoom":
        # 冲击缩放：段的前 20% 快速推上去然后停住（高潮点用）
        a = 0.26 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*(1+{a:.4f}*min({u}/0.2,1))')}"

    # ---- 卡点类 ----
    if move == "zoom_punch":
        a = 0.14 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*(1+{a:.4f}*{P})')}"

    if move == "heartbeat":
        # 心跳双脉冲：每拍脉冲，但一大一小交替（两拍一个心跳）
        a = 0.12 * amp
        return f"{up},{zp(f'{OVERSCAN:.3f}*(1+{a:.4f}*{P}*(1+0.4*cos(PI*on/{Tf:.4f})))')}"

    if move == "punch_pan":
        a = 0.12 * amp
        z = f"{OVERSCAN:.3f}*(1+{a:.4f}*{P})"
        return f"{up},{zp(z, f'{0.012 * amp:.4f}*iw*sin(2*PI*on/{Tf:.4f})')}"

    if move == "shake":
        z = f"{OVERSCAN:.3f}"
        return f"{up},{zp(z, f'{0.014 * amp:.4f}*iw*sin(2*PI*on/{Tf:.4f})', f'{0.011 * amp:.4f}*ih*sin(2*PI*on/{Tf:.4f}+1.5708)')}"

    if move == "sway":
        a = 0.030 * amp
        T = max(60.0 / max(bpm, 40.0), 0.05)
        return (f"{up},"
                f"rotate=a='{a:.5f}*sin(2*PI*t/{T:.4f})':ow={OW}:oh={OH}"
                f":fillcolor=black,"
                f"crop={W}:{H}:({OW}-{W})/2:({OH}-{H})/2")

    return ""


# ---------------------------------------------------------------- 卡点特效
FX_POOL = {
    "dynamic":   ["none", "beat_flash", "flash_black", "hflip", "chroma",
                  "split_mirror", "beat_pulse", "strobe", "flash_white",
                  "glitch", "echo", "flash_freeze"],
    "balanced":  ["none", "none", "beat_flash", "flash_black", "grain",
                  "beat_pulse", "letterbox", "echo", "warm_leak", "pip_card"],
    "cinematic": ["none", "none", "flash_black", "letterbox", "old_film",
                  "grain", "warm_leak", "echo"],
}

# 主体破坏性特效（特写镜头禁用）
_BAD_ON_CLOSE = {"split_mirror", "hflip", "strobe", "chroma", "glitch",
                 "pip_card", "flash_freeze"}


def fx_chain(name: str, amp: float, W: int, H: int, bpm: float = 120.0) -> str:
    """卡点特效滤镜链。{I} 是需要由渲染器替换成段号的标签占位符。"""
    T = max(60.0 / max(bpm, 40.0), 0.05)      # 一拍的秒数
    if name == "none":
        return ""
    if name == "flash_white":
        return f"fade=t=in:st=0:d={0.09 + 0.05 * amp:.3f}:color=white"
    if name == "flash_black":
        return f"fade=t=in:st=0:d={0.07 + 0.04 * amp:.3f}:color=black"
    if name == "flash_red":
        return f"fade=t=in:st=0:d={0.08 + 0.04 * amp:.3f}:color=red"
    if name == "beat_flash":
        # 每拍瞬间提亮再衰减（eq 支持 t 表达式）
        return (f"eq=brightness='{0.13 * amp:.3f}*exp(-10*(t-{T:.4f}"
                f"*floor(t/{T:.4f})))'")
    if name == "beat_dim":
        # 每拍瞬间压暗再恢复（反向闪，适合亮画面）
        return (f"eq=brightness='-{0.12 * amp:.3f}*exp(-10*(t-{T:.4f}"
                f"*floor(t/{T:.4f})))'")
    if name == "beat_pulse":
        # 饱和度与对比度随拍脉动
        p = f"(1-cos(2*PI*t/{T:.4f}))/2"
        return (f"eq=saturation='1+{0.30 * amp:.3f}*{p}'"
                f":contrast='1+{0.12 * amp:.3f}*{p}'")
    if name == "strobe":
        # 频闪：半拍一次明暗交替（低幅度，避免过闪）
        return f"eq=brightness='{0.09 * amp:.3f}*(1-cos(4*PI*t/{T:.4f}))/2'"
    if name == "hflip":
        return "hflip"
    if name == "chroma":
        v = int(3 + 5 * amp)
        return f"rgbashift=rh={v}:bh=-{v}"
    if name == "glitch":
        # 故障感：RGB 分离 + 噪点 + 对比，短暂卡点用
        v = int(4 + 6 * amp)
        return (f"rgbashift=rh={v}:bh=-{v}:gv={max(1, v // 3)},"
                f"noise=alls={int(6 + 8 * amp)}:allf=t,eq=contrast=1.14")
    if name == "echo":
        # 叠影拖尾：与前一帧混合，产生残影
        return f"tblend=all_mode=average:all_opacity={0.22 + 0.18 * amp:.3f}"
    if name == "flash_freeze":
        # 定格式强闪：白闪 + 高对比，视觉上像"定格一拍"
        return (f"fade=t=in:st=0:d={0.08 + 0.05 * amp:.3f}:color=white,"
                f"eq=contrast=1.22:saturation=1.10")
    if name == "warm_leak":
        # 暖调漏光：抬暗部 + 暖色偏移 + 轻晕影
        return ("curves=all='0/0.05 0.5/0.55 1/1',"
                "colorbalance=rs=0.10:gs=0.03:bs=-0.06,"
                "eq=saturation=1.12,vignette=PI/4.5")
    if name == "pip_card":
        # 画中画卡片：模糊压暗的背景 + 居中缩小前景，做"海报/封面"感
        # {{I}} 是给渲染器替换的标签占位符（f-string 里需写成双花括号）
        fw, fh = int(W * 0.62) // 2 * 2, int(H * 0.62) // 2 * 2
        return (f"split=2[pa{{I}}][pb{{I}}];"
                f"[pa{{I}}]gblur=sigma=24,eq=brightness=-0.10[pbg{{I}}];"
                f"[pb{{I}}]scale={fw}:{fh}[pfg{{I}}];"
                f"[pbg{{I}}][pfg{{I}}]overlay=(W-w)/2:(H-h)/2")
    if name == "grain":
        return f"noise=alls={int(3 + 7 * amp)}:allf=t"
    if name == "letterbox":
        bh = int(H * 0.11)
        return f"crop={W}:{H - 2 * bh}:0:{bh},pad={W}:{H}:0:{bh}:color=black"
    if name == "old_film":
        return ("noise=alls=8:allf=t,hue=h=15:s=0.7,"
                "eq=contrast=1.05:brightness=0.02,vignette=PI/4")
    if name == "split_mirror":
        half = W // 2
        # 左半 + 镜像右半 → 对称「蝴蝶」分屏，横向不变形
        return (f"split=2[b{{I}}a][b{{I}}b];"
                f"[b{{I}}b]hflip,crop={half}:{H}:{W - half}:0[b{{I}}c];"
                f"[b{{I}}a]crop={half}:{H}:0:0[b{{I}}d];"
                f"[b{{I}}d][b{{I}}c]hstack=inputs=2")
    return ""


# ---------------------------------------------------------------- 调色
GRADE_POOL = {
    "energetic": [
        "eq=contrast=1.16:saturation=1.32,vignette=PI/4",
        "eq=contrast=1.10:saturation=1.22,colorbalance=rs=0.10:bs=-0.06",
        "eq=contrast=1.22:saturation=1.38:gamma=1.02,vignette=PI/5",
        "colorbalance=rs=0.12:gs=0.02:bs=-0.08,eq=saturation=1.18:contrast=1.12",
        "eq=contrast=1.14:saturation=1.28,noise=alls=5:allf=t",
        "hue=h=6:s=1.22,eq=contrast=1.12:saturation=1.15",
    ],
    "chill": [
        "eq=saturation=1.12:contrast=1.05,colorbalance=rs=0.05:gs=0.02",
        "eq=saturation=1.08,vignette=PI/5",
        "colorbalance=rs=0.06:bs=-0.03,eq=brightness=0.02",
        "eq=saturation=1.15:contrast=1.08:gamma=1.03,hue=h=-4:s=1.1",
        "eq=saturation=1.05:brightness=0.015,noise=alls=4:allf=t",
    ],
    "calm": [
        "eq=saturation=0.88:brightness=0.03:contrast=0.98",
        "colorbalance=bs=0.05:rs=-0.03,eq=saturation=0.92",
        "eq=saturation=0.95:gamma=1.08,vignette=PI/5",
        "eq=saturation=0.90:contrast=1.02,hue=h=-6:s=0.95",
    ],
}

# 素材本身就很艳时用的低饱和平衡组（避免叠buff过艳）
GRADE_LOW_SAT = [
    "eq=saturation=0.95:contrast=1.08",
    "eq=saturation=0.92:gamma=1.04,vignette=PI/5",
    "colorbalance=gs=0.02,eq=saturation=0.90:contrast=1.05",
]


# ---------------------------------------------------------------- 转场
TRANS_TIERS = {
    "soft": ["fade", "dissolve", "fadeblack", "smoothleft", "smoothright",
             "smoothup", "smoothdown", "hblur", "fadegrays", "vertopen",
             "horzopen", "vertclose", "horzclose"],
    "mid":  ["slideleft", "slideright", "slideup", "slidedown", "wipeleft",
             "wiperight", "wipeup", "wipedown", "circleopen", "circleclose",
             "circlecrop", "radial", "coverleft", "coverright", "revealleft",
             "revealright", "rectcrop"],
    "hard": ["fadewhite", "pixelize", "diagtl", "diagbr", "hlslice",
             "hrslice", "vuslice", "vdslice", "squeezeh", "squeezev",
             "zoomin", "hlwind", "hrwind", "wipetl", "wipebr", "distance"],
}

# 方向性转场：可按素材运动方向顺势选择（匹配剪辑）
DIRECTIONAL_TRANS = {
    "left":  ["slideleft", "wipeleft", "coverleft", "revealleft", "smoothleft",
              "hlslice"],
    "right": ["slideright", "wiperight", "coverright", "revealright",
              "smoothright", "hrslice"],
    "up":    ["slideup", "wipeup", "smoothup"],
    "down":  ["slidedown", "wipedown", "smoothdown"],
}

# 各情绪的基准转场时长（秒），实际按段落能量档缩放
BASE_TDUR = {"calm": 0.62, "chill": 0.50, "energetic": 0.30}
HARD_CUT_DUR = 0.05          # 跳切：极短转场≈硬切


def tier_of(energy: float) -> str:
    if energy > 0.72:
        return "hard"
    if energy > 0.45:
        return "mid"
    return "soft"


def _pick(rng: random.Random, pool: list[str], prev: str) -> str:
    cand = [x for x in pool if x != prev] or pool
    return cand[rng.randrange(len(cand))]


def plan_shots(feats: list[dict], mood: str, style: str,
               rng: random.Random,
               camera_motion: str = "off") -> list[dict]:
    """素材感知编排：逐段决定「怎么剪」，并给出决策理由。

    feats[i] 可含：
      energy        该段所处音乐能量 0~1
      brightness    画面明暗 0~1
      colorfulness  画面色彩丰富度 0~1
      subject_center 主体居中度 0~1
      motion_dir    镜头自身横向运动方向（<0 左移 / >0 右移）
      motion        镜头运动强度 0~1
      shot_type     景别 close / medium / wide
      section       音乐结构 low / mid / high
      boundary      是否落在音乐结构边界
      jump          是否与上一段同镜头相邻窗口（可做跳切）
      kind          video / image

    返回每段：{move, fx, grade, transition, speed, ramp, tdur, why}
    规则要点：
      * 特写 → 保护构图：温和运动 + 不用破坏性特效
      * 全景/静止镜头 → 可以放开手脚：结构型特效、更猛的转场
      * 镜头本身运动强 → 特效收敛，避免画面过花
      * 音乐低能量段 → 氛围型特效 + 长转场；高能量段 → 强闪 + 短转场
      * 同镜头相邻窗口 → 跳切（极短转场）
      * 画面有横向运动 → 转场方向顺势（匹配剪辑）
      * build/落定位置 → 段内变速（先慢后快 / 先快后慢）

    camera_motion 三档：off 完全稳定 / subtle 仅温和缓推缓拉 / full 完整运镜库。
    特效、转场、调色、变速任何档位都照常生效。
    """
    cam_pool = list(CAMERA_POOL.get(style, CAMERA_POOL["balanced"]))
    fx_pool = list(FX_POOL.get(style, FX_POOL["balanced"]))
    grade_pool = GRADE_POOL.get(mood, GRADE_POOL["chill"])
    base_tdur = BASE_TDUR.get(mood, 0.5)
    if camera_motion == "subtle":
        cam_pool = [m for m in cam_pool if m in GENTLE_MOVES] or ["static"]

    plan: list[dict] = []
    prev_move = prev_fx = prev_trans = prev_grade = ""
    for i, f in enumerate(feats):
        e = f.get("energy", 0.5)
        tier = tier_of(e)
        why: list[str] = []
        shot_type = f.get("shot_type", "medium")
        section = f.get("section", "mid")
        md = f.get("motion_dir", 0.0)
        is_jump = bool(f.get("jump"))

        # ---- 转场：结构边界抬档，跳切用极短转场 ----
        if f.get("boundary"):
            tier = "hard"
            why.append("音乐结构边界")
        if is_jump:
            trans = "fade"
            tdur = HARD_CUT_DUR
            why.append("同镜头跳切")
        else:
            # 方向性匹配剪辑：画面往哪边动，转场就往哪边推
            if abs(md) > 0.25 and tier != "soft":
                key = "right" if md > 0 else "left"
                dir_pool = [t for t in DIRECTIONAL_TRANS[key]
                            if t in TRANS_TIERS[tier]] or DIRECTIONAL_TRANS[key]
                trans = _pick(rng, dir_pool, prev_trans)
                why.append("转场顺着画面" + ("右" if md > 0 else "左") + "移")
            else:
                trans = _pick(rng, TRANS_TIERS[tier], prev_trans)
            # 转场时长：硬档更短促，软档更长
            factor = {"hard": 0.62, "mid": 1.0, "soft": 1.2}[tier]
            tdur = float(min(0.9, max(0.05, base_tdur * factor)))

        # ---- 镜头运动 ----
        if camera_motion in (False, "off", "none"):
            move = "breath" if f["kind"] == "image" else "static"
        else:
            move_pool = cam_pool
            if shot_type == "close":
                safe = [m for m in cam_pool if m in _SUBJECT_SAFE]
                if safe:
                    move_pool = safe
                why.append("特写保护构图")
            move = _pick(rng, move_pool, prev_move)
            if abs(md) > 0.25 and rng.random() < 0.5:
                want = "pan_right" if md > 0 else "pan_left"
                if want in move_pool:
                    move = want
                    why.append("运镜顺势")
            if tier == "hard" and "crash_zoom" in move_pool and rng.random() < 0.3:
                move = "crash_zoom"
                why.append("高潮冲击缩放")
            if f["kind"] == "image":
                if move not in _IMAGE_SOFT:
                    move = rng.choice(sorted(_IMAGE_SOFT & set(move_pool)) or ["zoom_in"])
                if move == "static":
                    move = "zoom_in"

        # ---- 卡点特效 ----
        fx_pool_i = list(fx_pool)
        if shot_type == "close":
            fx_pool_i = [x for x in fx_pool_i if x not in _BAD_ON_CLOSE] or ["none"]
            why.append("特写不用破坏性特效")
        if f["kind"] == "image":
            fx_pool_i = [x for x in fx_pool_i
                         if x not in ("split_mirror", "strobe", "hflip",
                                      "glitch", "pip_card")]
        if f.get("motion", 0.0) > 0.60:
            keep = [x for x in fx_pool_i if x not in ("strobe", "glitch", "echo")]
            if keep:
                fx_pool_i = keep
            why.append("镜头本身在动，特效收敛")
        if section == "low":
            amb = [x for x in fx_pool_i
                   if x in ("none", "letterbox", "old_film", "warm_leak", "grain")]
            if amb and rng.random() < 0.6:
                fx_pool_i = amb
                why.append("低能量段配氛围特效")
        elif section == "high" and tier == "hard":
            strong = [x for x in fx_pool_i
                      if x in ("beat_flash", "beat_pulse", "flash_white",
                               "flash_black", "strobe", "echo", "glitch")]
            if strong:
                fx_pool_i = strong
                why.append("高潮段配节拍特效")
        fx = _pick(rng, fx_pool_i, prev_fx)
        # 明暗对比：暗画面配亮闪、亮画面配暗闪，才看得见
        if fx in ("flash_white", "flash_black", "beat_flash", "beat_dim",
                  "flash_freeze"):
            dark = f.get("brightness", 0.5) < 0.45
            fx = {"flash_white": "flash_white" if dark else "flash_black",
                  "flash_black": "flash_black" if not dark else "flash_white",
                  "beat_flash": "beat_flash" if dark else "beat_dim",
                  "beat_dim": "beat_dim" if not dark else "beat_flash",
                  "flash_freeze": "flash_freeze"}[fx]
            why.append("暗画面配亮闪" if dark else "亮画面配暗闪")

        # ---- 调色：艳丽素材换低饱和组，避免叠加过艳 ----
        pool = GRADE_LOW_SAT if f.get("colorfulness", 0.5) > 0.72 else grade_pool
        if pool is GRADE_LOW_SAT:
            why.append("素材本身艳丽，调色降饱和")
        grade = _pick(rng, pool, prev_grade)

        # ---- 速度 / 段内变速 ----
        speed, ramp = 1.0, None
        if f["kind"] == "video":
            r = rng.random()
            if style == "cinematic":
                speed = 0.62 if r < 0.35 else 1.0
            elif style == "dynamic":
                speed = 1.35 if r < 0.16 else (0.65 if r < 0.26 else 1.0)
            else:
                speed = 0.7 if r < 0.18 else (1.3 if r < 0.28 else 1.0)
            # 段内变速：build 段"先慢后快"，高潮/落定处"先快后慢"
            if not is_jump and f.get("motion", 0.0) < 0.6 and rng.random() < 0.26:
                if section in ("low", "mid"):
                    ramp = (0.6, 1.7)
                    why.append("段内变速·渐快 build")
                else:
                    ramp = (1.7, 0.6)
                    why.append("段内变速·渐慢落定")
                speed = 1.0

        plan.append({"move": move, "fx": fx, "grade": grade,
                     "transition": trans, "speed": speed, "ramp": ramp,
                     "tdur": tdur, "why": "；".join(why)})
        prev_move, prev_fx, prev_trans, prev_grade = move, fx, trans, grade
    return plan


def resolve_style(style: str, mood: str) -> str:
    if style in CAMERA_POOL:
        return style
    return {"energetic": "dynamic", "chill": "balanced",
            "calm": "cinematic"}.get(mood, "balanced")


def style_amp(style: str, intensity: float) -> float:
    base = {"dynamic": 1.15, "balanced": 0.85, "cinematic": 0.55}.get(style, 0.85)
    return max(0.2, min(1.8, base * intensity))
