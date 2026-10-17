"""剪辑手法库（shot language）+ 需求-供给评分匹配编排。

手法分四层：
  - 镜头运动 camera：推拉 / 平移 / 漂移 / 摇摆 / 呼吸 / 冲击缩放 / 对角 Ken Burns /
                    心跳脉冲 / 缩放脉冲 / 抖动（超采 + zoompan 逐帧驱动）
  - 卡点特效 fx：白闪 / 黑闪 / 红闪 / 每拍亮度脉冲 / 饱和度脉冲 / 频闪 /
                 镜像分屏 / 色差 / 颗粒 / 翻转 / 电影遮幅 / 复古胶片
  - 调色 grade：按情绪多套预设 + 低饱和平衡组（艳丽素材自动降饱和）
  - 转场 transition：soft / mid / hard 三档，按张力与乐句结构选取

编排机制（与旧版的本质区别）：
  旧版是「随机采样 + 规则排雷」——先 rng.choice 再删掉违规项，同一素材换个
  seed 就剪出完全不同的片子，"素材感知"只体现在排除法上。
  新版是「评分匹配」——每个候选手法按 `风格偏好 + 构图安全 + 张力贴合 +
  乐句角色 + 明暗对比 + 运动兼容` 六项打分，确定性取最高分；rng 仅用于
  分数完全相同的平局（seed 仍可复现）。同一素材必然得到同一套剪辑方案，
  素材变了方案才变——这才是"根据素材选择剪辑方式"。
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

# 各风格下镜头运动的基础偏好分（评分制的先验；不在表内的手法为 0）
CAM_BIAS = {
    "dynamic": {"zoom_punch": .30, "crash_zoom": .25, "punch_pan": .22,
                "shake": .18, "heartbeat": .15, "pan_left": .12, "pan_right": .12,
                "zoom_in": .10, "zoom_out": .10, "ken_diag": .08, "sway": .08,
                "drift": .05, "breath": .02, "static": .0},
    "balanced": {"zoom_in": .22, "zoom_out": .20, "pan_left": .15,
                 "pan_right": .15, "ken_diag": .15, "drift": .12, "breath": .10,
                 "zoom_punch": .10, "sway": .06, "static": .0},
    "cinematic": {"zoom_in": .20, "zoom_out": .20, "drift": .18, "breath": .15,
                  "ken_diag": .15, "pan_up": .12, "pan_down": .12, "static": .05},
}

# 图片素材不适合的运动
_IMAGE_SOFT = {"zoom_in", "zoom_out", "drift", "pan_up", "pan_down",
               "pan_left", "pan_right", "zoom_punch", "ken_diag",
               "breath", "static"}

# 疑似特写的镜头（构图边缘集中度高）只许用温和手法
_SUBJECT_SAFE = {"zoom_in", "zoom_out", "drift", "breath", "ken_diag",
                 "static", "crash_zoom"}

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

# 疑似特写镜头禁用的破坏性特效
_BAD_ON_CLOSE = {"split_mirror", "hflip", "strobe", "chroma", "glitch",
                 "pip_card", "flash_freeze"}

# 各风格下特效的基础偏好分
FX_BIAS = {
    "dynamic": {"beat_flash": .25, "flash_white": .20, "flash_black": .18,
                "beat_pulse": .18, "strobe": .15, "glitch": .15, "echo": .12,
                "flash_freeze": .12, "chroma": .10, "hflip": .08,
                "split_mirror": .08, "none": .05},
    "balanced": {"none": .22, "beat_flash": .15, "grain": .12, "beat_pulse": .12,
                 "flash_black": .10, "letterbox": .08, "warm_leak": .08,
                 "echo": .06, "pip_card": .05},
    "cinematic": {"none": .28, "letterbox": .20, "old_film": .18, "grain": .15,
                  "warm_leak": .12, "flash_black": .08, "echo": .05},
}

# 氛围型特效（低张力段落用）
_FX_AMBIENT = {"none", "letterbox", "old_film", "warm_leak", "grain"}
# 节拍型特效（高张力段落用）
_FX_BEAT = {"beat_flash", "beat_pulse", "flash_white", "flash_black",
            "strobe", "echo", "glitch", "flash_freeze"}


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

# 方向性转场：可按素材像素变化方向顺势选择（匹配剪辑）
DIRECTIONAL_TRANS = {
    "left":  ["slideleft", "wipeleft", "coverleft", "revealleft", "smoothleft",
              "hlslice"],
    "right": ["slideright", "wiperight", "coverright", "revealright",
              "smoothright", "hrslice"],
    "up":    ["slideup", "wipeup", "smoothup"],
    "down":  ["slidedown", "wipedown", "smoothdown"],
}

# 各情绪的基准转场时长（秒），实际按张力缩放
BASE_TDUR = {"calm": 0.62, "chill": 0.50, "energetic": 0.30}
HARD_CUT_DUR = 0.05          # 跳切：极短转场≈硬切


def tier_of(tension: float) -> str:
    if tension > 0.72:
        return "hard"
    if tension > 0.45:
        return "mid"
    return "soft"


_TIER_ORDER = {"soft": 0, "mid": 1, "hard": 2}


def _argmax(scores: dict[str, float], rng: random.Random) -> str:
    """确定性取最高分；完全并列时用 rng 打破（seed 仍可复现）。"""
    best = max(scores.values())
    top = [k for k, v in scores.items() if abs(v - best) < 1e-9]
    if len(top) == 1:
        return top[0]
    return top[rng.randrange(len(top))]


def _pick_rot(pool: list[str], i: int, prev: str) -> str:
    """确定性轮换（调色/转场等同分池）：按段号轮转且不与上一段重复。"""
    cand = [x for x in pool if x != prev] or pool
    return cand[i % len(cand)]


def plan_shots(feats: list[dict], mood: str, style: str,
               rng: random.Random,
               camera_motion: str = "off") -> list[dict]:
    """需求-供给评分匹配编排：逐段决定「怎么剪」，并给出决策理由与得分。

    feats[i] 可含：
      energy        该段所处音乐瞬时能量 0~1（保留字段，兼容旧报告）
      tension       该段目标张力 0~1（乐句级慢变量，决策主信号）
      tension_grad  张力梯度（>0 build / <0 回落）
      phrase_start  是否乐句首小节
      brightness    画面明暗 0~1
      colorfulness  画面色彩丰富度 0~1
      subject_center 构图边缘集中度 0~1（启发式）
      quality       窗口画质分 0~1（清晰度×曝光）
      motion        镜头运动强度 0~1
      motion_dir    像素变化方向（<0 左 / >0 右）
      shot_type     近似景别 close / medium / wide（启发式）
      section       音乐结构 low / mid / high
      boundary      是否落在音乐结构边界
      jump          是否与上一段同镜头相邻窗口（可做跳切）
      kind          video / image

    返回每段：{move, fx, grade, transition, speed, ramp, tdur, why}
    每个 why 附带最高分因子的得分，便于审计"这个决策是数据驱动还是随机"。

    camera_motion 三档：off 完全稳定 / subtle 仅温和缓推缓拉 / full 完整运镜库。
    特效、转场、调色、变速任何档位都照常生效。
    """
    cam_bias = CAM_BIAS.get(style, CAM_BIAS["balanced"])
    fx_bias = FX_BIAS.get(style, FX_BIAS["balanced"])
    grade_pool = GRADE_POOL.get(mood, GRADE_POOL["chill"])
    base_tdur = BASE_TDUR.get(mood, 0.5)

    plan: list[dict] = []
    prev_move = prev_fx = prev_trans = prev_grade = ""
    REPEAT_PENALTY = 0.12        # 相邻段同手法的确定性扣分（多样性约束）
    for i, f in enumerate(feats):
        tension = float(f.get("tension", f.get("energy", 0.5)))
        grad = float(f.get("tension_grad", 0.0))
        section = f.get("section", "mid")
        shot_type = f.get("shot_type", "medium")
        kind = f.get("kind", "video")
        md = float(f.get("motion_dir", 0.0))
        motion = float(f.get("motion", 0.0))
        bright = float(f.get("brightness", 0.5))
        colorf = float(f.get("colorfulness", 0.5))
        quality = float(f.get("quality", 0.5))
        is_jump = bool(f.get("jump"))
        phrase_start = bool(f.get("phrase_start"))
        why: list[str] = []

        # ==================== 转场 ====================
        tier = tier_of(tension)
        if f.get("boundary"):
            tier = "hard"
            why.append("音乐结构边界→硬转场")
        elif phrase_start and tier == "soft":
            tier = "mid"
            why.append("乐句开头抬一档转场")
        if is_jump:
            trans, tdur = "fade", HARD_CUT_DUR
            why.append("同镜头跳切")
        else:
            if abs(md) > 0.25 and tier != "soft":
                key = "right" if md > 0 else "left"
                dir_pool = [t for t in DIRECTIONAL_TRANS[key]
                            if t in TRANS_TIERS[tier]] or DIRECTIONAL_TRANS[key]
                trans = _pick_rot(dir_pool, i, prev_trans)
                why.append("转场顺着画面" + ("右" if md > 0 else "左") + "移")
            else:
                trans = _pick_rot(TRANS_TIERS[tier], i, prev_trans)
            # 张力越高转场越短促（0.9~0.5 倍缩放），硬档本身更短
            factor = {"hard": 0.62, "mid": 1.0, "soft": 1.2}[tier]
            tdur = float(min(0.9, max(0.05,
                                      base_tdur * factor * (1.1 - 0.4 * tension))))

        # ==================== 镜头运动（评分制）====================
        if camera_motion in (False, "off", "none"):
            move = "breath" if kind == "image" else "static"
        else:
            pool = list(CAMERA_POOL.get(style, CAMERA_POOL["balanced"]))
            if camera_motion == "subtle":
                pool = [m for m in pool if m in GENTLE_MOVES] or ["static"]
            scores: dict[str, float] = {}
            for mv in pool:
                if shot_type == "close" and mv not in _SUBJECT_SAFE:
                    continue                    # 疑似特写：禁用破坏性运镜
                if kind == "image" and mv not in _IMAGE_SOFT:
                    continue
                s = cam_bias.get(mv, 0.0)
                # 张力贴合：高张力配冲击/脉冲，低张力配呼吸/漂移
                if tension >= 0.7:
                    s += {"crash_zoom": .22, "zoom_punch": .20,
                          "heartbeat": .15, "punch_pan": .10}.get(mv, 0.0)
                elif tension <= 0.3:
                    s += {"breath": .20, "drift": .18, "ken_diag": .10,
                          "zoom_out": .08}.get(mv, 0.0)
                else:
                    s += {"zoom_in": .10, "zoom_out": .10,
                          "ken_diag": .10}.get(mv, 0.0)
                # 乐句首小节：冲击缩放/推进开场感更强
                if phrase_start:
                    s += {"crash_zoom": .15, "zoom_in": .10}.get(mv, 0.0)
                # 镜头本身动得猛：不再叠加抖动类（避免画面过花）
                if motion > 0.60:
                    s -= {"shake": .30, "punch_pan": .25,
                          "zoom_punch": .20}.get(mv, 0.0)
                    s += {"static": .12, "drift": .06}.get(mv, 0.0)
                # 像素变化有方向：平移顺势借力
                if md > 0.25 and mv == "pan_right":
                    s += .15
                elif md < -0.25 and mv == "pan_left":
                    s += .15
                # 画质差的窗口：少动比乱动安全
                if quality < 0.3 and mv not in ("static", "breath"):
                    s -= .10
                # 与上一段同手法的确定性扣分
                if mv == prev_move:
                    s -= REPEAT_PENALTY
                scores[mv] = s
            if not scores:                      # 全被规则排除时的兜底
                scores = {"static": 1.0}
            move = _argmax(scores, rng)
            top_s = max(scores.values())
            if shot_type == "close":
                why.append("疑似特写·保护构图")
            if tension >= 0.7 and move in ("crash_zoom", "zoom_punch"):
                why.append(f"高张力{tension:.2f}→冲击运镜")
            elif tension <= 0.3 and move in ("breath", "drift"):
                why.append(f"低张力{tension:.2f}→缓动")
            if motion > 0.60 and move in ("static", "drift", "breath"):
                why.append("镜头本身在动，运镜收敛")

        # ==================== 卡点特效（评分制）====================
        fx_pool = list(FX_POOL.get(style, FX_POOL["balanced"]))
        scores = {}
        for fx in set(fx_pool):                 # 池内有重复权重项，去重后打分
            if shot_type == "close" and fx in _BAD_ON_CLOSE:
                continue
            if kind == "image" and fx in ("split_mirror", "strobe", "hflip",
                                          "glitch", "pip_card"):
                continue
            s = fx_bias.get(fx, 0.0)
            # 结构贴合：低张力段偏氛围型，高张力段偏节拍型
            if section == "low" or tension <= 0.3:
                s += .18 if fx in _FX_AMBIENT else 0.0
            if section == "high" and tension >= 0.6:
                s += .18 if fx in _FX_BEAT else 0.0
            # 明暗对比：暗画面配亮闪、亮画面配暗闪，才看得见
            if fx in ("flash_white", "beat_flash"):
                s += .25 if bright < 0.45 else -.15
            elif fx in ("flash_black", "beat_dim"):
                s += .25 if bright >= 0.45 else -.15
            # 极强张力才上频闪/故障这类重特效
            if fx in ("strobe", "glitch", "flash_freeze"):
                s += .20 * max(0.0, tension - 0.6) / 0.4
            # 镜头本身在动：重特效收敛
            if motion > 0.60:
                s -= {"strobe": .25, "glitch": .25, "echo": .20}.get(fx, 0.0)
            # 与上一段同特效的确定性扣分
            if fx == prev_fx:
                s -= REPEAT_PENALTY
            scores[fx] = s
        if not scores:
            scores = {"none": 1.0}
        fx = _argmax(scores, rng)
        if shot_type == "close":
            why.append("疑似特写·不用破坏性特效")
        if section == "low" and fx in _FX_AMBIENT and fx != "none":
            why.append("低能量段配氛围特效")
        if section == "high" and tension >= 0.6 and fx in _FX_BEAT:
            why.append("高潮段配节拍特效")
        if fx in ("flash_white", "beat_flash") and bright < 0.45:
            why.append("暗画面配亮闪")
        if fx in ("flash_black", "beat_dim") and bright >= 0.45:
            why.append("亮画面配暗闪")

        # ==================== 调色 ====================
        pool = GRADE_LOW_SAT if colorf > 0.72 else grade_pool
        if pool is GRADE_LOW_SAT:
            why.append("素材本身艳丽，调色降饱和")
        grade = _pick_rot(pool, i, prev_grade)

        # ==================== 速度 / 段内变速（张力梯度驱动）====================
        speed, ramp = 1.0, None
        if kind == "video" and not is_jump and motion < 0.6:
            if grad > 0.03 and tension < 0.8:
                ramp = (0.6, 1.7)               # build：先慢后快，跟随张力爬坡
                why.append(f"段内变速·渐快 build(张力梯度+{grad:.2f})")
            elif grad < -0.03 and tension > 0.5:
                ramp = (1.7, 0.6)               # 落定：先快后慢，跟随张力回落
                why.append(f"段内变速·渐慢落定(张力梯度{grad:.2f})")
            elif style == "cinematic" and quality > 0.65 and tension < 0.5:
                speed = 0.62                    # 好素材 + 低张力 → 慢动作展示
                why.append(f"画质{quality:.2f}+低张力→慢动作展示")
            elif style == "dynamic" and tension > 0.75:
                speed = 1.35                    # 高张力快放压缩时间
                why.append(f"高张力{tension:.2f}→快放")

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
