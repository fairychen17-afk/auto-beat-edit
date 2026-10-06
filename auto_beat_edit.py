"""
游戏CG自动卡点剪辑 v12 究极版
在炸裂版基础上新增：
- 鱼眼脉冲
- 像素化故障
- 镜头光晕
适配 moviepy 2.x
"""

import os
import numpy as np
import librosa
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

FOLDER = r"C:\Users\甘雨\Desktop\虚照"
AUDIO_FILE = "@饺子WTF创作的原声_7656434362674048997(1).mp3"
OUTPUT_FILE = os.path.join(FOLDER, "虚照_卡点混剪_究极版.mp4")

AUDIO_START = 54.0
AUDIO_DURATION = 10.0
TARGET_RESOLUTION = (720, 960)
CLIP_ORDER = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]

MIN_SPEED = 0.8
MAX_SPEED = 1.3

FILTER_CONTRAST = 1.3
FILTER_SATURATION = 1.35
FILTER_BRIGHTNESS = 1.0
FILTER_VIGNETTE = 0.6
FILTER_TEAL_ORANGE = True
FILTER_FILM_GRAIN = 0.1

PULSE_INTENSITY = 0.08
SHAKE_INTENSITY = 3.0
ZOOM_SPEED = 0.12
GLITCH_INTENSITY = 6.0
MOTION_BLUR_INTENSITY = 0.4
HUE_SHIFT_AMOUNT = 15.0

# 新增究极效果
FISHEYE_ON = True
PIXEL_ON = True
FLARE_ON = True


def detect_beats(audio_path, start_time=0, duration=None):
    y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times = np.array(beat_times).flatten()
    tempo = float(tempo)
    
    half_beats = []
    for i in range(len(beat_times) - 1):
        half_beats.append(beat_times[i])
        half_beats.append((beat_times[i] + beat_times[i+1]) / 2)
    half_beats.append(beat_times[-1])
    half_beats = np.array(half_beats)
    
    print(f"  BPM: {tempo:.1f}, 整拍: {len(beat_times)}个")
    return beat_times, half_beats, tempo


def prepare_video_clips(video_files, target_resolution=None):
    clips = []
    for i, vf in enumerate(video_files):
        try:
            clip = VideoFileClip(vf)
            if target_resolution:
                tw, th = target_resolution
                cw, ch = clip.size
                scale = max(tw / cw, th / ch) * 1.2
                clip = clip.resized(scale)
                new_w, new_h = clip.size
                xc, yc = new_w / 2, new_h / 2
                clip = clip.cropped(
                    x1=int(xc - tw/2), y1=int(yc - th/2),
                    x2=int(xc + tw/2), y2=int(yc + th/2)
                )
            clip = clip.without_audio()
            clips.append(clip)
            print(f"  [{i+1}] {os.path.basename(vf)} - {clip.duration:.2f}s")
        except Exception as e:
            print(f"  警告: {os.path.basename(vf)} - {e}")
    return clips


def snap_to_grid(time, grid_times, min_gap=0.0):
    idx = np.argmin(np.abs(grid_times - time))
    snapped = grid_times[idx]
    if snapped < min_gap:
        later = grid_times[grid_times >= min_gap]
        if len(later) > 0:
            snapped = later[0]
    return float(snapped)


def fisheye(img, strength):
    """鱼眼畸变，输出尺寸不变"""
    h, w = img.shape[:2]
    dst_y, dst_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    nx = (dst_x - w/2) / (w/2)
    ny = (dst_y - h/2) / (h/2)
    r = np.sqrt(nx**2 + ny**2)
    r = np.clip(r, 0, 1)
    
    r_new = r * (1 + strength * r * r)
    r_safe = np.where(r > 0.001, r, 0.001)
    scale = r_new / r_safe
    
    src_x = w/2 + nx * scale * (w/2)
    src_y = h/2 + ny * scale * (h/2)
    
    src_x = np.clip(src_x, 0, w - 1).astype(int)
    src_y = np.clip(src_y, 0, h - 1).astype(int)
    
    return img[src_y, src_x]


def pixelate(img, block_size):
    h, w = img.shape[:2]
    if block_size < 2:
        return img
    sh = max(1, h // block_size)
    sw = max(1, w // block_size)
    yi = (np.arange(sh) * block_size).astype(int)
    xi = (np.arange(sw) * block_size).astype(int)
    small = img[np.ix_(yi, xi)]
    yr = np.repeat(np.arange(sh), block_size)[:h]
    xr = np.repeat(np.arange(sw), block_size)[:w]
    return small[np.ix_(yr, xr)]


def add_flare(img, intensity):
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.35
    y_arr, x_arr = np.ogrid[:h, :w]
    dist = np.sqrt((x_arr - cx)**2 * 1.5 + (y_arr - cy)**2)
    flare = np.exp(-dist**2 / (2 * 100**2)) * intensity
    flare += np.exp(-((x_arr - cx*0.7)**2 + (y_arr - cy*1.3)**2) / (2 * 45**2)) * intensity * 0.4
    flare_3d = np.stack([flare, flare*0.85, flare*0.6], axis=2)
    result = img.astype(np.float32) + flare_3d * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def add_effects(seg, seg_start_time, beat_times, seg_index, is_fast):
    seg_dur = seg.duration
    beat_dur_local = 60.0 / 172.3
    rng = np.random.RandomState(777 + seg_index * 19)
    tw, th = TARGET_RESOLUTION
    
    def process_frame(get_frame, t):
        frame = get_frame(t)
        f = frame.astype(np.float32)
        global_t = seg_start_time + t
        
        # 节拍信息
        nearest_beat_dist = np.min(np.abs(beat_times - global_t))
        is_near_beat = nearest_beat_dist < beat_dur_local * 0.3
        pulse_phase = (nearest_beat_dist / beat_dur_local) * np.pi
        pulse_amount = np.cos(pulse_phase) * 0.5 + 0.5
        
        fh, fw = f.shape[:2]
        
        # ===== 1. 缩放脉冲 + 推镜 + 震动 =====
        pulse_zoom = 1.0 + pulse_amount * PULSE_INTENSITY
        if not is_fast and seg_dur > 0.5:
            push_zoom = 1.0 + (t / seg_dur) * ZOOM_SPEED * min(seg_dur, 3.0)
        else:
            push_zoom = 1.0
        total_zoom = pulse_zoom * push_zoom
        
        shake_x = shake_y = 0.0
        if is_fast and is_near_beat:
            strength = SHAKE_INTENSITY * (1 - nearest_beat_dist / (beat_dur_local * 0.3))
            angle = rng.uniform(0, 2 * np.pi)
            shake_x = np.cos(angle) * strength
            shake_y = np.sin(angle) * strength
        
        if abs(total_zoom - 1.0) > 0.001 or abs(shake_x) > 0.1 or abs(shake_y) > 0.1:
            crop_w = int(fw / total_zoom)
            crop_h = int(fh / total_zoom)
            cx_pix = fw / 2 + shake_x * total_zoom
            cy_pix = fh / 2 + shake_y * total_zoom
            x1 = int(max(0, min(cx_pix - crop_w / 2, fw - crop_w)))
            y1 = int(max(0, min(cy_pix - crop_h / 2, fh - crop_h)))
            x2, y2 = x1 + crop_w, y1 + crop_h
            cropped = f[y1:y2, x1:x2, :]
            rows = np.linspace(0, crop_h - 1, fh).astype(int)
            cols = np.linspace(0, crop_w - 1, fw).astype(int)
            f = cropped[rows[:, np.newaxis], cols[np.newaxis, :], :]
        
        f_uint = np.clip(f, 0, 255).astype(np.uint8)
        
        # ===== 2. 鱼眼脉冲 =====
        if FISHEYE_ON and pulse_amount > 0.3:
            fe_str = pulse_amount * 0.12
            f_uint = fisheye(f_uint, fe_str)
            f = f_uint.astype(np.float32)
        
        # ===== 3. 像素化故障 =====
        if PIXEL_ON and is_fast and rng.random() < 0.15:
            block = rng.randint(6, 15)
            f_uint = pixelate(f_uint, block)
            f = f_uint.astype(np.float32)
        
        # ===== 4. Glitch色差 =====
        if is_fast and is_near_beat:
            ga = GLITCH_INTENSITY * (1 - nearest_beat_dist / (beat_dur_local * 0.3))
            if ga > 0.5:
                off = int(ga)
                f[:,:,0] = np.roll(f[:,:,0], -off, axis=1)
                f[:,:,2] = np.roll(f[:,:,2], off, axis=1)
                go = rng.randint(-max(1,off//2), max(2,off//2+1))
                f[:,:,1] = np.roll(f[:,:,1], go, axis=0)
        
        # ===== 5. 色相偏移 =====
        if is_fast:
            hs = HUE_SHIFT_AMOUNT * pulse_amount * (1 if seg_index % 2 == 0 else -1)
            if abs(hs) > 1.0:
                sr = np.radians(hs)
                cr = np.cos(sr)
                sh = np.sin(sr)
                r, g, b = f[:,:,0].copy(), f[:,:,1].copy(), f[:,:,2].copy()
                f[:,:,0] = r*(0.8+0.2*cr) + g*0.1*sh + b*0.1*-sh
                f[:,:,1] = r*0.1*-sh + g*(0.8+0.2*cr) + b*0.1*sh
                f[:,:,2] = r*0.1*sh + g*0.1*-sh + b*(0.8+0.2*cr)
        
        # ===== 6. 运动模糊 =====
        if is_fast and t < seg_dur * 0.4:
            mb = MOTION_BLUR_INTENSITY * (1 - t / (seg_dur * 0.4))
            bs = int(mb * 10)
            if bs > 1:
                blurred = f.copy()
                for i in range(1, bs + 1):
                    w_i = 1.0 / (i + 1)
                    shifted = np.roll(f, i, axis=1)
                    blurred = blurred * (1 - w_i * 0.3) + shifted * w_i * 0.3
                f = blurred
        
        # ===== 7. 调色 =====
        f = f * FILTER_BRIGHTNESS
        f = 128.0 + (f - 128.0) * FILTER_CONTRAST
        
        if FILTER_TEAL_ORANGE:
            lum = 0.299*f[:,:,0] + 0.587*f[:,:,1] + 0.114*f[:,:,2]
            hl = np.clip((lum - 100) / 100, 0, 1)
            sh_mask = 1.0 - hl
            f[:,:,0] += hl * 20
            f[:,:,1] += hl * 10
            f[:,:,2] -= hl * 14
            f[:,:,0] -= sh_mask * 9.6
            f[:,:,1] += sh_mask * 6.4
            f[:,:,2] += sh_mask * 16
        
        gray = 0.299*f[:,:,0] + 0.587*f[:,:,1] + 0.114*f[:,:,2]
        for c in range(3):
            f[:,:,c] = gray + (f[:,:,c] - gray) * FILTER_SATURATION
        
        # ===== 8. 镜头光晕 =====
        if FLARE_ON and pulse_amount > 0.5:
            f_uint = np.clip(f, 0, 255).astype(np.uint8)
            f_uint = add_flare(f_uint, pulse_amount * 0.6)
            f = f_uint.astype(np.float32)
        
        # ===== 9. 胶片颗粒 =====
        if FILTER_FILM_GRAIN > 0:
            grain = rng.normal(0, FILTER_FILM_GRAIN * 255, f.shape).astype(np.float32)
            f = f + grain
        
        # ===== 10. 撕裂转场 =====
        td = 0.05
        fh2, fw2 = f.shape[:2]
        if t < td and is_fast:
            ta = int(15 * (1 - t / td))
            mid = fh2 // 2
            f[:mid, :, :] = np.roll(f[:mid, :, :], ta, axis=1)
            f[mid:, :, :] = np.roll(f[mid:, :, :], -ta, axis=1)
        if t > seg_dur - td and is_fast:
            ta = int(15 * (1 - (seg_dur - t) / td))
            mid = fh2 // 2
            f[:mid, :, :] = np.roll(f[:mid, :, :], -ta, axis=1)
            f[mid:, :, :] = np.roll(f[mid:, :, :], ta, axis=1)
        
        # 暗角
        yv, xv = np.ogrid[:fh2, :fw2]
        dv = np.sqrt((xv - fw2/2)**2 + (yv - fh2/2)**2)
        mv = np.sqrt((fw2/2)**2 + (fh2/2)**2)
        vign = 1.0 - (dv / mv)**2 * FILTER_VIGNETTE
        vign3 = np.stack([vign, vign, vign], axis=2)
        f = f * vign3.astype(np.float32)
        
        f = np.clip(f, 0, 255)
        
        # 确保输出尺寸精确
        result = f.astype(np.uint8)
        if result.shape[0] != th or result.shape[1] != tw:
            new_result = np.zeros((th, tw, 3), dtype=np.uint8)
            sh2 = min(result.shape[0], th)
            sw2 = min(result.shape[1], tw)
            new_result[:sh2, :sw2, :] = result[:sh2, :sw2, :]
            result = new_result
        
        return result
    
    return seg.transform(process_frame, apply_to=['video'])


def create_edit(beat_times, half_beats, video_clips, tempo, target_dur=10.0):
    beat_dur = 60.0 / tempo
    half_beat_dur = beat_dur / 2
    print(f"\n生成卡点剪辑（究极版）...")
    print(f"  BPM: {tempo:.1f}")
    
    beat_pattern = [
        (0.5, "开头闪", "half"),
        (4.0, "长", "full"),
        (0.5, "短闪", "half"),
        (1.5, "中", "full"),
        (6.0, "铺垫最长", "full"),
        (0.5, "快切", "half"),
        (0.33, "快切", "half"),
        (0.5, "快切", "half"),
        (0.33, "快切", "half"),
        (0.66, "快切", "half"),
        (0.5, "快切", "half"),
        (0.5, "快切", "half"),
        (0.5, "快切", "half"),
        (0.66, "快切", "half"),
        (1.5, "快切小停顿", "full"),
        (0.33, "快切", "half"),
        (5.0, "结尾长", "full"),
        (0.5, "结尾短闪", "half"),
        (3.5, "结尾中长", "full"),
        (0.5, "收尾", "half"),
    ]
    
    total_pattern_beats = sum(p[0] for p in beat_pattern)
    total_beats = target_dur / beat_dur
    scale = total_beats / total_pattern_beats
    
    boundaries = [0.0]
    cum_beats = 0.0
    for beats, label, snap_type in beat_pattern:
        cum_beats += beats * scale
        target_time = cum_beats * beat_dur
        if target_time > target_dur + 0.2:
            break
        grid = half_beats if snap_type == "half" else beat_times
        min_gap = boundaries[-1] + (half_beat_dur if snap_type == "half" else beat_dur) * 0.5
        snapped = snap_to_grid(target_time, grid, min_gap)
        if snapped > target_dur + 0.1:
            break
        boundaries.append(snapped)
    
    if boundaries[-1] < target_dur - 0.3:
        end_snapped = snap_to_grid(target_dur, beat_times, boundaries[-1] + beat_dur * 0.5)
        if end_snapped <= target_dur + 0.2:
            boundaries.append(end_snapped)
        else:
            boundaries.append(target_dur)
    
    n_segs = len(boundaries) - 1
    seg_durs = [boundaries[i+1] - boundaries[i] for i in range(n_segs)]
    print(f"  共 {n_segs} 段")
    
    clip_assign = []
    clip_assign.extend([0, 0])
    clip_assign.extend([1, 1, 1])
    clip_assign.extend([2] * 12)
    clip_assign.extend([3] * 3)
    while len(clip_assign) < n_segs:
        clip_assign.append(2)
    clip_assign = clip_assign[:n_segs]
    
    all_segments = []
    clip_positions = [0.0] * len(video_clips)
    clip_positions[0] = video_clips[0].duration * 0.02
    clip_positions[1] = video_clips[1].duration * 0.08
    clip_positions[2] = video_clips[2].duration * 0.03
    clip_positions[3] = video_clips[3].duration * 0.2
    
    for si in range(n_segs):
        ci = clip_assign[si]
        clip = video_clips[ci]
        target_dur_seg = seg_durs[si]
        start_pos = clip_positions[ci]
        
        if start_pos >= clip.duration - 0.02:
            start_pos = max(0, clip.duration - 0.1)
        
        remaining = clip.duration - start_pos
        avg_speed = 1.06 + (si % 5) * 0.015
        avg_speed = min(avg_speed, MAX_SPEED)
        
        if remaining < target_dur_seg * MIN_SPEED:
            avg_speed = max(remaining / target_dur_seg, MIN_SPEED)
        
        source_dur = target_dur_seg * avg_speed
        if start_pos + source_dur > clip.duration:
            source_dur = clip.duration - start_pos
            avg_speed = source_dur / target_dur_seg
            avg_speed = max(avg_speed, MIN_SPEED)
        
        seg = clip.subclipped(start_pos, start_pos + source_dur)
        clip_positions[ci] = start_pos + source_dur
        
        actual_speed = seg.duration / target_dur_seg
        actual_speed = np.clip(actual_speed, MIN_SPEED, MAX_SPEED)
        seg = seg.with_speed_scaled(1.0 / actual_speed)
        
        is_fast = (target_dur_seg / beat_dur) < 0.8
        seg = add_effects(seg, boundaries[si], beat_times, si, is_fast)
        all_segments.append(seg)
        
        label = beat_pattern[si][1] if si < len(beat_pattern) else "?"
        fast_mark = "⚡" if is_fast else "  "
        print(f"    {fast_mark} 片段{si+1:2d} [素材{ci+1}|{label:6s}]: {seg.duration:.3f}s, {actual_speed:.2f}x")
    
    final = concatenate_videoclips(all_segments, method="compose")
    print(f"\n  拼接后: {final.duration:.3f}s")
    if final.duration > target_dur + 0.05:
        final = final.subclipped(0, target_dur)
    return final


def main():
    print("=" * 60)
    print(f"  游戏CG自动卡点剪辑 v12 (究极版)")
    print("=" * 60)
    
    video_files = []
    for fname in CLIP_ORDER:
        fp = os.path.join(FOLDER, fname)
        if os.path.exists(fp):
            video_files.append(fp)
    
    print(f"\n素材: {len(video_files)} 个")
    if not video_files:
        return
    
    audio_path = os.path.join(FOLDER, AUDIO_FILE)
    if not os.path.exists(audio_path):
        mp3s = [f for f in os.listdir(FOLDER) if f.endswith('.mp3')]
        if mp3s:
            audio_path = os.path.join(FOLDER, mp3s[0])
        else:
            return
    
    print(f"\n分析音乐（从{AUDIO_START}s开始）...")
    beat_times, half_beats, tempo = detect_beats(audio_path, AUDIO_START, AUDIO_DURATION + 2)
    
    print(f"\n加载素材...")
    video_clips = prepare_video_clips(video_files, TARGET_RESOLUTION)
    
    final = create_edit(beat_times, half_beats, video_clips, tempo, AUDIO_DURATION)
    
    print(f"\n加音乐...")
    full_audio = AudioFileClip(audio_path)
    video_dur = final.duration
    audio_end = min(AUDIO_START + video_dur, full_audio.duration)
    bgm = full_audio.subclipped(AUDIO_START, audio_end)
    if video_dur > bgm.duration:
        final = final.subclipped(0, bgm.duration)
    final = final.with_audio(bgm)
    
    print(f"\n导出中...")
    final.write_videofile(
        OUTPUT_FILE, fps=30, codec="libx264",
        audio_codec="aac", bitrate="12000k",
        preset="medium", threads=4
    )
    
    final.close()
    full_audio.close()
    for c in video_clips:
        c.close()
    
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✅ 完成！")
    print(f"   文件: {OUTPUT_FILE}")
    print(f"   大小: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
