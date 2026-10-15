"""合成一首原创 EDM 卡点 BGM：126 BPM 四四底鼓 + 反拍 hi-hat + 锯齿贝斯 +
supersaw 和弦垫 + 高潮能量斜坡 + 结尾下扫。输出 32 秒 WAV。"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
BPM = 126.0
DUR = 32.0
BEAT = 60.0 / BPM                      # 0.476s
ROOT = 45                              # A1 (MIDI)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def saw(freq, t):
    return 2.0 * ((freq * t) % 1.0) - 1.0


def kick(n):
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 22) + 45
    phase = np.cumsum(f) / SR * 2 * np.pi
    return np.sin(phase) * np.exp(-t * 9)


def hat(n):
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    t = np.arange(n) / SR
    # 简单高通：差分
    hp = np.diff(noise, prepend=0.0)
    return hp * np.exp(-t * 60) * 0.5


def bass_note(freq, n, cut=0.7):
    t = np.arange(n) / SR
    env = np.minimum(1.0, np.exp(-t * 3) * 2)
    return saw(freq, t) * env * 0.6


def supersaw_chord(freqs, n, detune=0.15):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for dt in (-detune, 0.0, detune):
            out += saw(f * (1 + dt / 100), t)
    return out / (len(freqs) * 3)


PROG = [  # 和弦进行 Am - F - C - G（每 4 拍一个）
    [57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62],
]
BASSLINE = [33, 29, 24, 31]             # 对应根音 A1 F1 C1 G1


def main(out_path: Path):
    n_total = int(DUR * SR)
    y = np.zeros(n_total)
    n_beats = int(DUR / BEAT)

    # 能量斜坡：60% 处到高潮
    def energy(t_sec):
        x = t_sec / DUR
        return 0.45 + 0.55 * np.clip(1.0 - abs(x - 0.62) / 0.45, 0, 1)

    beat_times = np.arange(0, n_beats) * BEAT

    def place(sig, start):
        """安全叠加：越界自动截断。"""
        start = int(start)
        end = min(start + len(sig), n_total)
        if end > start:
            y[start:end] += sig[:end - start]

    for i, bt in enumerate(beat_times):
        e = energy(bt)
        s = int(bt * SR)
        bar = (i // 4) % 4
        # 底鼓：每拍
        k = kick(int(0.25 * SR)) * (0.9 if i % 4 == 0 else 0.75) * min(1.0, e + 0.3)
        place(k, s)
        # 反拍 hi-hat
        h = hat(int(0.06 * SR)) * e
        place(h, (bt + BEAT / 2) * SR)
        # 贝斯：每拍根音
        b = bass_note(midi(BASSLINE[bar] + 12), int(BEAT * SR * 0.9)) * e * 0.8
        place(b, s)
        # 和弦垫：每小节
        if i % 4 == 0:
            ch = supersaw_chord([midi(n + 12) for n in PROG[bar]],
                                int(BEAT * 4 * SR)) * 0.16 * e
            place(ch, s)
        # 高潮加 16 分碎拍 hat
        if e > 0.85:
            for sub in (0.25, 0.75):
                hh = hat(int(0.03 * SR)) * 0.4
                place(hh, (bt + BEAT * sub) * SR)

    # 结尾 1s 下扫 + 淡出
    fade = np.ones(n_total)
    tail = int(1.2 * SR)
    fade[-tail:] = np.linspace(1, 0, tail) ** 1.5
    y *= fade

    # 软限幅 + 归一
    y = np.tanh(y * 1.2)
    y /= np.max(np.abs(y)) + 1e-9
    pcm = (y * 0.92 * 32767).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("BGM 已生成:", out_path, f"{DUR}s @{BPM}BPM")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else Path(__file__).parent.parent / "bgm_lib" / "autocut_edm_126.wav")
