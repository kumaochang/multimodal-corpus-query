"""Audio helpers shared by the transcription and prosody stages."""
import numpy as np
import soundfile as sf

FRAME_S, HOP_S = 0.025, 0.010
# A frame is pause if its RMS is this far below the segment's 95th-percentile frame level, and a
# pause is a run of such frames at least MIN_PAUSE_S long. Set 2026-10-11 on Study 1 audio only
# (142 utterances of three clips): at -35 dB / 0.20 s, 6 of 142 utterances had any pause, so the
# pause features were near-constant; at -25 dB / 0.15 s, 38 of 142 do. Not tuned on Study 3.
PAUSE_DB = -25.0
MIN_PAUSE_S = 0.15

def load_segment(wav, start_s, end_s):
    info = sf.info(str(wav))
    sr = info.samplerate
    a, _ = sf.read(str(wav), start=max(0, int(start_s * sr)), stop=min(info.frames, int(end_s * sr)),
                   dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a[:, 0]
    return a, sr

def rms_db(a, sr):
    n, h = int(FRAME_S * sr), int(HOP_S * sr)
    if len(a) < n:
        return np.array([-120.0])
    idx = np.arange(0, len(a) - n + 1, h)
    r = np.sqrt(np.array([np.mean(a[i:i + n] ** 2) for i in idx]) + 1e-12)
    return 20 * np.log10(r)

def pauses(a, sr):
    """Internal pauses as (start_s, end_s) relative to the segment: runs of frames at least
    PAUSE_DB below the segment's 95th-percentile frame level, lasting MIN_PAUSE_S or more,
    excluding runs touching either edge."""
    db = rms_db(a, sr)
    thr = np.percentile(db, 95) + PAUSE_DB
    low = db < thr
    out, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if i > 0 and j < len(low) and (j - i) * HOP_S >= MIN_PAUSE_S:
                out.append((i * HOP_S, j * HOP_S + FRAME_S - HOP_S))
            i = j
        else:
            i += 1
    return out
