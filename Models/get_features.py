import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import numpy as np
import librosa

N_MFCC = 40
SAMPLE_RATE = 16000
# ================== 全局变换器 (CPU上定义一次) ==================
MFCC_T = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_mels": N_MFCC}
)
DELTA_T = T.ComputeDeltas()
DELTA2_T = T.ComputeDeltas()


# ================== 特征提取函数 ==================
def extract_f0_librosa(waveform, sample_rate=SAMPLE_RATE):
    x = waveform.squeeze(0).numpy()
    f0, _, _ = librosa.pyin(x, fmin=50, fmax=500, sr=sample_rate)
    f0 = np.nan_to_num(f0)  # 将 NaN 转 0
    return torch.tensor(f0, dtype=torch.float32)


def extract_ser_features(waveform, sample_rate=SAMPLE_RATE):
    """提取适合语音情绪识别的特征"""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ---------- MFCC + Δ + ΔΔ ----------
    mfcc = MFCC_T(waveform).squeeze(0).T        # [Time, n_mfcc]
    delta = DELTA_T(mfcc.T).T
    delta2 = DELTA2_T(delta.T).T
    mfcc_full = torch.cat([mfcc, delta, delta2], dim=1)  # [T, feat_dim]

    # ---------- RMS & log Energy ----------
    frame_len = int(0.025 * sample_rate)
    hop_len = int(0.010 * sample_rate)
    if waveform.shape[1] < frame_len:
        waveform = F.pad(waveform, (0, frame_len - waveform.shape[1]))

    rms = torch.sqrt(torch.mean(waveform.unfold(1, frame_len, hop_len)**2, dim=1)).squeeze()
    log_e = torch.log1p(rms)

    # # ---------- 基频 F0 ----------
    # f0 = extract_f0_librosa(waveform, sample_rate)

    # ---------- 对齐时间维度 ----------
    t_len = mfcc_full.shape[0]

    def align(feat):
        if feat.shape[0] < 2:
            return feat.expand(t_len)
        else:
            return F.interpolate(feat.unsqueeze(0).unsqueeze(0), size=t_len, mode='linear', align_corners=False).squeeze()

    rms = align(rms)
    log_e = align(log_e)
    # f0 = align(f0)

    # ---------- 拼接特征 ----------
    # features = torch.cat([rms.unsqueeze(1), log_e.unsqueeze(1), f0.unsqueeze(1), mfcc_full], dim=1)
    features = torch.cat([rms.unsqueeze(1), log_e.unsqueeze(1), mfcc_full], dim=1)
    return features.cpu()

