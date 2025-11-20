import cv2
import numpy as np
from keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from get_features import extract_ser_features
from moviepy import VideoFileClip
import json
import matplotlib.pyplot as plt
import av


SAMPLE_RATE = 16000
SEGMENT_DURATION = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("class2idx.json", "r") as f:
    class2idx = json.load(f)
idx2class = {v: k for k, v in class2idx.items()}

def detect_emotions_from_video(video_path):
    model = load_model('model_file.h5')

    video = cv2.VideoCapture(video_path)

    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    emotion_sums = np.zeros(len(labels_dict))
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))

            result = model.predict(reshaped, verbose=0)[0]  # 取出概率数组
            emotion_sums += result
            frame_count += 1

            label = np.argmax(result)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, labels_dict[label], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    avg_probabilities = emotion_sums / frame_count
    final_label_index = np.argmax(avg_probabilities)
    final_emotion = labels_dict[final_label_index]

    results = {labels_dict[i]: float(avg_probabilities[i]) for i in range(len(labels_dict))}

    return {"final_emotion": final_emotion, "probabilities": results}


# ================== Waveform Encoder (Conv + LSTM) ==================
class WaveformEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256, lstm_layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        x = self.conv(x)          # (B, H, T)
        x = x.transpose(1, 2)     # (B, T, H)
        x, _ = self.lstm(x)       # (B, T, 2H)
        x = self.proj(x)          # (B, T, H)
        return x


# ================== Feature LSTM Encoder ==================
class FeatureLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.proj_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x)    # (B, T, H)
        x, _ = self.lstm(x) # (B, T, 2H)
        x = self.proj_out(x)
        return self.norm(x) # (B, T, H)


# ================== Cross Attention Fusion ==================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_wave, feat_spec):
        # Q = feat_spec, K/V = feat_wave
        attn_out, _ = self.attn(feat_spec, feat_wave, feat_wave)
        return self.norm(feat_spec + attn_out)


# ================== Dual Stream Model (feat → wave → fusion) ==================
class DualStreamEmotionModel(nn.Module):
    def __init__(self, N_MFCC_dim, num_classes=7, hidden_dim=256):
        super().__init__()

        # Encoders
        self.feat_encoder = FeatureLSTMEncoder(N_MFCC_dim, hidden_dim=hidden_dim)
        self.wave_encoder = WaveformEncoder(hidden_dim=hidden_dim)

        # Fusion module
        self.fusion = CrossAttentionFusion(hidden_dim)

        # Post-fusion LSTM
        self.post_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.post_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, waveform, features):
        # ========== 1. encode features first ==========
        feat = self.feat_encoder(features)  # (B, T_feat, H)

        # ========== 2. encode raw waveform ==========
        wave = self.wave_encoder(waveform)  # (B, T_wave, H)

        # ========== 3. align waveform length to features ==========
        wave = wave.transpose(1, 2)                    # (B, H, T_wave)
        wave = F.adaptive_avg_pool1d(wave, feat.shape[1])
        wave = wave.transpose(1, 2)                    # (B, T_feat, H)

        # ========== 4. cross-attention fusion ==========
        fused = self.fusion(wave, feat)                # (B, T_feat, H)

        # ========== 5. post-fusion LSTM ==========
        fused_lstm, _ = self.post_lstm(fused)          # (B, T, 2H)
        fused_lstm = self.post_proj(fused_lstm)        # (B, T, H)

        # ========== 6. global average pooling + classifier ==========
        pooled = fused_lstm.mean(dim=1)
        return self.classifier(pooled)
# ================== Load model ==================
def load_ser_model(model_path, N_MFCC_dim, num_classes):
    model = DualStreamEmotionModel(N_MFCC_dim=N_MFCC_dim, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model_path = "emotion_lstm_best.pth"
N_MFCC_dim = 122
num_classes = len(class2idx)
model = load_ser_model(model_path, N_MFCC_dim, num_classes)

def predict(waveform_tensor):
    model.eval()

    feature_tensor = extract_ser_features(waveform_tensor)

    # 调整维度
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    if waveform_tensor.dim() == 2:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, 1, T_wave]
    if feature_tensor.dim() == 2:
        feature_tensor = feature_tensor.unsqueeze(0)  # [1, T_feat, N_MFCC]

    waveform_tensor = waveform_tensor.to(DEVICE)
    feature_tensor = feature_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(waveform_tensor, feature_tensor)  # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)  # [B, num_classes]

    probs = probs.squeeze(0).cpu().numpy()
    avg_probabilities = probs
    final_label_index = np.argmax(avg_probabilities)
    final_emotion = idx2class[final_label_index]

    results = {idx2class[i]: float(avg_probabilities[i]) for i in range(len(idx2class))}
    return {"final_emotion": final_emotion, "probabilities": results}


def load_audio_with_pyav(video_path):
    container = av.open(video_path)
    audio_stream = next(s for s in container.streams if s.type == 'audio')

    frames = []
    for frame in container.decode(audio_stream):
        frame_tensor = torch.tensor(frame.to_ndarray())
        frames.append(frame_tensor)

    waveform = torch.cat(frames, dim=-1)  # 合并所有帧

    waveform = waveform.mean(dim=0, keepdim=True)
    seg_len = int(SAMPLE_RATE * SEGMENT_DURATION)
    total_len = waveform.shape[1]
    segments_saved = 0

    if total_len < seg_len:
        segment = F.pad(waveform, (0, seg_len - total_len))

    else:
        num_segments = total_len // seg_len

        # 正常5秒分段
        for s in range(num_segments):
            start = s * seg_len
            end = start + seg_len
            segment = waveform[:, start:end]

        # 处理最后不足5秒的尾巴
        remainder = total_len % seg_len
        if remainder > 0:
            segment = F.pad(waveform[:, -remainder:], (0, seg_len - remainder))

    return segment


if __name__ == "__main__":
    vediopath = '../Video_Song_Actor_01/Actor_01/01-02-01-01-01-01-01.mp4'
    detectedEmotions = detect_emotions_from_video(vediopath)
    print(detectedEmotions)

    waveform = load_audio_with_pyav(vediopath)# audio met SAMPLE_RATE van 16000 en schape van [1,SAMPLE_RATE*SEGMENT_DURATION]

    result = predict(waveform)
    print(result)
