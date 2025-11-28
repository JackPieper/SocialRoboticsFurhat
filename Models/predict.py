import cv2
import numpy as np
from keras.models import load_model
import torch
import torch.nn.functional as F
import json
import av

import json
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification


MODEL_DIR = "Models/final"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5

# ===== Load model, processor and label mapping =====
print("Loading processor & model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label mapping
label_map_path = f"{MODEL_DIR}/label_mapping.json"
with open(label_map_path, "r", encoding="utf-8") as f:
    label_info = json.load(f)

idx2class = {int(k): v for k, v in label_info["idx2class"].items()}

print("Loaded label mapping:", idx2class)




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




# ===== Audio loading function =====
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = transform(waveform)
    waveform = waveform.squeeze().numpy().astype(np.float32)
    return waveform


def predict(segments):
    probs_total = torch.zeros(len(idx2class))

    for seg in segments:
        seg = seg.squeeze()  # [T]
        inputs = processor(
            seg,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
            probs_total += probs

    probs_avg = probs_total / len(segments)

    pred_id = torch.argmax(probs_avg).item()
    final_emotion = idx2class[pred_id]

    results = {
        idx2class[i]: float(probs_avg[i])
        for i in range(len(idx2class))
    }

    return {
        "final_emotion": final_emotion,
        "probabilities": results
    }

def load_audio_with_pyav(video_path):
    container = av.open(video_path)
    audio_stream = next(s for s in container.streams if s.type == 'audio')

    samples = []

    # 解码帧
    for frame in container.decode(audio_stream):
        array = frame.to_ndarray()          # shape: (samples,) 或 (channels, samples)
        array = np.mean(array, axis=0) if array.ndim > 1 else array  # 转 mono
        samples.append(torch.tensor(array, dtype=torch.float32))

    # 拼接成一个大 waveform
    waveform = torch.cat(samples).unsqueeze(0)  # [1, T]

    # 分成多个 5 秒 segment
    seg_len = SAMPLE_RATE * SEGMENT_DURATION
    total_len = waveform.shape[1]

    segments = []
    for start in range(0, total_len, seg_len):
        segment = waveform[:, start:start+seg_len]
        if segment.shape[1] < seg_len:
            segment = F.pad(segment, (0, seg_len - segment.shape[1]))
        segments.append(segment)

    return segments    # 返回多个 segment






