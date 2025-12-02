import json
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

MODEL_DIR = "Models/final"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5

# ===== Controleer GPU beschikbaarheid =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gebruikt apparaat:", device)

# ===== Laad processor & model =====
print("Processor en model laden...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)          # Verplaats model naar GPU
model.eval()

# ===== Laad label mapping =====
label_map_path = f"{MODEL_DIR}/label_mapping.json"
with open(label_map_path, "r", encoding="utf-8") as f:
    label_info = json.load(f)
idx2class = {int(k): v for k, v in label_info["idx2class"].items()}
print("Label mapping geladen:", idx2class)


# ===== Functie om audio te laden =====
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = transform(waveform)
    waveform = waveform.squeeze().numpy().astype(np.float32)
    return waveform


# ===== Functie voor voorspelling op GPU =====
def predict(segments):
    # Maak een tensor op GPU voor totale waarschijnlijkheden
    probs_total = torch.zeros(len(idx2class), device=device)

    for seg in segments:
        seg = seg.squeeze()  # [T]

        # processor geeft standaard CPU tensors, verplaats naar GPU
        inputs = processor(
            seg,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # logits op GPU
            probs = torch.softmax(logits, dim=-1).squeeze()
            probs_total += probs

    # Gemiddelde waarschijnlijkheden over alle segments
    probs_avg = probs_total / len(segments)

    # Predicted label
    pred_id = torch.argmax(probs_avg).item()
    final_emotion = idx2class[pred_id]

    # Resultaten terugzetten naar CPU en float voor eenvoudiger gebruik
    results = {
        idx2class[i]: float(probs_avg[i].cpu())
        for i in range(len(idx2class))
    }

    return {
        "final_emotion": final_emotion,
        "probabilities": results
    }







