import torch
from furhat_remote_api import FurhatRemoteAPI
import cv2
from keras.models import load_model
import numpy as np
from Models.predict import predict
import threading
import queue
import sounddevice as sd
from stories import story1, story2
import matplotlib.pyplot as plt
import json

# ----------------------
# Globale variabelen
# ----------------------
model_file = 'Models/model_file.h5'
ip = 'localhost'

video_queue = queue.Queue()
audio_queue = queue.Queue()
stop_event = threading.Event()
endStory = threading.Event()

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Opslag van alle voorspellingen
video_results = []  # lijst van lijsten voor video-emoties
audio_results = []  # lijst van lijsten voor audio-emoties
fused_results = []  # lijst van lijsten voor samengevoegde emoties

# ----------------------
# Camera bediening
# ----------------------
def startCam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Fout: Kan webcam niet openen.")
        return None
    return cam

def stopCam(cam):
    cam.release()
    cv2.destroyAllWindows()

# ----------------------
# Videodraad
# ----------------------
def video_thread(cam, model, video_queue, stop_event):
    faceDetect = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        emotion_sums = [0]*7
        frame_count = 0
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48)) / 255.0
            reshaped = resized.reshape(1,48,48,1)
            result = model.predict(reshaped, verbose=0)
            emotion_sums = [a+b for a,b in zip(emotion_sums, result[0])]
            frame_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            label = np.argmax(result)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if frame_count > 0:
            avg_prob = [x/frame_count for x in emotion_sums]
            video_queue.put(avg_prob)
            video_results.append(avg_prob)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()
            break

# ----------------------
# Audiodraad
# ----------------------
def audio_thread(audio_queue, stop_event):
    rate = 16000
    while not stop_event.is_set():
        audio = sd.rec(int(rate*1), samplerate=rate, channels=1, dtype='float32')
        sd.wait()
        if np.max(np.abs(audio)) < 0.05:
            audio_queue.put([0]*7)
            audio_results.append([0]*7)
            continue
        waveform = torch.from_numpy(audio.squeeze().copy()).unsqueeze(0)
        segments = [waveform]
        emotion = predict(segments)
        probs = [
            emotion["probabilities"]["angry"],
            emotion["probabilities"]["disgust"],
            emotion["probabilities"]["fearful"],
            emotion["probabilities"]["happy"],
            emotion["probabilities"]["neutral"],
            emotion["probabilities"]["sad"],
            emotion["probabilities"]["surprised"]
        ]
        audio_queue.put(probs)
        audio_results.append(probs)

# ----------------------
# Fusie en normalisatie van emoties
# ----------------------
def fuse_emotions(video_queue, audio_queue):
    if not video_queue.empty() and not audio_queue.empty():
        v_probs = video_queue.get()
        a_probs = audio_queue.get()
        fused = [x + y for x, y in zip(v_probs, a_probs)]
        total = sum(fused)
        normalized = [x/total for x in fused] if total > 0 else [0]*7
        fused_results.append(normalized)
        return normalized
    return None

# ----------------------
# Plot en opslaan
# ----------------------
def plot_results(fused_results, filename="emotion_plot.png"):
    emotions = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    data = np.array(fused_results)
    x = np.arange(len(data))

    plt.figure(figsize=(12,6))
    for i in range(7):
        plt.plot(x, data[:,i], label=emotions[i], linewidth=2)
    plt.xlabel("Tijd / Frame index")
    plt.ylabel("Waarschijnlijkheid")
    plt.title("Veranderingen in samengevoegde emotie waarschijnlijkheid")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------------
# Opslaan van alle resultaten als JSON
# ----------------------
def save_all_results():
    with open("video_results.json", "w") as f:
        json.dump(video_results, f)
    with open("audio_results.json", "w") as f:
        json.dump(audio_results, f)
    with open("fused_results.json", "w") as f:
        json.dump(fused_results, f)

# ----------------------
# Hoofdprogramma
# ----------------------
if __name__ == "__main__":
    foer = FurhatRemoteAPI(ip)
    foer.set_voice(name='Matthew')

    cam = startCam()
    if cam is None:
        exit()

    video_model = load_model(model_file)

    foer.say(text='Hello, please start a story, or ask me something in your terminal')
    com = input("Start furhat:\n1: Start story 1 (sad)\n2: start story 2 (happy)\nor give a message to continue chatting.\n")

    emp = ""
    while emp.strip() not in ["y", "n"]:
        emp = input("Enable empathy? (y/n)\n")
    bEmp = emp.strip() == "y"

    # Start video en audio threads
    v_thread = threading.Thread(target=video_thread, args=(cam, video_model, video_queue, stop_event), daemon=True)
    a_thread = threading.Thread(target=audio_thread, args=(audio_queue, stop_event), daemon=True)
    v_thread.start()
    a_thread.start()

    # Start het gekozen verhaal
    match com:
        case "1": story1(foer)
        case "2": story2(foer)

    # Verhaal beëindigen, threads stoppen
    endStory.set()
    stop_event.set()
    v_thread.join()
    a_thread.join()
    stopCam(cam)

    # Verwerk resterende queue data
    while not video_queue.empty() and not audio_queue.empty():
        fuse_emotions(video_queue, audio_queue)

    # Plot en opslaan
    plot_results(fused_results, filename="emotion_plot.png")
    save_all_results()

    print("Alle resultaten opgeslagen. Programma afgesloten.")


