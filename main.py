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
import random
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import zmq
import time
import os

# ---------------------- Global Variables ----------------------
model_file = 'Models/model_file.h5'
ip = '10.140.59.100'

video_queue = queue.Queue()
audio_queue = queue.Queue()
stop_event = threading.Event()
results_lock = threading.Lock()

labels_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

video_results = []
audio_results = []
fused_results = []

is_speaking = threading.Event()

# ---------------------- Camera Selection ----------------------
def list_cameras(max_tested=5):
    available_cams = []
    print("\n=== Available Cameras ===")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cams.append(i)
                print(f"{i}: Camera {i}")
            cap.release()
    return available_cams

def choose_camera():
    cams = list_cameras()
    if not cams:
        print("No cameras detected")
        return None
    while True:
        try:
            cam_id = int(input("Select camera ID: "))
            if cam_id in cams:
                return cam_id
            else:
                print("Invalid camera ID, try again")
        except:
            print("Invalid input, enter a number")

def startCam(cam_id):
    cam = cv2.VideoCapture(cam_id)
    if not cam.isOpened():
        print("Error: Cannot open camera")
        return None
    return cam

def stopCam(cam):
    cam.release()
    cv2.destroyAllWindows()

# ---------------------- Frame Processing ----------------------
def process_frame(frame, model, showVideo, faceDetect):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    emotion_probs = []

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48)) / 255.0
        reshaped = resized.reshape(1,48,48,1)
        prob = model.predict(reshaped, verbose=0)[0]
        emotion_probs.append(prob)

        if showVideo:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            label_idx = int(np.argmax(prob))
            cv2.putText(frame, labels_dict[label_idx], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if showVideo:
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()

    return np.mean(emotion_probs, axis=0)

# ---------------------- Video Thread (Local Camera) ----------------------
def video_thread(cam, model, video_queue, stop_event, showVideo):
    faceDetect = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')

    while not stop_event.is_set():
        emotion_sums = np.zeros(len(labels_dict))
        frame_count = 0
        end_time = datetime.now() + timedelta(seconds=1)

        while datetime.now() < end_time:
            ret, frame = cam.read()
            if not ret:
                continue
            prob = process_frame(frame, model, showVideo, faceDetect)
            emotion_sums += prob
            frame_count += 1

        if frame_count > 0:
            avg_probabilities = emotion_sums / frame_count
            video_queue.put(avg_probabilities)
            with results_lock:
                video_results.append(avg_probabilities)

# ---------------------- Video Thread (Furhat Camera) ----------------------
def furhat_video_thread(video_queue, stop_event, model, showVideo, furhat_ip):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{furhat_ip}:3000")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    faceDetect = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')

    while not stop_event.is_set():
        try:
            emotion_sums = np.zeros(len(labels_dict))
            frame_count = 0
            end_time = datetime.now() + timedelta(seconds=1)

            while datetime.now() < end_time:
                img_bytes = socket.recv()
                meta_json = socket.recv_string()

                if len(img_bytes) == 0:
                    continue

                try:
                    img = Image.open(BytesIO(img_bytes)).convert('RGB')
                except Exception as e:
                    print("Skipping invalid image:", e)
                    continue

                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                prob = process_frame(frame, model, showVideo, faceDetect)
                emotion_sums += prob
                frame_count += 1

            if frame_count > 0:
                avg_probabilities = emotion_sums / frame_count
                video_queue.put(avg_probabilities)
                with results_lock:
                    video_results.append(avg_probabilities)

        except Exception as e:
            print("Furhat camera error:", e)
            time.sleep(0.1)
            continue

    cv2.destroyAllWindows()

# ---------------------- Audio Thread (Local Microphone) ----------------------
def audio_thread(audio_queue, stop_event, micDeviceID, rate=16000):
    sd.default.device = (micDeviceID, None)
    sd.default.samplerate = rate
    while not stop_event.is_set():
        if is_speaking.is_set():
            audio_queue.put([0]*7)
            with results_lock:
                audio_results.append([0]*7)
            time.sleep(0.1)
            continue

        try:
            audio = sd.rec(int(rate*1), samplerate=rate, channels=1, dtype='float32')
            sd.wait()
        except Exception as e:
            print("Audio capture error:", e)
            continue

        if np.max(np.abs(audio)) < 0.05:
            audio_queue.put([0]*7)
            with results_lock:
                audio_results.append([0]*7)
            continue

        waveform = torch.from_numpy(audio.squeeze().copy()).unsqueeze(0)
        emotion = predict([waveform])
        probs = [
            emotion["probabilities"].get('angry',0),
            emotion["probabilities"].get('disgust',0),
            emotion["probabilities"].get('fearful',0),
            emotion["probabilities"].get('happy',0),
            emotion["probabilities"].get('neutral',0),
            emotion["probabilities"].get('sad',0),
            emotion["probabilities"].get('surprised',0)
        ]
        audio_queue.put(probs)
        with results_lock:
            audio_results.append(probs)

# ---------------------- Audio Thread (Furhat Microphone) ----------------------
def furhat_audio_thread(audio_queue, stop_event):
    ZMQ_URL_audio = f"tcp://{ip}:3001"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_URL_audio)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    TARGET_SAMPLES = SAMPLE_RATE * 1
    TARGET_BYTES = TARGET_SAMPLES * SAMPLE_WIDTH

    while not stop_event.is_set():
        if is_speaking.is_set():
            audio_queue.put([0]*7)
            with results_lock:
                audio_results.append([0]*7)
            time.sleep(0.1)
            continue

        audio_buffer = bytearray()
        while len(audio_buffer) < TARGET_BYTES and not stop_event.is_set():
            socks = dict(poller.poll(timeout=50))
            if socket in socks:
                data = socket.recv()
                audio_buffer.extend(data)
            else:
                continue

        if stop_event.is_set():
            break

        if len(audio_buffer) < TARGET_BYTES:
            audio_queue.put([0]*7)
            with results_lock:
                audio_results.append([0]*7)
            continue

        audio_buffer = audio_buffer[:TARGET_BYTES]
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

        if np.max(np.abs(audio_np)) < 0.05:
            audio_queue.put([0]*7)
            with results_lock:
                audio_results.append([0]*7)
            continue

        waveform = torch.from_numpy(audio_np.copy()).unsqueeze(0)
        emotion = predict([waveform])
        probs = [
            emotion["probabilities"].get('angry', 0),
            emotion["probabilities"].get('disgust', 0),
            emotion["probabilities"].get('fearful', 0),
            emotion["probabilities"].get('happy', 0),
            emotion["probabilities"].get('neutral', 0),
            emotion["probabilities"].get('sad', 0),
            emotion["probabilities"].get('surprised', 0)
        ]
        audio_queue.put(probs)
        with results_lock:
            audio_results.append(probs)

# ---------------------- Empathy Mode Thread ----------------------
def emotion_thread(foer):
    stop_event.wait(3)
    while not stop_event.is_set():
        cnt = random.random()
        if cnt >= 0.333:
            fused = [x + y for x, y in zip(video_results[-1], audio_results[-1])]
            idx = int(np.argmax(fused))
            emotion = labels_dict.get(idx, "Neutral")
            showEmotion(foer, emotion)

        stop_event.wait(10)

def showEmotion(foer, emotion):
    print("EMOTION DETECTED:", emotion)
    if emotion.lower() == "angry":
        foer.gesture(name="ExpressAnger", async_req=True)
    elif emotion.lower() == "happy":
        foer.gesture(name="BigSmile", async_req=True)
    elif emotion.lower() == "sad":
        foer.gesture(name="ExpressSad", async_req=True)
    elif emotion.lower() == "surprise":
        foer.gesture(name="Surprise", async_req=True)
    elif emotion.lower() == "fear":
        foer.gesture(name="ExpressFear", async_req=True)
    elif emotion.lower() == "disgust":
        foer.gesture(name="ExpressDisgust", async_req=True)

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


# ---------------------- Plotting and Saving ----------------------
def plot_results(fused_results, save_folder, session_folder_name, filename="emotion_plot.png"):
    # Ensure the folder exists
    full_path = os.path.join(save_folder, session_folder_name)
    os.makedirs(full_path, exist_ok=True)

    emotions = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    data = np.array(fused_results)
    x = np.arange(len(data))

    plt.figure(figsize=(12, 6))
    for i in range(7):
        plt.plot(x, data[:, i], label=emotions[i], linewidth=2)

    plt.xlabel("Frame index")
    plt.ylabel("Probability")
    plt.title("Emotion Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save path
    save_path = os.path.join(full_path, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")

def save_all_results(parent_folder, session_folder_name="session_results"):
    # Create the full path
    full_path = os.path.join(parent_folder, session_folder_name)
    os.makedirs(full_path, exist_ok=True)

    with results_lock:
        with open(os.path.join(full_path, "video_results.json"), "w") as f:
            json.dump([v.tolist() for v in video_results], f, indent=4)
        with open(os.path.join(full_path, "audio_results.json"), "w") as f:
            json.dump([a.tolist() if isinstance(a, np.ndarray) else a for a in audio_results], f, indent=4)
        with open(os.path.join(full_path, "fused_results.json"), "w") as f:
            json.dump([f_.tolist() if isinstance(f_, np.ndarray) else f_ for f_ in fused_results], f, indent=4)

    print(f"All results saved in folder: {full_path}")

# ---------------------- Audio Device Selection ----------------------
def choose_input_device():
    devices = sd.query_devices()
    valid_devices = [dev for dev in devices if dev['max_input_channels'] > 0]

    print("\n=== Available Audio Input Devices ===")
    for idx, dev in enumerate(valid_devices):
        print(f"{idx}: {dev['name']} ({dev['max_input_channels']} in)")

    while True:
        try:
            micID = int(input("Select audio device ID: "))
            if 0 <= micID < len(valid_devices):
                original_index = devices.index(valid_devices[micID])
                return original_index
            else:
                print("Invalid input, try again")
        except:
            print("Invalid input, try again")

# ---------------------- Main Program ----------------------
if __name__ == "__main__":

    # --- Camera Selection ---
    cam_mode = ""
    while cam_mode.lower().strip() not in ["1", "2"]:
        cam_mode = input("\n1: Local Camera\n2: Furhat Camera\n")
    use_furhat_cam = (cam_mode.lower().strip() == "2")
    if not use_furhat_cam:
        cam_id = choose_camera()

    # --- Audio Input Selection ---
    mic_mode = ""
    while mic_mode not in ["1", "2"]:
        mic_mode = input("\nSelect audio input:\n1: Local Microphone\n2: Furhat Microphone\n")

    use_furhat_audio = (mic_mode == "2")

    if not use_furhat_audio:
        micDeviceID = choose_input_device()

    # --- Display Video ---
    show = ""
    while show.lower().strip() not in ["y", "n"]:
        show = input("\nDisplay video and predictions? (y/n)\n")
    showVideo = (show.lower().strip() == "y")

    # --- Enable Empathy Mode ---
    emp = ""
    while emp.lower().strip() not in ["y", "n"]:
        emp = input("Enable empathy mode? (y/n)\n")
    bEmp = (emp.lower().strip() == "y")

    # --- Connect to Furhat ---
    foer = FurhatRemoteAPI(ip)
    foer.set_voice(name='Matthew')
    foer.say(text="Hello, please start a story, or ask me something in your terminal")

    # --- Story Selection ---
    com = ""
    while com.lower().strip() not in ["1", "2"]:
        com = input("Select a story:\n1: story1 (sad)\n2: story2 (happy)\n")

    # --- Load Video Model ---
    video_model = load_model(model_file)

    # =========================
    # Start Video Thread
    # =========================
    if use_furhat_cam:
        v_thread = threading.Thread(
            target=furhat_video_thread,
            args=(video_queue, stop_event, video_model, showVideo, ip),
            daemon=True
        )
    else:
        cam = startCam(cam_id)
        v_thread = threading.Thread(
            target=video_thread,
            args=(cam, video_model, video_queue, stop_event, showVideo),
            daemon=True
        )

    # =========================
    # Start Audio Thread
    # =========================
    if use_furhat_audio:
        a_thread = threading.Thread(
            target=furhat_audio_thread,
            args=(audio_queue, stop_event),
            daemon=True
        )
    else:
        a_thread = threading.Thread(
            target=audio_thread,
            args=(audio_queue, stop_event, micDeviceID),
            daemon=True
        )

    # Start Threads
    v_thread.start()
    a_thread.start()

    # --- Empathy Mode Thread ---
    if bEmp:
        e_thread = threading.Thread(target=emotion_thread, args=(foer,), daemon=True)
        e_thread.start()

    # --- Play Story ---
    if com == "1":
        story1(foer, is_speaking)
    elif com == "2":
        story2(foer, is_speaking)

    # =========================
    # Stop all Threads
    # =========================
    stop_event.set()

    v_thread.join()
    a_thread.join()
    if bEmp:
        e_thread.join()

    if not use_furhat_cam:
        stopCam(cam)

    # =========================
    # Emotion Fusion
    # =========================
    while not video_queue.empty() and not audio_queue.empty():
        fuse_emotions(video_queue, audio_queue)

    # =========================
    # Save and Plot Results
    # =========================

    parent_folder = "Output"
    session_folder_name = input("Enter a name for this session folder: ")
    plot_results(fused_results,parent_folder, session_folder_name)
    save_all_results(parent_folder, session_folder_name)

    print("All results saved. Program ended.")


