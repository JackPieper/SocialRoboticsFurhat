import ollama
import torch
from charset_normalizer import detect
from furhat_remote_api import FurhatRemoteAPI
from furhat_realtime_api import AsyncFurhatClient
import furhat_realtime_api
import cv2
from datetime import datetime, timedelta
from Models.get_features import extract_ser_features
import os
from keras.models import load_model
import numpy as np
from librosa.util import frame
from Models import predict
import asyncio
import threading
import json
from Models.predict import load_audio_with_pyav, predict, load_ser_model
import sounddevice as sd

model = 'llama3.2:3b'
ip = '127.0.0.1'

messages = []
flag = threading.Event()

with open("Models/class2idx.json", "r") as f:
    class2idx = json.load(f)
idx2class = {v: k for k, v in class2idx.items()}
model_path = "Models/emotion_lstm_best.pth"
N_MFCC_dim = 122
num_classes = len(class2idx)
sModel = load_ser_model(model_path, N_MFCC_dim, num_classes)

def getResponse(prompt):
    messages.append({"role": "system", "content": 'Give short responses, fit for imitating human converse. Do not give excessive explanations.'})
    ollama.chat(model=model, messages=messages)
    messages.append({"role": "user", "content": prompt})
    rep = ollama.chat(model=model, messages=messages)
    print(rep["message"]["content"])
    return rep["message"]["content"]

def getCurEmo(face, voice):
    if face["probabilities"][face["final_emotion"]] +\
       voice["probabilities"][face["final_emotion"]] <\
       face["probabilities"][voice["final_emotion"]] +\
       voice["probabilities"][voice["final_emotion"]]:
        return face["final_emotion"]
    else:
        return voice["final_emotion"]

def showEmotion(foer, emotion):
    print(emotion)
    match emotion:
        case "Angry":
            foer.set_face(expr="angry")
        case "Neutral":
            foer.set_face(expr="neutral")
        case "Happy":
            foer.set_face(expr="happy")
        case "Sad":
            foer.set_face(expr="sad")
        case "Surprise":
            foer.set_face(expr="surprise")
        case "Fear":
            foer.set_face(expr="fear")
        case "Disgust":
            foer.set_face(expr="disgust")

def startCam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return 0
    return cam

def stopCam(cam):
    cam.release()
    cv2.destroyAllWindows()

def getVid(cam):
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) #Starts the window early so it shows frames from the start
    frames = []
    end_time = datetime.now() + timedelta(seconds=1)
    while datetime.now() < end_time:
        ret, frame = cam.read()
        if not ret:  # Check if frame retrieval was successful
            continue  # Skip the rest of the loop if frame retrieval fails
        frames.append(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    return frames

def sEmote():
    sec = 1
    rate = 16000
    last5 = []
    print("a")

    # while not flag.is_set():
    audio = sd.rec(int(sec * rate),
                       samplerate=rate,
                       channels=1,
                       dtype='float32')
    sd.wait()
    floatArray = audio.flatten().tolist()
    last5.append(floatArray)
    if len(last5) > 5:
        last5 = last5[-5:]
    use = [frm for seg in last5 for frm in seg]
    emotion = predict(torch.from_numpy(np.array(use, dtype=np.float32)))
    showEmotion(foer, emotion)

def emote(cam, foer):
    last5 = []
    while not flag.is_set():
        if not cam.isOpened():
            return
        last5.append(getVid(cam))
        if len(last5) > 5:
            last5 = last5[-5:]
        elif len(last5) < 5:
            for i in range(5):
                last5.append(last5[-1])
            last5 = last5[-5:]
        use = [frm for seg in last5 for frm in seg] # use is all frames of the last 5 seconds
        emotion = predict(use) # Now throw use into the model to get the resulting emotion
        foer.set_face(expr=emotion)

if __name__ == "__main__":
    messages.append({"role": "system",
                     "content": 'Give short responses but not too short, fit for imitating human converse. \
                     Do not give excessive explanations.'})
    ollama.chat(model=model, messages=messages)
    foer = FurhatRemoteAPI('127.0.0.1')
    foer.set_voice(name='Matthew')
    #set furhat's expression to sad

    foer.say(text='Hello, my name is furhat. How are you?', blocking=True)
    messages.append({"role":"assistant", "content":"Hello, my name is furhat. How are you?"})
    cam = startCam()
    # t = threading.Thread(target=emote, args=(cam, foer), daemon=True)
    # t.start()
    # print('foer')
    # t2 = threading.Thread(target=sEmote(), daemon=True)
    # print('foer1')
    # t2.start()
    # print('foer2')

    while 1:
        print('foer3')
        pmt = foer.listen()
        print(pmt)
        if pmt.message.strip().lower() == 'end':
            # t.join()
            # t2.join()
            break
        else:
            foer.say(text=getResponse(pmt.message.strip()), blocking=True)
            sEmote()
    stopCam(cam)
