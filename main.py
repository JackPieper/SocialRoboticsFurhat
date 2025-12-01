import torch
from furhat_remote_api import FurhatRemoteAPI
import cv2
from datetime import datetime, timedelta
from keras.models import load_model
import numpy as np
from numpy.ma.core import argmax
from Models import predict
import threading
from Models.predict import predict
import sounddevice as sd
from stories import story1, story2
from util import getResponse
import json

model = 'llama3.2:3b' #nieuwe modellen kijken welke de meest passende antwoorden geven
ip = ('localhost')

msgs = []
flag = threading.Event()
endStory = threading.Event()

def startCam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return 0
    return cam

def stopCam(cam):
    cam.release()
    cv2.destroyAllWindows()

def vidRes(cam, model):
    if not cam.isOpened():
        return
    temp = getVid(cam, model)
    return temp

def getVid(cam, model):
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) #Starts the window early so it shows frames

    detectedEmotions = []
    faceDetect = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    emotion_sums = np.zeros(len(labels_dict))
    frame_count = 0

    end_time = datetime.now() + timedelta(seconds=1)
    while datetime.now() < end_time:
        ret, frame = cam.read()
        if not ret:  # Check if frame retrieval was successful
            continue  # Skip the rest of the loop if frame retrieval fails

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            emotion_sums += [a + b for a, b in zip(result[0], emotion_sums)]
            frame_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            detectedEmotions.append(labels_dict[label])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        avg_probabilities = emotion_sums / frame_count

    return avg_probabilities

def audRes():
    sec = 1
    rate = 16000
    audio = sd.rec(
        int(sec * rate),
        samplerate=rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    if np.max(np.abs(audio)) < 0.05:
        return [0, 0, 0, 0, 0, 0, 0]

    waveform = torch.from_numpy(audio.squeeze().copy()).unsqueeze(0)  # [1, 16000]
    segments = [waveform]
    emotion = predict(segments)
    return [emotion["probabilities"]["angry"], emotion["probabilities"]["disgust"], emotion["probabilities"]["fearful"], emotion["probabilities"]["happy"], emotion["probabilities"]["neutral"], emotion["probabilities"]["sad"], emotion["probabilities"]["surprised"]]

def showEmotion(foer, emotion):
    print("EMOTION FOUND:", emotion)
    match emotion.lower():
        case "angry":
            foer.gesture(name="ExpressAnger", async_req=True)
        case "happy":
            foer.gesture(name="BigSmile", async_req=True)
        case "sad":
            foer.gesture(name="ExpressSad", async_req=True)
        case "surprise":
            foer.gesture(name="Surprise", async_req=True)
        case "fear":
            foer.gesture(name="ExpressFear", async_req=True)
        case "disgust":
            foer.gesture(name="ExpressDisgust", async_req=True)

def getEmo(cam, foer, reflect):
    model = load_model('Models/model_file.h5')
    lastFrames = []
    lastAudio = []
    emotions = []
    it = 0
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    while not flag.is_set():
        print("iteration:", it)
        it+=1
        lastFrames.append(vidRes(cam, model))
        lastAudio.append(audRes())
        lastFrames = format(lastFrames)
        lastAudio = format(lastAudio)
        print("a:", labels_dict[np.argmax(lastAudio[0])], "v:", labels_dict[np.argmax(lastFrames[0])])
        resultaat = getAvg(lastAudio, lastFrames)
        emotions.append(labels_dict[argmax(resultaat)])
        if endStory.is_set():
            with open("tests.jsonl", "a") as f:
                f.write(json.dumps(emotions) + "\n")
            endStory.clear()

        if len(np.where(resultaat == np.max(resultaat))[0]) == 1 and reflect:
            showEmotion(foer, labels_dict[argmax(resultaat)])

def format(var):
    if len(var) > 5:
        var = var[-5:]
    elif len(var) < 5:
        for i in range(5-len(var)):
            var.append(var[-1])
    return var

def getAvg(list1, list2):
    new = []
    for i in range(len(list1[0])):
        a = 0
        for j in range(len(list1)):
            a += list1[j][i] + list2[j][i]
        new.append(a)
    return new

if __name__ == "__main__":
    foer = FurhatRemoteAPI(ip)
    foer.set_voice(name='Matthew')
    cam = startCam()

    foer.say(text='Hello, please start a story, or ask me something in your terminal')
    com = input("Start furhat:\n1: Start story 1 (sad)\n2: start story 2 (happy)\nor give a message to continue chatting.\n")

    foer.say(text='would you like to enable empathy?')
    emp = ""
    while emp.strip() != ("y" or "n"):
        emp = input("Enable empathy? (y/n)\n")

    if emp.strip() == "y":
        bEmp = True
    else:
        bEmp = False

    t = threading.Thread(target=getEmo, args=(cam, foer, bEmp), daemon=True)
    t.start()
    match com:
        case "1":
            story1(foer)
        case "2":
            story2(foer)
    endStory.set()

    while 1:
        pmt = foer.listen()
        print(pmt)
        if pmt.message.strip().lower() == 'and':
            flag.set()
            break
        else:
            foer.say(text=getResponse(pmt.message.strip(), messages=msgs), blocking=True)
    stopCam(cam)

