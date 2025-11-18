import ollama
from furhat_remote_api import FurhatRemoteAPI
from furhat_realtime_api import AsyncFurhatClient
import furhat_realtime_api

model = 'llama3.2:3b'
ip = '127.0.0.1'

messages = []

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

if __name__ == '__main__':
    messages.append({"role": "system",
                     "content": 'Give short responses but not too short, fit for imitating human converse. Do not give excessive explanations.'})
    ollama.chat(model=model, messages=messages)
    foer = FurhatRemoteAPI('127.0.0.1')
    voices = foer.get_voices()
    foer.set_voice(name='Matthew')
    foer.say(text='Hello, my name is furhat. How are you?', blocking=True)
    messages.append({"role":"assistant", "content":"Hello, my name is furhat. How are you?"})
    while 1:
        pmt = input()
        if pmt == 'end':
            break
        else:
            foer.say(text=getResponse(pmt))
