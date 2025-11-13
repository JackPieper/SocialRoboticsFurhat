import ollama
import furhat_realtime_api

model = 'llama3.2:3b'

messages = []

def getResponse(prompt):
    messages.append({"role": "system", "content": 'Give short responses, fit for imitating human converse. Do not give excessive explanations.'})
    ollama.chat(model=model, messages=messages)
    messages.append({"role": "user", "content": prompt})
    rep = ollama.chat(model=model, messages=messages)
    return rep["message"]["content"]

while 1:
    prmt = input()
    if prmt == 'end':
        break
    else:
        print(getResponse(prmt))

if __name__ == "__main__":

