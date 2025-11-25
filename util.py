import ollama

def getResponse(prompt, messages=None, instruct="Give short responses, fit for imitating human converse. Do not give excessive explanations. You will follow the conversation of the person in front of you and not steer the conversation as long as the other person has something interesting to say"):
    if instruct is not None:
        messages.append({"role": "system", "content":instruct})
    messages.append({"role": "user", "content": prompt})
    rep = ollama.chat(model='llama3.2:3b', messages=messages)
    print("You said:", prompt)
    print(rep["message"]["content"])
    return rep["message"]["content"]