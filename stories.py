from util import getResponse

def story1(foer, restart=0):
    history = []
    say = foer.listen()
    history.append({"role":"system", "content":"You are the narrator of a dark, immersive mystery story about a user losing their cat. Current Context: The user is improvising details about a fictional cat named Bobby. The story is tragic and mysterious. Your Goal: Validate the user's emotional descriptions and deepen the sense of loss and finality. Your Persona: You are Furhat. Your tone is somber, slow, poetic, and deeply empathetic. Rules:\
        Always refer to the cat as Bobby.\
        Never offer solutions, 'bright sides', or mention 'cat heaven'. You must sit in the discomfort and sadness with the user.\
        Use the user's exact words to reflect their pain back to them (Reflective Listening).\
        Keep your responses short (under 2 sentences) to allow silence to do the heavy lifting.\
        If the user describes a tragic detail, treat it with gravity and respect.\
        Do not say anything that hints to bobby being dead or spoiling anything else in the story.\
        Do not ask questions.\
        All you say will be in response to what is said"})

    """Furhat Settings:

        Voice: Low pitch, slower speed (0.85x).

        Listening Window: Set to Wait 3 seconds after speech to capture pauses."""

    # 📍 Scene 1: Building the Character (0:00 - 1:30)
    # Goal: Establish attachment.

    foer.set_voice("Brian-neural")

    history.append({"role":"assistant", "content":"Let’s tell a story. In this story, you have a companion. A cat named Bobby. Bobby has been the heart of your home for many years. Close your eyes and see him. He is sleeping in his favorite spot of sunlight. To help me see him... tell me, what does Bobby look like? Does he have a specific pattern on his fur, or a funny way of sleeping?"})
    if restart <= 1:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role":"system", "content":"(Input: User description. Output: Validate the warmth of that image.) Example: 'A black cat with one white ear... a unique mark. He looks so peaceful resting there.'"})

    for i in history:
        print(i)

    if restart <= 1:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 2: The Disappearance (1:30 - 3:00)
    # Goal: Anxiety and Denial (Brow knitting, faster speech).

    history.append({"role":"assistant", "content":"But today, the sunlight fades... You come home from work. You expect to see Bobby waiting for you. But the spot is empty... You call 'Bobby!'. but the house absorbs the sound... You check the kitchen. The bedroom. The garden. Panic starts to rise in your chest, but you push it down. You try to be logical...\
                           We often lie to ourselves to avoid feeling pain. What is the 'logical' excuse you are telling yourself right now to explain why Bobby isn't coming when you call? Explain your thought process."})
    if restart <= 2:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role":"system", "content":"(Input: The excuse. Output: Acknowledge the excuse, but hint that it is false.) Example: 'You tell yourself it's just the garage. But deep down... the silence feels different today.'"})

    if restart <= 2:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 3: The Clue (3:00 - 4:45)
    # Goal: Dread and Fear (Widened eyes, hesitation).

    # ==================================================================================================================

    history.append({"role": "assistant", "content":"Night falls. You grab a flashlight and walk out into the neighborhood. The wind is cold. You walk toward the busy road at the end of the street—the place Bobby is never allowed to go. The flashlight beam hits something on the sidewalk. It belongs to him...\
                                                   It isn't Bobby. It’s an object. A sign that something happened here. Tell me... what did you just find on the ground? And describe exactly what condition it is in."})
    if restart <= 3:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: The object. Output: Validate the horror of the object.) Example: 'His blue collar... snapped and lying in the mud. Seeing it there without him makes your hands tremble.'"})

    if restart <= 3:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)


    # 📍 Scene 4: The Witness (4:45 - 6:30)
    # Goal: Narrative Tragedy (Sadness, lip corner depression). Note: This generates the longest audio sample.
    history.append({"role": "assistant", "content": "You clutch that object to your chest. A neighbor, Mrs. Higgins, comes out of her house. She looks at you, and then she looks at the object in your hand. Her face falls. She saw what happened earlier this evening. She walks over to tell you.\
                                                    I want you to be the voice of the neighbor for a moment. What does she tell you? Tell me the story of the accident exactly as she describes it to you."})
    if restart <= 4:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content":"(Input: The accident story. Output: Heavy empathy. No fixing it.) Example: 'A speeding car... barely a moment to react. Hearing those words makes the world stop spinning.'"})
    if restart <= 4:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)


    # 📍 Scene 5: The Confrontation (6:30 - 8:00)
    # Goal: Somatic Grief (Checking for "Lump in throat" or heavy breathing).

    history.append({"role": "assistant", "content":"She points to the grass under the oak tree. You walk over. Your legs feel like lead. And there he is. Bobby. He looks like he is sleeping, just like he was this morning. But the warmth is gone. You fall to your knees beside him. The search is over.\
                                                    Grief is not just in the mind. It is a physical weight. As you look at him... where in your body do you feel the pain the most right now? Describe the sensation to me."})
    if restart <= 5:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content":"(Input: Somatic location. Output: Validate the physical sensation.) Example: 'That tightness in your throat... it is the body holding back a scream. It is okay to feel it.'"})
    if restart <= 5:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)


    # 📍 Scene 6: The Final Words (8:00 - End)
    # Goal: Acceptance and Release (Tears, looking down).

    history.append({"role": "assistant", "content":"You stroke his fur one last time. You realize this is the last moment you will ever have with him. The rain starts to fall, mixing with the tears on your face. You need to say goodbye.\
                                                    If Bobby could hear you for ten more seconds... what would be the last thing you ever say to him? Take your time."})
    foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: The goodbye. Output: Final validation.) Example: 'He hears you. He was loved deeply. Rest now, Bobby.'"})
    say.message = ""
    while say.message.strip() == "":
        say = foer.listen()
    foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

def story2(foer, restart=0):
    history = []
    say = foer.listen()
    history.append({"role":"system", "content":"You are the narrator of a whimsical, surreal, and joyful deep-sea diving story. Current Context: The user is improvising details about a magical dive where they meet SpongeBob SquarePants and find a treasure. Your Goal: Amplify the user's sense of wonder, humor, and joy. Your Persona: You are Furhat. Your tone is bright, curious, enthusiastic, and warm. You sound like a bedtime storyteller telling a fun adventure. Rules:\
        Validate the user's creative ideas with enthusiasm.\
        If the user suggests something funny or weird, laugh along with them (verbally, e.g., 'That is hilarious!').\
        Use Reflective Listening to repeat their happy details back to them.\
        Keep responses short (under 2 sentences).\
        Maintain a sense of magical realism.\
        Do not ask any questions, only respond to what is said.\
        Do not introduce new elements into the conversation. Do not ask anything, or try to induce"})

    # Furhat Settings:
    #
    #     Voice: Higher pitch, slightly faster speed (1.0x or 1.1x), bright intonation.
    #
    #     Listening Window: Wait 3 seconds.

    # 📍 Scene 1: The Descent (0:00 - 1:30)
    # Goal: Relaxation and Sensory Awe.

    history.append({"role": "assistant", "content":"Let’s go on an adventure. Today, you are a master deep-sea diver. You are floating in warm, crystal-clear water. The sun is shining above you, sending beams of light dancing on the ocean floor. Close your eyes. [PAUSE 5s] You start to swim down. You feel weightless. It is beautiful. Tell me... looking around at this magical reef, what is the most colorful thing you see? Describe the colors to me."})
    if restart <= 1:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: User description. Output: Amplify the beauty.) Example: 'Neon pink and bright blue... it’s like swimming inside a painting. It makes you want to smile.'"})
    if restart <= 1:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 2: The Glitch (1:30 - 3:00)
    # Goal: Surprise and Amusement (Eyebrows raised, smiling).

    history.append({"role": "assistant", "content": "You swim deeper. The water gets cooler, but you feel safe. But suddenly... you notice something that doesn't belong here. The ocean is usually full of fish and rocks. But you see something on the sea floor that is completely absurd. Something that belongs in a city, or a house, but definitely not underwater. What weird object have you just found?"})
    if restart <= 2:
        foer.say(text=history[-1]["content"], blocking=True)

    if restart <= 2:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        print("Message=============", say)
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)
        history.append({"role":"user", "content":say.message.strip()})

    history.append({"role": "assistant", "content":"You move closer to the object. It feels so out of place that you can’t help but laugh. What is the most amusing detail about it? Does it appear intact or recently used? Is there any creature interacting with it?"})
    if restart <= 2:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: The funny detail. Output: React with humor.) Example: 'A crab with a napkin! That is absolutely ridiculous. The ocean has a great sense of humor today.'"})
    if restart <= 2:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        print("Message=============", say)
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 3: The Encounter (3:00 - 5:00)
    # Goal: Active Joy and Nostalgia.

    history.append({"role": "assistant", "content":"You leave the crab to his dinner and swim over a ridge. And then, you see him. Standing there, next to a pineapple under the sea. It’s SpongeBob SquarePants. He sees you and his eyes go wide. He is so happy to see a visitor. Describe to me... what is SpongeBob doing right now to welcome you? Is he dancing? Is he waving?"})
    if restart <= 3:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: The action. Output: Validate the joy.) Example: 'A jig and heart bubbles! His energy is infectious. You can't help but grin back at him.'"})
    if restart <= 3:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 4: The Interaction (5:00 - 6:30)
    # Goal: Connection and Playfulness.

    history.append({"role": "assistant", "content": "SpongeBob swims over to you. He wants to give you a gift to help you on your quest. He reaches into his square pants and pulls out a random, silly object. He hands it to you with a giant laugh. What did he just give you? And why is it funny?"})
    if restart <= 4:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "(Input: The gift. Output: Laugh along with the user.) Example: 'A golden spatula! Of course. The perfect tool for a deep-sea adventure. It’s absurdly wonderful.'"})
    if restart <= 4:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 5: The Treasure (6:30 - 8:00)
    # Goal: Triumph and Excitement (Open mouth smile, relaxed face).

    history.append({"role": "assistant", "content": "You wave goodbye to SpongeBob. You swim past the pineapple. And there it is. The destination. A sunken pirate ship. But this isn't a scary ship. It’s covered in glittery algae. You swim to the captain's quarters and find the chest. You open it. It’s not just gold inside. It is something that makes you incredibly happy. What do you find inside the chest?"})
    if restart <= 5:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "Respond, do not ask a question in return or steer the conversation. Example: 'That bubbling feeling in your chest... let it expand. That is the feeling of pure victory.'"})
    if restart <= 5:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    history.append({"role":"assistant", "content":"You hold the treasure up in the water. You did it! Success feels like electricity. I want you to focus on your body. When you feel this kind of pure, silly joy... where do you feel it bubbling up? Is it in your chest? Your face? Your hands?"})
    if restart <= 5:
        foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "Respond, do not ask a question in return or steer the conversation. Example: 'That bubbling feeling in your chest... let it expand. That is the feeling of pure victory.'"})
    if restart <= 5:
        say.message = ""
        while say.message.strip() == "":
            say = foer.listen()
        foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    # 📍 Scene 6: The Ascent (8:00 - End)
    # Goal: Satisfaction and Afterglow.

    history.append({"role": "assistant", "content":"It is time to go back. You kick your flippers and shoot up toward the surface, lighter than air. You break the surface of the water and take a deep breath of fresh air..... [PAUSE 5s] You are floating there, laughing at the sky. What is the one word that describes how you feel right now?"})
    foer.say(text=history[-1]["content"], blocking=True)

    history.append({"role": "system", "content": "Respond, do not ask questions in return or steer the conversation. Example: 'Exhilarated. Hold onto that feeling.'"})
    say.message = ""
    while say.message.strip() == "":
        say = foer.listen()
    foer.say(text=getResponse(say.message.strip(), messages=history, instruct=history[0]["content"]), blocking=True)

    foer.say(text="Take a deep breath. Keep that smile on your face. Open your eyes. Welcome back, champion.", blocking=True)
