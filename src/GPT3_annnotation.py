import os
import sys

import tqdm
from openai import OpenAI
import json
API_KEY = ""

CLIENT = OpenAI(api_key=API_KEY)

def GPT_Completion(texts, client):
    # Call the API key under your account (in a secure way)
    try:
        response = client.completions.create(model="gpt-3.5-turbo-instruct", # text-davinci-002	is deprecated
                                             prompt=texts,
                                             temperature=0.5,
                                             top_p=1,
                                             max_tokens=77,
                                             frequency_penalty=0,
                                             presence_penalty=0)
        return response.choices[0].text.replace("\n", "").strip()
    except:
        return 'Error!'
# Create a GPT3-annotation_BABEL4MotionScript

GPT3_captions = dict()

Create_GPTs = False
if Create_GPTs:
    with open('action_to_sent_template.json', 'r') as fp:
                BABELis = json.load(fp)

    for label in tqdm.tqdm(BABELis):
        if label not in GPT3_captions:
            # create k=4 captions using GPT3
            GPT3_captions[label] = []
            for i in range(4):
                prompt = f"Describe a person's body movements who is performing the action {label} in details"

                GPT_generated = GPT_Completion(texts=prompt, client=CLIENT)
                GPT3_captions[label].append(GPT_generated)

    # Writing JSON data
    with open("gpt3_annotations_BABEL4MotionScript.json", 'w') as f:
        json.dump(GPT3_captions, f, indent=4)

Test_GPT = True
if Test_GPT:
    total, counter = 0, 0
    with open('action_to_sent_template.json', 'r') as fp:
        BABELis = json.load(fp)

    for label in tqdm.tqdm(BABELis):
        total+=1
        if label not in GPT3_captions:
            counter+=1
    print(total, counter)



