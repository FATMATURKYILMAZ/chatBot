# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:16:43 2023

@author: ASUS
"""

import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents=json.load(f)
    
FILE="data.pth"
data=torch.load(FILE)    

input_size=data['input_size']
hidden_size=data['hidden_size']
output_size=data['output_size']
all_word=data['all_words']
tags=data['tags']
model_state=data['model_state']

model=NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

bot_Name="Yusuf"
print("Let's chat! Type quit to exit" )
while True:
    sentence=input('You:')
    if sentence =="quit":
        break
    sentence=tokenize(sentence)
    X=bag_of_words(sentence, all_word)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy()
    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    
    probs=torch.softmax(output, dim=1)
    prob=probs[0][predicted.item()]
    
    if prob.item()>0.75:
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print(f"{bot_Name}:{random.choice(intent['responses'])}")
                
    else:
        print(f"{bot_Name}:I do not understand...")
            
    
    