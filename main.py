import streamlit as st
import numpy as np
from keras.models import load_model
import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer
import json
from gtts import gTTS
import time
import os

# Load model and data
lemmatizer = WordNetLemmatizer()
model = load_model('new_chatbot_model.h5')
intents = json.loads(open('intents.json').read())
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)
classes = pickle.load(open('classes.pkl', 'rb'))

# Clean and bag-of-words logic
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure I understand. Could you rephrase?"
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CyberSecurity ChatBot", layout="centered")
st.title("🛡️ CyberSecurity ChatBot")
st.markdown("---")

# Choose interaction type
option = st.radio("Choose Interaction Type", ("Text to Text", "Text to Voice"))

user_input = st.text_input("You:", "")

if st.button("Submit") and user_input:
    ints = predict_class(user_input, model)
    response = get_response(ints, intents)

    if option == "Text to Text":
        st.success(f"CyberBot: {response}")

    elif option == "Text to Voice":
        text_to_speech(response)
        st.success(f"CyberBot: {response}")
        time.sleep(1)
        st.audio("response.mp3", format="audio/mp3")
