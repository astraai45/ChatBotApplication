import streamlit as st
import pickle, json, random
from keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# -------------------- config / resources --------------------
USERNAME = ["sandeep", "balaji"]
PASSWORD = ["sandeep", "balaji@123"]

lemmatizer = WordNetLemmatizer()
words   = pickle.load(open("words.pkl",   "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

model = load_model("new_chatbot_model.h5")

# -------------------- NLP helpers --------------------
def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bow(sentence):
    bag = [0] * len(words)
    for w in clean_up_sentence(sentence):
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    probs = model.predict(np.array([bow(sentence)]))[0]
    results = [(classes[i], p) for i, p in enumerate(probs) if p > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase?"
    tag = intents_list[0][0]
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

# -------------------- session state --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, answer) tuples

# -------------------- login page --------------------
if not st.session_state.logged_in:
    st.title("Login to the ChatBot Application")

    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USERNAME and pwd in PASSWORD:
            st.session_state.logged_in = True
            st.success("Successfully Logged into the Application")
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
        else:
            st.error("Invalid credentials")

# -------------------- chat interface --------------------
else:
    st.title("Cyber‑Security ChatBot")
    st.image("chatbot.png", use_container_width=True)

    # --- display existing history ---
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

    # --- input & response ---
    question = st.text_input("Ask something:", key="input_box")
    if st.button("Submit") and question:    
        intent  = predict_class(question)
        answer  = get_response(intent)
        st.session_state.history.append((question, answer))
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # --- logout button ---
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.history.clear()
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
