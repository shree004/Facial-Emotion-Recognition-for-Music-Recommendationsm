import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

# Initialize Streamlit app
st.header("Emotion Based Music Recommendation")

# Custom CSS for the Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #191414;
        color: #1DB954;
    }
    .stTextInput>div>div>input {
        color: #1DB954;
        background-color: #282828;
        border-color: #1DB954;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #1DB954;
        color: #191414;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1ED760;
        color: #191414;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for emotion detection
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load previous emotion (if available)
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

# Reset session state if emotion is not available
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Function to load Keras model and labels
def load_emotion_model():
    try:
        model = load_model("model.h5")
        label = np.load("labels.npy")
        return model, label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the trained model and labels
model, label = load_emotion_model()

# Class for processing emotion detection
class EmotionProcessor:
    def __init__(self):
        self.holis = mp.solutions.holistic.Holistic()
        self.drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = self.holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            pred = label[np.argmax(model.predict(lst))]

            np.save("emotion.npy", np.array([pred]))

        self.drawing.draw_landmarks(frm, res.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=self.drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=self.drawing.DrawingSpec(thickness=1))
        self.drawing.draw_landmarks(frm, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        self.drawing.draw_landmarks(frm, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Input fields for language and singer
lang = st.text_input("Language", key="language")
singer = st.text_input("Singer", key="singer")

# Start capturing emotion if language and singer are provided
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="emotion", desired_playing_state=True,
                video_processor_factory=EmotionProcessor)

# Button to recommend songs
btn = st.button("Recommend songs")

# When button is clicked
if btn:
    # Check if emotion is detected
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        # Open Spotify search with provided parameters
        webbrowser.open(f"https://open.spotify.com/search/{lang}%20{singer}%20{emotion}%20/playlists")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
