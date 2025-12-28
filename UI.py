import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque, Counter
import threading
import time
import whisper
import sounddevice as sd
import wavio
import numpy as np
import random
import json
from datetime import datetime
import requests
import sys
import queue

# ---------------------------
# Device Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Enhanced TTS with Queue System
# ---------------------------
import pyttsx3

class TTSEngine:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.init_engine()
        
    def init_engine(self):
        """Initialize TTS engine with better settings"""
        try:
            self.engine = pyttsx3.init()
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            print("Available TTS voices:")
            for i, voice in enumerate(voices):
                print(f"  {i}: {voice.name}")
            
            # Configure properties
            self.engine.setProperty("rate", 160)  # Natural speaking speed
            self.engine.setProperty("volume", 1.0)
            
            # Try to find a good voice
            preferred_voices = ['zira', 'eva', 'samantha', 'female', 'david', 'hazel']
            for voice in voices:
                voice_name = voice.name.lower()
                if any(pref in voice_name for pref in preferred_voices):
                    self.engine.setProperty('voice', voice.id)
                    print(f"✅ Selected TTS voice: {voice.name}")
                    break
            
            # Start the speech processing thread
            self._start_speech_thread()
            print("✅ TTS engine initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            self.engine = None
    
    def _start_speech_thread(self):
        """Start a thread to process speech queue"""
        def speech_worker():
            while True:
                try:
                    text = self.speech_queue.get(timeout=1)
                    if text and self.engine:
                        self.is_speaking = True
                        
                        # Split into sentences for more natural speech
                        sentences = self._split_into_sentences(text)
                        
                        for sentence in sentences:
                            if sentence.strip():
                                self.engine.say(sentence.strip())
                        
                        self.engine.runAndWait()
                        self.is_speaking = False
                        
                        # Small pause between speech items
                        time.sleep(0.5)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Speech error: {e}")
                    self.is_speaking = False
        
        threading.Thread(target=speech_worker, daemon=True).start()
    
    def _split_into_sentences(self, text):
        """Split text into sentences for more natural speech"""
        # Simple sentence splitting
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?':
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences if sentences else [text]
    
    def speak(self, text):
        """Add text to speech queue"""
        if not text or not self.engine:
            print(f"(TTS would say: {text})")
            return
        
        # Clean the text
        text = str(text).strip()
        if not text:
            return
        
        # Add to queue
        self.speech_queue.put(text)
        print(f"🗣️ Added to TTS queue: {text[:50]}...")

# Initialize TTS engine
tts = TTSEngine()

# Simple speak function for compatibility
def speak(text):
    """Public speak function that uses the TTS engine"""
    tts.speak(text)

# ---------------------------
# Hugging Face LLM
# ---------------------------
class HuggingFaceLLM:
    def __init__(self, api_token=None):
        self.api_token = api_token
        self.conversation_history = []
        self.max_history = 6  # Keep last 3 exchanges
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        
        # Available models (free tier compatible)
        self.models = [
            {
                "name": "microsoft/DialoGPT-small",
                "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small",
                "context_window": 512
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                "context_window": 512
            },
            {
                "name": "google/flan-t5-small",
                "url": "https://api-inference.huggingface.co/models/google/flan-t5-small",
                "context_window": 512
            },
        ]
        
        self.current_model = self.models[0]
        print(f"🤖 Using Hugging Face model: {self.current_model['name']}")
    
    def generate_response(self, user_input: str, emotion: str, face_emotion: str = "Neutral") -> str:
        """Generate response and ensure it's spoken"""
        api_response = self._call_huggingface_api(user_input, emotion, face_emotion)
        if api_response:
            return api_response
        return self._fallback_response(user_input, emotion)
    
    def _call_huggingface_api(self, user_input: str, emotion: str, face_emotion: str):
        """Call Hugging Face Inference API"""
        prompt = self._build_prompt(user_input, emotion, face_emotion)
        
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.8 if emotion in ["Happy", "Surprise"] else 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.current_model["url"],
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list):
                    if "generated_text" in result[0]:
                        response_text = result[0]["generated_text"].strip()
                    else:
                        response_text = result[0].get("text", "").strip()
                else:
                    response_text = str(result).strip()
                
                response_text = self._clean_response(response_text)
                self._add_to_history(user_input, response_text)
                
                return response_text
                
            else:
                print(f"❌ API Error {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None
    
    def _build_prompt(self, user_input: str, emotion: str, face_emotion: str) -> str:
        """Build prompt for the model"""
        context = f"""You are EmoBot, an empathetic AI companion. The user is feeling {emotion} 
        (face shows {face_emotion}). Respond warmly in 1-2 short sentences.

Previous conversation:"""
        
        for i, exchange in enumerate(self.conversation_history[-self.max_history:]):
            if i % 2 == 0:
                context += f"\nUser: {exchange['content']}"
            else:
                context += f"\nEmoBot: {exchange['content']}"
        
        context += f"\n\nUser (feeling {emotion.lower()}): {user_input}"
        context += f"\nEmoBot:"
        
        return context
    
    def _clean_response(self, text: str) -> str:
        """Clean up the model response"""
        text = text.strip()
        
        if "EmoBot:" in text:
            text = text.split("EmoBot:")[-1].strip()
        
        text = ' '.join(text.split())
        
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text[:150]
    
    def _fallback_response(self, user_input: str, emotion: str) -> str:
        """Enhanced rule-based responses"""
        user_input_lower = user_input.lower()
        
        emotion_responses = {
            "Happy": [
                "That's wonderful to hear! Your positivity is absolutely contagious! 😊",
                "I'm genuinely happy that you're feeling good! What's making your day special?",
                "Your happiness brightens this conversation! I'd love to hear more!",
            ],
            "Sad": [
                "I'm truly sorry to hear you're feeling this way. I'm here for you. 💙",
                "That sounds really difficult. Remember, it's okay to not be okay sometimes.",
                "I can sense the weight in your words. Would sharing more help?",
            ],
            "Angry": [
                "That sounds incredibly frustrating. Would talking through it help?",
                "I understand why you'd feel that way. Sometimes things can be aggravating.",
                "Let's take a deep breath together. What's making you feel this way?",
            ],
            "Surprise": [
                "Wow, that's genuinely surprising! I'd love to hear more about what happened!",
                "I didn't see that coming either! What was your reaction?",
                "That's quite unexpected! How are you processing this?",
            ],
            "Fear": [
                "That sounds really scary. You're not alone in this - I'm here with you.",
                "It's understandable to feel afraid. What might help you feel safer?",
                "Fear can feel overwhelming. Let's take this one step at a time.",
            ],
            "Neutral": [
                "I understand. Could you tell me more about that?",
                "That's interesting. What are your thoughts about this?",
                "Thanks for sharing. How does this make you feel?",
            ]
        }
        
        topic_responses = {
            "ai": "AI is fascinating! As an AI assistant, I'm curious what interests you most?",
            "college": "College life has its rhythm. How are you finding the balance?",
            "study": "Studying requires focus. What subjects are you working on?",
            "exam": "Exams can be stressful. How are you preparing?",
            "game": "Games are great escapes! What are you playing lately?",
            "music": "Music speaks to the soul! What are you listening to?",
            "movie": "Movies create entire worlds! Seen anything good recently?",
        }
        
        for topic, response in topic_responses.items():
            if topic in user_input_lower:
                if emotion != "Neutral" and random.random() > 0.4:
                    emotion_resp = random.choice(emotion_responses[emotion])
                    return f"{response} {emotion_resp}"
                return response
        
        responses = emotion_responses.get(emotion, emotion_responses["Neutral"])
        selected = random.choice(responses)
        self._add_to_history(user_input, selected)
        return selected
    
    def _add_to_history(self, user_input: str, response: str):
        """Add to conversation history"""
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]

# Initialize LLM
llm = HuggingFaceLLM()

# ---------------------------
# Emotion Recognition Model
# ---------------------------
try:
    checkpoint = torch.load("emotion_resnet.pth", map_location=device)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    print("✅ Emotion model loaded!")
except Exception as e:
    print(f"⚠️ Emotion model error: {e}")
    classes = ["Neutral", "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# Emotion Detection
# ---------------------------
TEXT_EMOTION_DURATION = 10
EMOTION_KEYWORDS = {
    "Happy": ["happy", "good", "great", "nice", "fun", "awesome", "enjoyed", "love", "excited"],
    "Sad": ["sad", "tired", "boring", "upset", "lonely", "bad", "down", "depressed"],
    "Angry": ["angry", "annoying", "hate", "irritating", "frustrated", "mad", "upset"],
    "Surprise": ["surprise", "shock", "wow", "unexpected", "astonish"],
    "Fear": ["fear", "scared", "afraid", "worried", "anxious", "nervous"],
    "Disgust": ["disgust", "gross", "yuck", "dislike", "hate"]
}

emotion_state = {"face": "Neutral", "text": "Neutral", "text_time": 0.0, "final": "Neutral"}
emotion_buffer = deque(maxlen=20)

def detect_text_emotion(text: str) -> str:
    text = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    return "Neutral"

def smooth_emotion() -> str:
    if len(emotion_buffer) == 0:
        return "Neutral"
    
    recent = list(emotion_buffer)[-5:] if len(emotion_buffer) >= 5 else list(emotion_buffer)
    return Counter(recent).most_common(1)[0][0]

def get_final_emotion() -> str:
    now = time.time()
    
    if emotion_state["text"] != "Neutral":
        if now - emotion_state["text_time"] <= TEXT_EMOTION_DURATION:
            emotion_state["final"] = emotion_state["text"]
            return emotion_state["final"]
        else:
            emotion_state["text"] = "Neutral"
    
    emotion_state["final"] = emotion_state["face"]
    return emotion_state["final"]

# ---------------------------
# Chat Management with TTS Integration
# ---------------------------
def append_chat_and_speak(sender: str, message: str, speak_it: bool = True):
    """Append message to chat and optionally speak it"""
    chat_log.config(state="normal")
    
    timestamp = datetime.now().strftime("%H:%M")
    
    if sender == "You":
        chat_log.insert("end", f"[{timestamp}] {sender}: \"{message}\"\n\n")
    else:
        chat_log.insert("end", f"{sender}: \"{message}\"\n\n")
        # Always speak bot responses
        if speak_it:
            speak(message)
    
    lines = chat_log.get("1.0", "end").splitlines()
    if len(lines) > 20:
        chat_log.delete("1.0", f"{len(lines)-19}.0")
    
    chat_log.config(state="disabled")
    chat_log.see("end")

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Whisper Setup
# ---------------------------
whisper_model = None
listening = False
recording_duration = 8

def init_whisper():
    global whisper_model
    try:
        print("🔊 Loading Whisper...")
        whisper_model = whisper.load_model("base")
        print("✅ Whisper ready!")
    except Exception as e:
        print(f"⚠️ Whisper error: {e}")
        whisper_model = None

threading.Thread(target=init_whisper, daemon=True).start()

def get_microphone_index() -> int:
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if 'microphone' in device['name'].lower():
                    return i
        return 0
    except:
        return 0

MIC_INDEX = get_microphone_index()

def listen_and_respond():
    global listening
    if listening:
        return
    
    listening = True
    
    if whisper_model is None:
        append_chat_and_speak("Bot", "Speech recognition loading... Please type.", speak_it=True)
        listening = False
        return
    
    fs = 16000

    append_chat_and_speak("Bot", f"Listening for {recording_duration} seconds...", speak_it=False)

    try:
        audio_data = sd.rec(int(recording_duration * fs), 
                           samplerate=fs, 
                           channels=1, 
                           device=MIC_INDEX,
                           dtype='float32')
        sd.wait()
        wavio.write("temp_audio.wav", (audio_data * 32767).astype(np.int16), fs, sampwidth=2)

        result = whisper_model.transcribe("temp_audio.wav", fp16=torch.cuda.is_available())
        user_text = result["text"].strip()
        print(f"✅ Recognized: {user_text}")

        if not user_text or len(user_text) < 2:
            append_chat_and_speak("Bot", "I didn't catch that. Please try again.", speak_it=True)
            listening = False
            return

    except Exception as e:
        append_chat_and_speak("Bot", "Microphone error.", speak_it=True)
        print(f"❌ Recording error: {e}")
        listening = False
        return

    append_chat_and_speak("You", user_text, speak_it=False)
    
    text_emotion = detect_text_emotion(user_text)
    if text_emotion != "Neutral":
        emotion_state["text"] = text_emotion
        emotion_state["text_time"] = time.time()
    
    final_emotion = get_final_emotion()
    face_emotion = emotion_state["face"]
    
    append_chat_and_speak("Bot", "Thinking...", speak_it=False)
    
    def generate_and_speak():
        reply = llm.generate_response(user_text, final_emotion, face_emotion)
        
        chat_log.config(state="normal")
        chat_log.delete("end-2l", "end-1l")
        
        timestamp = datetime.now().strftime("%H:%M")
        chat_log.insert("end", f"{timestamp} 🤖 Bot: \"{reply}\"\n\n")
        chat_log.config(state="disabled")
        chat_log.see("end")
        
        # SPEAK THE RESPONSE - This is the key fix
        speak(reply)
        
        topic_var.set(f"💭 {llm.current_model['name']} | {len(llm.conversation_history)//2} msgs")
        
        global listening
        listening = False
    
    threading.Thread(target=generate_and_speak, daemon=True).start()

# ---------------------------
# Button Functions with TTS
# ---------------------------
def start_listening():
    if listening:
        return
    threading.Thread(target=listen_and_respond, daemon=True).start()

def send_message():
    user_text = user_entry.get().strip()
    if not user_text or user_text == "Type your message here...":
        return

    user_entry.delete(0, tk.END)
    
    text_emotion = detect_text_emotion(user_text)
    if text_emotion != "Neutral":
        emotion_state["text"] = text_emotion
        emotion_state["text_time"] = time.time()
    
    final_emotion = get_final_emotion()
    face_emotion = emotion_state["face"]
    
    append_chat_and_speak("You", user_text, speak_it=False)
    append_chat_and_speak("Bot", "Thinking...", speak_it=False)
    
    def generate_and_speak():
        reply = llm.generate_response(user_text, final_emotion, face_emotion)
        
        chat_log.config(state="normal")
        chat_log.delete("end-2l", "end-1l")
        
        timestamp = datetime.now().strftime("%H:%M")
        chat_log.insert("end", f"{timestamp} 🤖 Bot: \"{reply}\"\n\n")
        chat_log.config(state="disabled")
        chat_log.see("end")
        
        # SPEAK THE RESPONSE
        speak(reply)
        
        topic_var.set(f"💭 {llm.current_model['name']} | {len(llm.conversation_history)//2} msgs")
    
    threading.Thread(target=generate_and_speak, daemon=True).start()

def speak_last_message():
    chat_text = chat_log.get("1.0", "end-1c")
    lines = chat_text.split("\n")
    last_bot_message = ""
    
    for line in reversed(lines):
        if "Bot:" in line and "Thinking" not in line:
            if '"' in line:
                message_start = line.find('"') + 1
                message_end = line.rfind('"')
                if message_start < message_end:
                    message = line[message_start:message_end]
                    if message.strip():
                        last_bot_message = message
                        break
    
    if last_bot_message:
        speak(last_bot_message)
    else:
        speak("No recent response to speak")

# ---------------------------
# Tkinter GUI Setup
# ---------------------------
root = tk.Tk()
root.title("🤖 Emotion-Aware Chatbot")
root.geometry("1000x800")
root.configure(bg="#0f172a")

# Make window responsive
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, bg="#0f172a")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Top bar
top_bar = tk.Frame(main_frame, bg="#1e293b", height=70)
top_bar.pack(fill="x", pady=(0, 20))

emotion_var = tk.StringVar(value="🎭 Detecting emotion...")
topic_var = tk.StringVar(value=f"💭 {llm.current_model['name']}")

emotion_label = tk.Label(top_bar, textvariable=emotion_var, fg="#38bdf8", bg="#1e293b", 
                         font=("Segoe UI", 15, "bold"))
emotion_label.pack(side="left", padx=30)

topic_label = tk.Label(top_bar, textvariable=topic_var, fg="#a5b4fc", bg="#1e293b", 
                      font=("Segoe UI", 13))
topic_label.pack(side="right", padx=30)

# Content area
content_frame = tk.Frame(main_frame, bg="#0f172a")
content_frame.pack(fill="both", expand=True)

# Left - Camera
left_frame = tk.Frame(content_frame, bg="#0f172a")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

camera_header = tk.Label(left_frame, text="👁️ Live Emotion Detection", bg="#1e293b", 
                        fg="white", font=("Segoe UI", 13, "bold"), pady=12)
camera_header.pack(fill="x")

camera_label = tk.Label(left_frame, bg="black", relief="ridge", bd=4)
camera_label.pack(fill="both", expand=True, pady=(10, 0))

# Right - Chat
right_frame = tk.Frame(content_frame, bg="#0f172a")
right_frame.pack(side="right", fill="both", expand=True)

chat_header = tk.Label(right_frame, text="💬 AI Conversation", bg="#1e293b", 
                      fg="white", font=("Segoe UI", 13, "bold"), pady=12)
chat_header.pack(fill="x")

chat_container = tk.Frame(right_frame, bg="#334155")
chat_container.pack(fill="both", expand=True, pady=(10, 0))

chat_scrollbar = tk.Scrollbar(chat_container)
chat_scrollbar.pack(side="right", fill="y")

chat_log = tk.Text(chat_container, height=22, width=50, bg="#f1f5f9", fg="#0f172a", 
                   font=("Segoe UI", 10), wrap="word", relief="flat",
                   yscrollcommand=chat_scrollbar.set)
chat_log.pack(side="left", fill="both", expand=True, padx=3, pady=3)
chat_scrollbar.config(command=chat_log.yview)

# Initial greeting with voice
welcome_msg = "Hello! I'm your emotion-aware AI companion. I can see how you're feeling and respond accordingly. How are you today?"
append_chat_and_speak("Bot", welcome_msg, speak_it=True)

# Control Panel
control_frame = tk.Frame(main_frame, bg="#1e293b", pady=20)
control_frame.pack(fill="x", pady=(25, 0))

# Input area
input_frame = tk.Frame(control_frame, bg="#1e293b")
input_frame.pack(fill="x", padx=30)

user_entry = tk.Entry(input_frame, font=("Segoe UI", 13), 
                     relief="sunken", bd=3, bg="#e2e8f0", fg="#0f172a",
                     width=45)
user_entry.pack(side="left", fill="x", expand=True, padx=(0, 20))
user_entry.insert(0, "Type your message here...")

def clear_placeholder(event):
    if user_entry.get() == "Type your message here...":
        user_entry.delete(0, tk.END)
        user_entry.config(fg="#0f172a")

def restore_placeholder(event):
    if user_entry.get() == "":
        user_entry.insert(0, "Type your message here...")
        user_entry.config(fg="#64748b")

user_entry.bind("<FocusIn>", clear_placeholder)
user_entry.bind("<FocusOut>", restore_placeholder)
user_entry.config(fg="#64748b")

# Buttons
button_frame = tk.Frame(input_frame, bg="#1e293b")
button_frame.pack(side="right")

btn_style = {
    "font": ("Segoe UI", 11, "bold"),
    "padx": 22,
    "pady": 10,
    "bd": 0,
    "relief": "raised",
    "cursor": "hand2"
}

mic_btn = tk.Button(button_frame, text="🎤 Voice", command=start_listening,
                   bg="#ef4444", fg="white", activebackground="#dc2626",
                   **btn_style)
mic_btn.pack(side="left", padx=5)

speak_btn = tk.Button(button_frame, text="🔊 Speak", command=speak_last_message,
                     bg="#10b981", fg="white", activebackground="#059669",
                     **btn_style)
speak_btn.pack(side="left", padx=5)

send_btn = tk.Button(button_frame, text="📤 Send", command=send_message,
                    bg="#3b82f6", fg="white", activebackground="#2563eb",
                    **btn_style)
send_btn.pack(side="left", padx=5)

# Status
status_frame = tk.Frame(control_frame, bg="#1e293b", pady=12)
status_frame.pack(fill="x", padx=30)

status_label = tk.Label(status_frame, 
                       text=f"🤖 Model: {llm.current_model['name']} | 🔊 TTS: {'Ready ✓' if tts.engine else 'Offline'} | 🎭 Emotion: {'Active ✓' if model else 'Text-only'}",
                       bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 10))
status_label.pack()

instructions = tk.Label(status_frame, 
                       text="💡 Press Enter to send | 🎤 for voice (8s) | 🔊 to repeat last message",
                       bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 9))
instructions.pack(pady=(8, 0))

# ---------------------------
# Video Processing
# ---------------------------
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if model is not None:
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(tensor).argmax(dim=1).item()
                    if pred < len(classes):
                        detected_emotion = classes[pred]
                        emotion_buffer.append(detected_emotion)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                text_size = cv2.getTextSize(detected_emotion, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x+text_size[0]+10, y), (0, 255, 0), -1)
                cv2.putText(frame, detected_emotion, (x+5, y-10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)
                
            except:
                pass
            break

    emotion_state["face"] = smooth_emotion()
    current_emotion = get_final_emotion()
    
    emotion_colors = {
        "Happy": "#22c55e",
        "Sad": "#3b82f6",
        "Angry": "#ef4444",
        "Surprise": "#eab308",
        "Fear": "#8b5cf6",
        "Neutral": "#64748b"
    }
    
    emotion_color = emotion_colors.get(current_emotion, "#64748b")
    emotion_label.config(fg=emotion_color)
    emotion_var.set(f"🎭 Emotion: {current_emotion}")

    height, width = frame.shape[:2]
    display_width = 500
    display_height = int((display_width / width) * height)
    
    if display_height > 0:
        frame = cv2.resize(frame, (display_width, display_height))
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
    
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    root.after(50, update_frame)

# Start video feed
update_frame()

# Bind Enter key
root.bind('<Return>', lambda event: send_message())

# Test TTS at startup
print("Testing TTS system...")
speak("Emotion aware chatbot is now ready. I can see your emotions and respond accordingly.")

print("\n" + "="*70)
print("🤖 EMOTION-AWARE CHATBOT WITH TTS")
print("="*70)
print("System Status:")
print(f"  TTS Engine: {'Ready ✓' if tts.engine else 'Not available'}")
print(f"  Speech Queue: Active")
print(f"  Hugging Face Model: {llm.current_model['name']}")
print(f"  Emotion Detection: {'Active ✓' if model else 'Text-only'}")
print("="*70)
print("Voice Features:")
print("  1. All bot responses will be spoken")
print("  2. Click 🔊 to repeat last message")
print("  3. 8-second voice recording")
print("  4. Queue system prevents speech overlap")
print("="*70)
print("✅ System ready. Bot will greet you with voice.")
print("="*70 + "\n")

root.mainloop()

cap.release()
cv2.destroyAllWindows()