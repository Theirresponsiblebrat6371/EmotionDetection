import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque, Counter
import speech_recognition as sr
import pyttsx3
import threading




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tts = pyttsx3.init()
tts.setProperty("rate", 165)   # speaking speed




checkpoint = torch.load("emotion_resnet.pth", map_location=device)
classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)




transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])





emotion_buffer = deque(maxlen=15)

def smooth_emotion():
    if len(emotion_buffer) == 0:
        return "Neutral"
    return Counter(emotion_buffer).most_common(1)[0][0]




topics = []

def add_topic(topic):
    if topic not in topics:
        topics.append(topic)
        if len(topics) > 2:
            topics.pop(0)





def generate_reply(text, emotion):
    text = text.lower()

    if "ai" in text:
        add_topic("AI")
        return "AI is fascinating. Are you working on a project?"

    if "college" in text or "exam" in text:
        add_topic("College")
        return "College life can be stressful. How are your studies going?"

    if "game" in text:
        add_topic("Games")
        return "Games are a great way to relax. What do you play?"

    if emotion == "sad":
        return "You seem a bit sad. Want to talk about it?"

    if emotion == "happy":
        return "You look happy! Something good happened?"

    if emotion == "angry":
        return "I sense some frustration. Want to explain?"

    return "Tell me more."





root = tk.Tk()
root.title("Emotion Aware Assistant")
root.geometry("950x750")
root.configure(bg="#111")


top_bar = tk.Frame(root, bg="#222", height=60)
top_bar.pack(fill="x")

emotion_var = tk.StringVar(value="Emotion: Detecting...")
topic_var = tk.StringVar(value="Topics: None")

emotion_label = tk.Label(
    top_bar, textvariable=emotion_var,
    fg="white", bg="#222",
    font=("Arial", 15, "bold")
)
emotion_label.pack(side="left", padx=20)

topic_label = tk.Label(
    top_bar, textvariable=topic_var,
    fg="white", bg="#222",
    font=("Arial", 14)
)
topic_label.pack(side="right", padx=20)




camera_label = tk.Label(root)
camera_label.pack(pady=20)





chat_frame = tk.Frame(root, bg="#111")
chat_frame.pack(fill="x", pady=10)

chat_log = tk.Label(
    chat_frame,
    text="Bot: Hello! I can see your emotions 😊",
    fg="white", bg="#111",
    font=("Arial", 12),
    justify="left", wraplength=800
)
chat_log.pack(anchor="w", padx=20)





input_frame = tk.Frame(root, bg="#111")
input_frame.pack(fill="x", pady=10)

user_entry = tk.Entry(
    input_frame, font=("Arial", 12), width=60
)
user_entry.pack(side="left", padx=20)

def send_message():
    user_text = user_entry.get()
    if not user_text:
        return

    emotion = smooth_emotion()
    reply = generate_reply(user_text, emotion)

    chat_log.config(
        text=f"You: {user_text}\nBot: {reply}"
    )

    topic_var.set(
        "Topics: " + " | ".join(topics) if topics else "Topics: None"
    )

    user_entry.delete(0, tk.END)

send_btn = tk.Button(
    input_frame, text="Send",
    command=send_message
)
send_btn.pack(side="left")






cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).item()
            emotion_buffer.append(classes[pred])

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        break

    emotion = smooth_emotion()
    emotion_var.set(f"Emotion: {emotion}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
