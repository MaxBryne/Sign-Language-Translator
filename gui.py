import datetime
import numpy as np
import tkinter as tk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

model = load_model('Sign_language_detector.h5')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

def valid_time():
    current_time = datetime.datetime.now().time()
    start_time = datetime.time(18, 0)
    end_time = datetime.time(22, 0)
    return start_time <= current_time <= end_time

def preprocess_image(image_path, img_size=(64, 64)):
    image = load_img(image_path, target_size=img_size)
    image = img_to_array(image)
    image = image/255.0
    return image

def predict_sign(image_path):
    if not valid_time():
        messagebox.showwarning("Time Restriction", "Prediction are only allowed between 6 PM to 10 PM")
        return

    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    prediction_class = np.argmax(prediction)
    result =  label_encoder.inverse_transform([prediction_class])[0]
    result_label.config(text=f"Prediction: {result}")
    print(result)

def upload_image():
    file = filedialog.askopenfilename()
    if file:
        img = Image.open(file)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image = img)
        panel.image = img
        
        predict_sign(file)
        
root = tk.Tk()
root.title("Sign Language Detection")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

panel = tk.Label(root)
panel.pack(pady=20)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=20)

root.mainloop()