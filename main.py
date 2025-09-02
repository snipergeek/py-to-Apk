import cv2
import numpy as np
import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

# --- Base de données ---
faces_dir = "."  # toutes les images sont à la racine
known_face_encodings = []
known_face_names = []

def load_faces():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for filename in os.listdir(faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if filename.lower() == "main.py":
                continue
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(faces_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                continue
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))
            known_face_encodings.append(face_resized)
            known_face_names.append(name)

load_faces()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Seuil initial ---
threshold = 5000

def compare_faces(face, known_faces, threshold=threshold):
    min_diff = float('inf')
    name_match = None
    for i, known_face in enumerate(known_faces):
        diff = np.mean(np.abs(face.astype("float") - known_face.astype("float")))
        if diff < min_diff:
            min_diff = diff
            name_match = known_face_names[i]
    if min_diff < threshold:
        return name_match
    else:
        return None

def calibrer_seuil(face_resized, known_faces):
    diffs = [np.mean(np.abs(face_resized.astype("float") - known_face.astype("float"))) for known_face in known_faces]
    seuil = min(diffs) * 1.5
    print(f"Seuil calibré : {seuil}")
    return seuil

# --- Kivy App ---
class FaceApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.img1 = Image()
        layout = BoxLayout(orientation='vertical')
        
        # Boutons
        btn_calibrate = Button(text="Calibrer Seuil", size_hint=(1, 0.1))
        btn_calibrate.bind(on_press=self.calibrate_threshold)
        btn_add_person = Button(text="Ajouter Personne", size_hint=(1, 0.1))
        btn_add_person.bind(on_press=self.add_person)
        
        layout.add_widget(self.img1)
        layout.add_widget(btn_calibrate)
        layout.add_widget(btn_add_person)
        
        Clock.schedule_interval(self.update, 1.0/30.0)
        return layout

    def calibrate_threshold(self, instance):
        ret, frame = self.capture.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))
            global threshold
            threshold = calibrer_seuil(face_resized, known_face_encodings)

    def add_person(self, instance):
        ret, frame = self.capture.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return
        
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (100, 100))
        
        # Popup pour entrer le nom
        popup_layout = BoxLayout(orientation='vertical')
        input_name = TextInput(hint_text='Nom de la personne', multiline=False)
        btn_save = Button(text='Enregistrer')
        popup_layout.add_widget(Label(text="Entrez le nom de la personne :"))
        popup_layout.add_widget(input_name)
        popup_layout.add_widget(btn_save)
        popup = Popup(title="Ajouter une personne", content=popup_layout, size_hint=(0.8, 0.5))
        
        def save_person(instance_button):
            name = input_name.text.strip()
            if name:
                filename = f"{name}.jpg"
                cv2.imwrite(filename, face_resized)
                known_face_encodings.append(face_resized)
                known_face_names.append(name)
                print(f"Personne '{name}' ajoutée !")
            popup.dismiss()
        
        btn_save.bind(on_press=save_person)
        popup.open()

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))
            match = compare_faces(face_resized, known_face_encodings, threshold)

            if match:
                color = (0, 255, 0)
                label = match
            else:
                color = (0, 0, 255)
                label = "Inconnu"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = image_texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    FaceApp().run()
