import os 
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
def run_detector():
        pth = "mask_detector.h5"
        if not os.path.exists(pth):
            print(f"Error: {pth} not found. Train the model first.")
            return  
        print("Loading mask detector")
        model = load_model(pth)
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + cascade_path
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier XML.")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam stream.")
            return
        print("Webcam active. Press 'escape key' on your keyboard to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed camera.")
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            for (x, y, w, h) in faces:
                roi_bgr = frame[y:y+h, x:x+w]
                if roi_bgr.size == 0:
                    continue
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                resized_roi = cv2.resize(roi_rgb, (224, 224))
                normalized_roi = resized_roi.astype("float32") / 255.0
                tensor_input = np.expand_dims(normalized_roi, axis=0)
                predictions = model.predict(tensor_input, verbose=0)[0]
                mask_score, no_mask_score = predictions[0], predictions[1]
                has_mask = mask_score > no_mask_score
                confidence = mask_score if has_mask else no_mask_score
                label_text = f"{'Mask' if has_mask else 'No Mask'}: {confidence * 100:.1f}%"
                box_color = (0, 255, 0) if has_mask else (0, 0, 255)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.imshow("Real-Time Fave Mask Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
     run_detector()
