import cv2
import sys
import numpy as np
import tkinter as tk
from GUI import Ui_MainWindow
from collections import deque
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow

LSTM_model = load_model("D:/DoAn3/HumanAction/Demo/Model3.h5")
CLASSES_LIST = ["Archery", "Biking", "HorseRiding", "WalkingWithDog", "BodyWeightSquats", "Bowling",
                 "BoxingPunchingBag", "Diving", "Drumming", "GolfSwing"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20


def predict_on_video(video_file_path):
    # Initialize the VideoCapture object to read from the video file
    show_video = False
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output.avi', fourcc, 30, (original_video_width, original_video_height))

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    predicted_probability = 0


    # Khởi tạo VideoCapture
    video_reader = cv2.VideoCapture(video_file_path)
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        if show_video:
            # Display the predicted action and probability on the video
            cv2.imshow('Action Recognition', frame)
        # Predict the action on each frame and display the video
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LSTM_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            predicted_probability = predicted_labels_probabilities[predicted_label] * 100

        # Display the predicted action and probability on the video
        cv2.putText(frame, f'Predicted Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Probability: {predicted_probability:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)
        
        # Write the frame to the output video
        video_writer.write(frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Release the VideoCapture and VideoWriter objects and close the window
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

def predict_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_img = resized_img / 255
    img_sequence = np.stack([normalized_img] * SEQUENCE_LENGTH, axis=0)
    img_sequence = np.expand_dims(img_sequence, axis=0)

    # Make predictions
    predicted_labels_probabilities = LSTM_model.predict(img_sequence)[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    predicted_probability = predicted_labels_probabilities[predicted_label] * 100

    # Overlay the prediction on the image
    overlay_text = f'Predicted Action: {predicted_class_name} \n Probability: {predicted_probability:.2f}%'
    cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with the overlay
    cv2.imshow("Image with Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.on_video_button_clicked)
        self.ui.pushButton_2.clicked.connect(self.on_image_button_clicked)
        self.ui.pushButton_3.clicked.connect(self.on_exit_button_clicked)

    def on_video_button_clicked(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            predict_on_video(file_path)

    def on_image_button_clicked(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            predict_image(file_path)

    def on_exit_button_clicked(self):
        QApplication.quit()
        print("Exit button clicked")


def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
