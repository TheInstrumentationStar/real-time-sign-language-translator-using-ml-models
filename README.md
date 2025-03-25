# real-time-sign-language-translator-using-ml-models

this is the main.py file

import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam


# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Labels dictionary

labels_dict = {0: 'Good', 1:'morning', 2:'Head', 3:'headaches', 4:'and', 5:'Dizziness', 
               6:'Two', 7:'days', 8:'Medium', 9:'Sounds', 10: 'No', 11:'Light', 12:'Yes',
                 13:'Too Much', 14:'Work', 15:'Sleep', 16:'Okay', 17:'Thank You!', 18:'Severe', 19:"Bearable",
                 20:"A",21: "B",22: "C",23: "D",24: "E",25: "F",26: "G",27: "H",28: "I",29: "J",30: "K",31: "L",
                 32: "M",33: "N",34: "O",35: "P",36: "Q",37: "R",38: "S",39: "T",40: "U",41: "V",42: "W",43: "X",
                 44: "Y",45: "Z",46:"nothing",47:"space",48:"del"}
  # Update labels as per your dataset

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not detected.")

# Create a virtual camera
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_, y_ = [], []

        if results.multi_hand_landmarks:
            hands_data = []

            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    hand_data.append(landmark.x - min(x_))
                    hand_data.append(landmark.y - min(y_))

                hands_data.append(hand_data)

            if len(hands_data) == 2:
                data_aux.extend(hands_data[0])
                data_aux.extend(hands_data[1])
            elif len(hands_data) == 1:
                data_aux.extend(hands_data[0])
                data_aux.extend([0] * len(hands_data[0]))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            text = predicted_character
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 30

            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cam.send(frame)
        cam.sleep_until_next_frame()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
