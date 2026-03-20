import cv2
import mediapipe as mp
import time
import pyautogui

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Finger detection
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


cap = cv2.VideoCapture(0)

p_time = 0
last_action_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

            label = handedness.classification[0].label

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(hand_landmarks)
            lm = hand_landmarks.landmark

            thumb_tip = lm[4]
            thumb_ip = lm[3]

            gesture = "UNKNOWN"

            # Gesture detection
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "FIST"

            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "OPEN HAND"

            elif (fingers == [1, 0, 0, 0, 0] and thumb_tip.y < thumb_ip.y):
                gesture = "THUMBS UP"

            elif (fingers == [1, 0, 0, 0, 0] and thumb_tip.y > thumb_ip.y):
                gesture = "THUMBS DOWN"

            # ===== ACTION CONTROL (with delay) =====
            if time.time() - last_action_time > 1:

                if gesture == "THUMBS UP":
                    pyautogui.press("volumeup")
                    last_action_time = time.time()

                elif gesture == "THUMBS DOWN":
                    pyautogui.press("volumedown")
                    last_action_time = time.time()

                elif gesture == "FIST":
                    pyautogui.press("space")  # Pause
                    last_action_time = time.time()

                elif gesture == "OPEN HAND":
                    pyautogui.press("space")  # Play
                    last_action_time = time.time()           

            # Display gesture
            h, w, _ = frame.shape
            cx, cy = int(lm[0].x * w), int(lm[0].y * h)

            cv2.putText(frame, f'{label}: {gesture}', (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) != 0 else 0
    p_time = c_time 

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Control System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()                      