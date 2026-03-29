import cv2
import mediapipe as mp
import socket

SERVER_IP = "0.0.0.0"
PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils 
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    hand_states = {"Right": "NONE", "Left": "NONE"}

    if results.multi_hand_landmarks:
        for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            label = hand_info.classification[0].label 
            
            tip_ids = [8, 12, 16, 20]
            fingers = [1 if hand_lms.landmark[i].y < hand_lms.landmark[i-2].y else 0 for i in tip_ids]
            state = "OPEN" if sum(fingers) >= 3 else "FIST"
            hand_states[label] = state

    msg = f"R:{hand_states['Right']}|L:{hand_states['Left']}"
    sock.sendto(msg.encode(), (SERVER_IP, PORT))

    cv2.putText(img, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Local Monitor with Bones", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
