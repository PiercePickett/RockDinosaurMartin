import cv2
import mediapipe as mp
import socket
import time

# --- 1. Network Setup ---
SERVER_IP = "127.0.0.1"
PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- 2. MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# --- 3. Timing + State ---
DWELL_TIME = 4.0

last_gesture = None
gesture_start_time = 0
gesture_locked = False   # 🔑 prevents repeat firing

command_mode = False
command_bits = [0, 0, 0, 0]  # [R, G, B, Y]

print(f"Streaming to {SERVER_IP}:{PORT}...")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb.flags.writeable = False
    results = hands.process(img_rgb)
    img_rgb.flags.writeable = True

    hand_counts = {"Right": 0, "Left": 0}

    # --- Hand Detection ---
    if results.multi_hand_landmarks:
        for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            label = hand_info.classification[0].label

            # Fingers (index → pinky)
            tip_ids = [8, 12, 16, 20]
            fingers = [1 if hand_lms.landmark[i].y < hand_lms.landmark[i-2].y else 0 for i in tip_ids]

            # Thumb
            t_tip = hand_lms.landmark[4].x
            t_base = hand_lms.landmark[2].x
            if label == "Right":
                fingers.append(1 if t_tip < t_base else 0)
            else:
                fingers.append(1 if t_tip > t_base else 0)

            hand_counts[label] = sum(fingers)

    # --- Gesture String ---
    msg = f"{hand_counts['Right']}/{hand_counts['Left']}"
    current_gesture = msg
    now = time.time()

    # --- Dwell + State Machine ---
    if current_gesture != last_gesture:
        last_gesture = current_gesture
        gesture_start_time = now
        gesture_locked = False   # allow new gesture
    else:
        if (now - gesture_start_time) >= DWELL_TIME and not gesture_locked:
            gesture_locked = True  # 🔒 fire once

            print(f"Confirmed: {current_gesture}")

            # --- ENTER ATTENTION ---
            if current_gesture == "4/4" and not command_mode:
                command_mode = True
                command_bits = [0, 0, 0, 0]
                print(">>> ENTERED ATTENTION MODE <<<")

            # --- EDIT MODE ---
            elif command_mode:

                if current_gesture == "1/0":   # RED
                    command_bits = [1, 0, 0, 0]

                elif current_gesture == "3/0": # GREEN
                    command_bits = [0, 1, 0, 0]

                elif current_gesture == "4/0": # BLUE
                    command_bits = [0, 0, 1, 0]

                elif current_gesture == "2/0": # YELLOW
                    command_bits = [0, 0, 0, 1]

                elif current_gesture == "0/4": # INVERT
                    command_bits = [1 - b for b in command_bits]

                elif current_gesture == "0/0": # NONE
                    command_bits = [0, 0, 0, 0]

                elif current_gesture == "3/3": # DELETE
                    command_bits = [0, 0, 0, 0]
                    print("Command Cleared")

                elif current_gesture == "2/2": # SEND
                    bit_string = "".join(map(str, command_bits))
                    sock.sendto(bit_string.encode(), (SERVER_IP, PORT))
                    print(f">>> SENT: {bit_string} <<<")

                    command_mode = False  # exit mode

    # --- UI ---
    cv2.putText(img, f"Gesture: {msg}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"Mode: {'ATTENTION' if command_mode else 'IDLE'}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(img, f"Bits: {''.join(map(str, command_bits))}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Gesture Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()