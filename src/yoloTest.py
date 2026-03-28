import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Allow for 2 hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

tip_ids = [8, 12, 16, 20]
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) # Mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # Zip together landmarks and their labels (Left/Right)
        for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get the label (Note: MediaPipe labels are sometimes inverted when flipping)
            hand_label = hand_info.classification[0].label 
            
            fingers = []
            
            # Thumb Logic (X-axis)
            # For Right hand, thumb is 'open' if tip.x > joint.x
            # For Left hand, it's the opposite.
            # if hand_label == "Right":
            #     if hand_lms.landmark[4].x > hand_lms.landmark[3].x: fingers.append(1)
            #     else: fingers.append(0)
            # else:
            #     if hand_lms.landmark[4].x < hand_lms.landmark[3].x: fingers.append(1)
            #     else: fingers.append(0)

            # Four Fingers Logic (Y-axis)
            for id in tip_ids:
                if hand_lms.landmark[id].y < hand_lms.landmark[id - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)
            gesture = "FIST" if total == 0 else "OPEN" if total >= 4 else f"{total} UP"
            
            # Display label near the specific hand
            h, w, c = img.shape
            cx, cy = int(hand_lms.landmark[9].x * w), int(hand_lms.landmark[9].y * h)
            cv2.putText(img, f"{hand_label}: {gesture}", (cx-50, cy-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Multi-Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
