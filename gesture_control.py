import cv2
import mediapipe as mp
import pyautogui
import math
from pynput.mouse import Button, Controller

mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def get_finger_angles(landmarks):
    """Calculate bending angles for index and middle fingers"""
    index_angle = math.degrees(math.atan2(landmarks[8][1]-landmarks[6][1], 
                                         landmarks[8][0]-landmarks[6][0]) - 
                              math.atan2(landmarks[5][1]-landmarks[6][1],
                                         landmarks[5][0]-landmarks[6][0]))
    
    middle_angle = math.degrees(math.atan2(landmarks[12][1]-landmarks[10][1], 
                                          landmarks[12][0]-landmarks[10][0]) - 
                               math.atan2(landmarks[9][1]-landmarks[10][1],
                                          landmarks[9][0]-landmarks[10][0]))
    return abs(index_angle), abs(middle_angle)

def detect_clicks(frame, landmarks):
    index_angle, middle_angle = get_finger_angles(landmarks)
    
    # 👇 Index bent (angle > 120°) + Middle straight (angle < 60°) → Left Click
    if index_angle > 120 and middle_angle < 60:
        mouse.click(Button.left)
        cv2.putText(frame, "LEFT CLICK", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return "👈 LEFT CLICK"
    
    # 🖕 Middle bent (angle > 120°) + Index straight (angle < 60°) → Right Click
    elif middle_angle > 120 and index_angle < 60:
        mouse.click(Button.right)
        cv2.putText(frame, "RIGHT CLICK", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return "👉 RIGHT CLICK"
    
    return ""

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in hand.landmark]
            
            # Visual finger state indicators
            index_angle, middle_angle = get_finger_angles(landmarks)
            cv2.putText(frame, f"Index: {int(index_angle)}°", (50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(frame, f"Middle: {int(middle_angle)}°", (50, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            
            # Detect clicks
            gesture_text = detect_clicks(frame, landmarks)
            
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS)
            
        cv2.imshow('Gesture Controls', frame)
        if cv2.waitKey(1) == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()