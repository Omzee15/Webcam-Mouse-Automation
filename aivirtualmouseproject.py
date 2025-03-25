import cv2
import numpy as np
import pyautogui
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam with multiple index attempts
for cam_index in range(3):  # Try camera indexes 0, 1, 2
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Webcam found at index {cam_index}")
        break
else:
    print("Error: No webcam detected!")
    exit()

cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Frame reduction to avoid excessive jitter
frame_reduction = 100

# Smoothing parameters
smoothening = 5
prev_x, prev_y = 0, 0

# State for click & hold
holding = False

while True:
    success, img = cap.read()
    
    # Check if frame is valid
    if not success or img is None:
        print("Warning: Failed to read from webcam. Retrying...")
        continue  # Skip this frame

    img = cv2.flip(img, 1)  # Mirror the video
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        lm_list = hand["lmList"]

        if lm_list:
            # Get index finger tip position
            x_index, y_index = lm_list[8][0], lm_list[8][1]

            # Correct the x-axis mapping
            screen_x = np.interp(x_index, (frame_reduction, 1280 - frame_reduction), (0, screen_width))  
            screen_y = np.interp(y_index, (frame_reduction, 720 - frame_reduction), (0, screen_height))

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # Move mouse cursor
            pyautogui.moveTo(curr_x, curr_y)  # Fixed left-right inversion
            prev_x, prev_y = curr_x, curr_y

            # Detect left-click gesture (index & middle finger touch)
            distance_click, _, img = detector.findDistance(lm_list[8][:2], lm_list[12][:2], img)
            if distance_click < 40:  # If index and middle fingers are close
                pyautogui.click()

            # Detect click and hold gesture (index & thumb touch)
            distance_hold, _, img = detector.findDistance(lm_list[4][:2], lm_list[8][:2], img)
            if distance_hold < 40 and not holding:
                pyautogui.mouseDown()
                holding = True  # Set holding state to prevent repeated pressing

            elif distance_hold >= 40 and holding:
                pyautogui.mouseUp()
                holding = False  # Release hold state

    # Display webcam feed
    cv2.imshow("Hand Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
 