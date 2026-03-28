import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self):
        # Set up mediapipe hands solution
        self.mpHands = mp.solutions.hands
        # Create the hand detector
        self.hands = self.mpHands.Hands()
        # Set up drawing utilities for landmarks
        self.mpdraw = mp.solutions.drawing_utils

    def find_hands(self, img):
        # Mediapipe requires RGB, OpenCV gives BGR so we convert
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the frame and store results on self so find_positions can use them
        self.results = self.hands.process(imgRGB)

        # If hands are detected, draw landmarks and connections on the image
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                self.mpdraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_positions(self, img):
        # Empty list to store landmark positions
        lm_list = []

        if self.results.multi_hand_landmarks:
            # Loop over landmarks of the first detected hand (index 0)
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, c = img.shape
                # Convert normalized coordinates (0-1) to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

        # Returns a list of [id, x, y] for each of the 21 landmarks
        return lm_list

