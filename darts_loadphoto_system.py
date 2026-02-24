# Last update: Feb. 25, 2026
# CBGAigop hahaha

# Haven't tried this code yet with an actual dartboard
# I'll try my best to look one during these days
# Updated: Now supports image input instead of live camera feed

import cv2
import numpy as np
import math


class DartScoringSystem:
    
    # -----------------------------
    # Initialization of values
    # -----------------------------
    def __init__(self, image_path):
        
        # Loads a dartboard image instead of using live camera feed
        # Make sure the image is in the same folder as this script
        self.image = cv2.imread(image_path)

        # ADDED: Check if image loaded correctly
        if self.image is None:
            raise ValueError("Image not found. Check file name/path.")

        # ADDED: Resize image to 960x960 for consistent window size
        self.image = cv2.resize(self.image, (960, 960))

        self.output = self.image.copy()

        # Calibration state
        self.calibrated = False 
        self.center = None
        self.radius = None

        # Ring boundaries (will be computed after calibration)
        self.inner_bull = None
        self.outer_bull = None
        self.triple_inner = None
        self.triple_outer = None
        self.double_inner = None

        # Create resizable window
        cv2.namedWindow("Dart Scoring System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dart Scoring System", 960, 960)

        # Mouse click event (used to simulate dart hit on image)
        cv2.setMouseCallback("Dart Scoring System", self.mouse_click)

    # -----------------------------
    # 1️ Detect Dartboard
    # -----------------------------
    def detect_board(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscaling
        blur = cv2.GaussianBlur(gray, (9, 9), 2) # Gaussian Blur for denoising
        # You can change the values/parameters until the optimal state is reached;
        # Just avoid doing it too much as this will water down to indiscernible form 

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=200,
            maxRadius=600
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            return (x, y, r)

        return None

    # -----------------------------
    # 2️ Compute Ring Boundaries
    # -----------------------------
    def compute_rings(self):
        r = self.radius
        self.inner_bull = int(r * 0.05)
        self.outer_bull = int(r * 0.1)
        self.triple_inner = int(r * 0.55)
        self.triple_outer = int(r * 0.6)
        self.double_inner = int(r * 0.9)

    # -----------------------------
    # 3️ Draw Board Overlay
    # -----------------------------
    def draw_board_overlay(self, frame):
        x, y = self.center
        r = self.radius

        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw scoring rings
        cv2.circle(frame, (x, y), self.inner_bull, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), self.outer_bull, (255, 255, 0), 2)
        cv2.circle(frame, (x, y), self.triple_inner, (0, 255, 255), 2)
        cv2.circle(frame, (x, y), self.triple_outer, (0, 255, 255), 2)
        cv2.circle(frame, (x, y), self.double_inner, (0, 0, 255), 2)

    # -----------------------------
    # 4️ Compute Score
    # -----------------------------
    def compute_score(self, dart_x, dart_y):
        cx, cy = self.center

        dx = dart_x - cx
        dy = dart_y - cy

        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Outside board
        if distance > self.radius:
            return 0

        # Bulls
        if distance <= self.inner_bull:
            return 50

        if distance <= self.outer_bull:
            return 25

        # Multiplier
        multiplier = 1

        if self.triple_inner <= distance <= self.triple_outer:
            multiplier = 3
        elif self.double_inner <= distance <= self.radius:
            multiplier = 2

        # IMPORTANT FIX:
        # Rotate angle so 0° is at TOP (20 sector)
        angle = math.degrees(math.atan2(-dy, dx))

        # Shift so 0° aligns with 20 (top)
        angle = (angle + 90) % 360

        sector = int(angle // 18)

        dart_order = [
            20,1,18,4,13,6,10,15,2,17,
            3,19,7,16,8,11,14,9,12,5
        ]

        base_score = dart_order[sector]

        return base_score * multiplier

    # -----------------------------
    # Mouse Click Event (Simulates Dart Throw)
    # -----------------------------
    def mouse_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            # First click = center
            if not self.calibrated and self.center is None:
                self.center = (x, y)
                print("Center set")

            # Second click = outer edge
            elif not self.calibrated and self.center is not None:
                cx, cy = self.center
                self.radius = int(math.sqrt((x - cx)**2 + (y - cy)**2))
                self.compute_rings()
                self.calibrated = True
                print("Calibration Complete")

            # After calibrated = score clicks
            elif self.calibrated:
                score = self.compute_score(x, y)
                print("Score:", score)

                cv2.circle(self.output, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(self.output, f"Score: {score}",
                            (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

    # -----------------------------
    # 5️ Main Loop
    # -----------------------------
    def run(self):
        while True:

            display = self.output.copy()

            if not self.calibrated:
                cv2.putText(display,
                            "Click CENTER then OUTER EDGE to calibrate",
                            (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            else:
                self.draw_board_overlay(display)

            cv2.imshow("Dart Scoring System", display)

            # Stop loop if window manually closed
            if cv2.getWindowProperty("Dart Scoring System", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                break

            if key == ord('q'):
                break

        cv2.destroyAllWindows()


# Run system
if __name__ == "__main__":
    system = DartScoringSystem("dartboard.png")  # Replace with your image filename whatever it is
    system.run()
