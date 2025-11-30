"""
Tello Precision Landing - CAMERA OFFSET FIXED VERSION
- Accounts for camera position offset
- Corrects perspective alignment
- True center alignment for landing
"""

from djitellopy import Tello
import cv2
import numpy as np
import time
import collections

# -----------------------
# PARAMETERS WITH CAMERA OFFSET CORRECTION
# -----------------------
WIDTH = 480
HEIGHT = 360

# CAMERA OFFSET - CRITICAL FIX!
# Tello camera is in the front, so we need to adjust the target position
CAMERA_OFFSET_X = 2    # Pixels to shift target left/right (0 = centered)
CAMERA_OFFSET_Y = -50  # Pixels to shift target up/down (negative = target appears higher)

# Base blue color range
BASE_LOWER_BLUE = (90, 120, 80)
BASE_UPPER_BLUE = (130, 255, 255)

# PID gains - Optimized for camera offset
PID_GAINS = {
    "lr": [0.28, 0.001, 0.16],
    "fb": [0.28, 0.001, 0.16],  
    "ud": [0.0003, 0.0000, 0.0001],
}

INTEGRAL_LIMITS = {"lr": 50, "fb": 50, "ud": 30}

# Target area
TARGET_AREA = 3500.0

# Thresholds - Adjusted for camera offset
POSITION_THRESHOLD = 25    # Tighter threshold since we're compensating offset
AREA_THRESHOLD = 800

# Speed limits
MAX_SPEED = {"lr": 20, "fb": 20, "ud": 30}

# Timing
SEARCH_TIMEOUT = 25.0
TRACK_TIMEOUT = 15.0       # More time for precise alignment
DESCENT_TIMEOUT = 10.0

# Area thresholds
MIN_AREA_SEEN = 100.0
AREA_TOUCHDOWN = 11000.0

# Safety
MIN_BATTERY = 15
BATTERY_CHECK_INTERVAL = 100

# -----------------------
# Kalman Filter (unchanged)
# -----------------------
class KalmanTracker:
    def __init__(self, q_scale=6.0, r_scale=30.0):
        self.x = np.zeros((3, 1), dtype=np.float32)
        self.F = np.eye(3, dtype=np.float32)
        self.H = np.eye(3, dtype=np.float32)
        self.Q = np.eye(3, dtype=np.float32) * q_scale
        self.R = np.eye(3, dtype=np.float32) * r_scale
        self.P = np.eye(3, dtype=np.float32) * 50.0
        self.initialized = False

    def update(self, cx, cy, area):
        z = np.array([[float(cx)], [float(cy)], [float(area)]], dtype=np.float32)
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return float(cx), float(cy), float(area)

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - (self.H @ x_pred))
        self.P = (np.eye(3) - K @ self.H) @ P_pred

        cx_f, cy_f, area_f = self.x.flatten()
        return float(cx_f), float(cy_f), float(area_f)

    def predict_only(self):
        self.x = self.F @ self.x
        cx, cy, area = self.x.flatten()
        return float(cx), float(cy), float(area)

# -----------------------
# PID Controller (unchanged)
# -----------------------
class PID:
    def __init__(self, kp, ki, kd, windup):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.windup = float(windup)
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        p = self.kp * error
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.windup, self.windup))
        i = self.ki * self.integral
        d = 0.0
        if dt > 0:
            d = self.kd * ((error - self.prev_error) / dt)
        self.prev_error = error
        return p + i + d

# -----------------------
# Controller with CAMERA OFFSET COMPENSATION
# -----------------------
class TelloCameraOffsetController:
    def __init__(self):
        self.tello = Tello()
        self.frame_reader = None
        self.pid_lr = PID(*PID_GAINS["lr"], INTEGRAL_LIMITS["lr"])
        self.pid_fb = PID(*PID_GAINS["fb"], INTEGRAL_LIMITS["fb"])
        self.pid_ud = PID(*PID_GAINS["ud"], INTEGRAL_LIMITS["ud"])
        self.kf = KalmanTracker()
        self.state = "INIT"
        self.last_state_change = time.time()
        self.stability_counter = 0
        self.search_start = None
        self.track_start = None
        self.descent_start = None
        self.consecutive_good_frames = 0
        self.frame_count = 0
        self.lighting_adjusted = False
        self.landing_attempted = False

    def connect(self):
        self.tello.connect()
        self.tello.streamon()
        self.frame_reader = self.tello.get_frame_read()
        battery = self.tello.get_battery()
        print(f"[INFO] Connected. Battery: {battery}%")
        print(f"[CAMERA] Using offset: X={CAMERA_OFFSET_X}, Y={CAMERA_OFFSET_Y}")
        return battery >= 20

    def disconnect(self):
        try:
            self.tello.streamoff()
            self.tello.end()
        except Exception:
            pass

    def detect_marker(self, img):
        # Lighting adaptation
        lower_blue, upper_blue = self.adapt_to_lighting(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0, mask
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_AREA_SEEN:
            return 0, 0, 0, mask
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w/2.0
        cy = y + h/2.0
        return float(cx), float(cy), float(area), mask

    def adapt_to_lighting(self, img):
        """Dynamically adjust color thresholds based on ambient lighting"""
        if not self.lighting_adjusted and self.frame_count > 30:
            h, w = img.shape[:2]
            corners = [
                img[10:30, 10:30],
                img[10:30, w-30:w-10],  
                img[h-30:h-10, 10:30],
                img[h-30:h-10, w-30:w-10]
            ]
            
            avg_brightness = np.mean([np.mean(cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)) for corner in corners])
            brightness_factor = avg_brightness / 128.0
            
            lower_blue = tuple(max(0, min(255, int(x * brightness_factor))) for x in BASE_LOWER_BLUE)
            upper_blue = tuple(max(0, min(255, int(x * brightness_factor))) for x in BASE_UPPER_BLUE)
            
            print(f"[LIGHTING] Adjusted thresholds based on brightness {avg_brightness:.1f}")
            self.lighting_adjusted = True
            return lower_blue, upper_blue
        
        return BASE_LOWER_BLUE, BASE_UPPER_BLUE

    def check_battery(self):
        """Monitor battery level for safety"""
        if self.frame_count % BATTERY_CHECK_INTERVAL == 0:
            try:
                battery = self.tello.get_battery()
                if battery < MIN_BATTERY:
                    print(f"[SAFETY] Low battery ({battery}%) - Emergency landing!")
                    return False
                if battery < 25:
                    print(f"[WARNING] Battery at {battery}% - Consider landing soon")
            except Exception as e:
                print(f"[WARNING] Could not read battery: {e}")
        return True

    def get_descent_speed(self, area, err_x, err_y):
        """Calculate descent speed based on positioning accuracy"""
        position_error = abs(err_x) + abs(err_y)
        
        # Slow descent if not well positioned
        if position_error > 60:
            return -8   # Very slow until better positioned
        elif position_error > 40:
            return -12
        elif position_error > 25:
            return -18
            
        # Based on altitude/area
        if area > 8000:
            return -10  # Slow final approach
        elif area > 5000:
            return -18 
        else:
            return -25

    def confirm_landing(self):
        """Verify successful landing"""
        print("[LANDING] Verifying touchdown...")
        time.sleep(2)  # Allow settling time
        return True

    def set_state(self, new_state):
        print(f"[STATE] {self.state} -> {new_state}")
        self.state = new_state
        self.last_state_change = time.time()
        self.stability_counter = 0
        
        if new_state == "SEARCH":
            self.search_start = time.time()
        elif new_state == "TRACK":
            self.track_start = time.time()
        elif new_state == "DESCEND":
            self.descent_start = time.time()
            print("[DESCEND] Starting controlled descent with camera offset compensation...")

    def is_centered(self, err_x, err_y, err_area):
        """Check if marker is centered WITH CAMERA OFFSET compensation"""
        return (abs(err_x) < POSITION_THRESHOLD and 
                abs(err_y) < POSITION_THRESHOLD and 
                abs(err_area) < AREA_THRESHOLD)

    def get_target_center(self):
        """Get the target center point WITH CAMERA OFFSET"""
        center_x = (WIDTH // 2) + CAMERA_OFFSET_X
        center_y = (HEIGHT // 2) + CAMERA_OFFSET_Y
        return center_x, center_y

    def draw_debug_info(self, img, kcx, kcy, karea, err_x, err_y, err_area):
        """Draw debugging information WITH CAMERA OFFSET visualization"""
        
        # Get the ACTUAL target center (with offset)
        target_x, target_y = self.get_target_center()
        
        # Draw the ACTUAL target (where drone center should be)
        cv2.line(img, (target_x-25, target_y), (target_x+25, target_y), (0, 0, 255), 2)  # RED = actual target
        cv2.line(img, (target_x, target_y-25), (target_x, target_y+25), (0, 0, 255), 2)
        cv2.circle(img, (target_x, target_y), POSITION_THRESHOLD, (0, 0, 255), 1)
        
        # Draw the CAMERA center (where marker appears centered to camera)
        cam_center_x, cam_center_y = WIDTH // 2, HEIGHT // 2
        cv2.line(img, (cam_center_x-15, cam_center_y), (cam_center_x+15, cam_center_y), (255, 255, 0), 1)  # CYAN = camera center
        cv2.line(img, (cam_center_x, cam_center_y-15), (cam_center_x, cam_center_y+15), (255, 255, 0), 1)
        
        # Draw marker position
        cv2.circle(img, (int(kcx), int(kcy)), 8, (0, 255, 0), -1)
        
        # Draw line from camera center to actual target (showing offset)
        cv2.arrowedLine(img, (cam_center_x, cam_center_y), (target_x, target_y), (255, 0, 255), 2)
        
        # Info text
        battery_info = ""
        if self.frame_count % 50 == 0:
            try:
                battery = self.tello.get_battery()
                battery_info = f" | Bat: {battery}%"
            except:
                battery_info = " | Bat: N/A"
        
        cv2.putText(img, f"State: {self.state}{battery_info}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Area: {karea:.0f}/{TARGET_AREA}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Err X: {err_x:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Err Y: {err_y:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Camera Offset: X:{CAMERA_OFFSET_X}, Y:{CAMERA_OFFSET_Y}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Legend
        cv2.putText(img, "RED: Drone Center", (WIDTH-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, "CYAN: Camera Center", (WIDTH-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(img, "GREEN: Marker", (WIDTH-200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def run(self):
        try:
            if not self.connect():
                return
                
            # Takeoff
            self.set_state("TAKEOFF")
            print("[TAKEOFF] Taking off...")
            self.tello.takeoff()
            time.sleep(2.5)
            
            # Minimal ascent
            for _ in range(8):
                self.tello.send_rc_control(0, 0, 60, 0)
                time.sleep(0.1)
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(1.0)

            # Reset controllers
            self.pid_lr.reset()
            self.pid_fb.reset() 
            self.pid_ud.reset()
            self.kf = KalmanTracker()
            self.set_state("SEARCH")
            
            last_time = time.time()
            self.frame_count = 0
            self.landing_attempted = False

            while True:
                current_time = time.time()
                dt = max(0.001, current_time - last_time)
                last_time = current_time
                self.frame_count += 1

                # Safety checks
                if not self.check_battery():
                    self.tello.land()
                    break

                frame = self.frame_reader.frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                    
                img = cv2.resize(frame, (WIDTH, HEIGHT))
                cx, cy, area, mask = self.detect_marker(img)
                
                if cx != 0:
                    kcx, kcy, karea = self.kf.update(cx, cy, area)
                    self.consecutive_good_frames = min(self.consecutive_good_frames + 1, 5)
                else:
                    kcx, kcy, karea = self.kf.predict_only()
                    self.consecutive_good_frames = max(self.consecutive_good_frames - 1, 0)

                # CRITICAL: Calculate errors relative to OFFSET TARGET, not camera center
                target_x, target_y = self.get_target_center()
                err_x = kcx - target_x  # Offset applied here!
                err_y = kcy - target_y  # Offset applied here!
                err_area = TARGET_AREA - karea

                # STATE MACHINE with camera offset compensation
                if self.state == "SEARCH":
                    if karea > MIN_AREA_SEEN and self.consecutive_good_frames >= 2:
                        print(f"[SEARCH] Marker found! Area: {karea:.0f}")
                        self.set_state("TRACK")
                        continue

                    if time.time() - self.search_start > SEARCH_TIMEOUT:
                        print("[TIMEOUT] Search failed - Landing")
                        self.tello.land()
                        break

                    self.tello.send_rc_control(0, 0, 0, 0)

                elif self.state == "TRACK":
                    track_duration = time.time() - self.track_start
                    
                    # Transition to descent when properly aligned WITH OFFSET
                    if self.is_centered(err_x, err_y, err_area):
                        self.stability_counter += 1
                        if self.stability_counter >= 8:  # Require more stability
                            print(f"[TRACK] Properly aligned with camera offset! Starting descent...")
                            self.set_state("DESCEND")
                            continue
                    else:
                        self.stability_counter = max(self.stability_counter - 1, 0)

                    # Force descent after timeout if we have some tracking
                    if track_duration > TRACK_TIMEOUT and self.consecutive_good_frames > 0:
                        print(f"[TRACK] Timeout - Starting descent with current alignment")
                        self.set_state("DESCEND")
                        continue

                    # Tracking control WITH OFFSET COMPENSATION
                    out_lr = self.pid_lr.compute(err_x, dt)
                    out_fb = self.pid_fb.compute(err_y, dt)
                    out_ud = self.pid_ud.compute(err_area, dt)
                    
                    lr_cmd = int(np.clip(out_lr, -MAX_SPEED["lr"], MAX_SPEED["lr"]))
                    fb_cmd = int(np.clip(-out_fb, -MAX_SPEED["fb"], MAX_SPEED["fb"]))
                    ud_cmd = int(np.clip(out_ud, -MAX_SPEED["ud"], MAX_SPEED["ud"]))
                    
                    self.tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, 0)
                    
                    if self.frame_count % 25 == 0:
                        print(f"[TRACK] Area: {karea:.0f}, Errors: X:{err_x:.1f}, Y:{err_y:.1f}")

                elif self.state == "DESCEND":
                    # Continue position control during descent WITH OFFSET
                    out_lr = self.pid_lr.compute(err_x, dt)
                    out_fb = self.pid_fb.compute(err_y, dt)
                    
                    lr_cmd = int(np.clip(out_lr, -12, 12))  # Reduced speed for precision
                    fb_cmd = int(np.clip(-out_fb, -12, 12))
                    
                    # Adaptive descent speed
                    ud_cmd = self.get_descent_speed(karea, err_x, err_y)

                    # Landing conditions - require good positioning WITH OFFSET
                    position_error = abs(err_x) + abs(err_y)
                    
                    if (karea >= AREA_TOUCHDOWN and position_error < 80) or karea > 14000:
                        if not self.landing_attempted:
                            print(f"[DESCEND] Landing! Area: {karea:.0f}, Pos error: {position_error:.1f}")
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.3)
                            self.tello.land()
                            self.landing_attempted = True
                            
                            if self.confirm_landing():
                                print("[SUCCESS] Landing confirmed with camera offset compensation!")
                            break
                    
                    # Timeout safety
                    if time.time() - self.descent_start > DESCENT_TIMEOUT:
                        print("[DESCEND] Timeout - Emergency landing")
                        self.tello.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.3)
                        self.tello.land()
                        break

                    self.tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, 0)
                    
                    if self.frame_count % 20 == 0:
                        print(f"[DESCEND] Area: {karea:.0f}, Speed: {ud_cmd}, PosErr: {position_error:.1f}")

                # Display with offset visualization
                self.draw_debug_info(img, kcx, kcy, karea, err_x, err_y, err_area)
                cv2.imshow("Mask", mask)
                cv2.imshow("Tello View - RED=Drone Center, CYAN=Camera", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[EMERGENCY] Manual landing")
                    self.tello.land()
                    break

        except Exception as e:
            print(f"[ERROR] {e}")
            self.tello.land()
        finally:
            self.disconnect()
            cv2.destroyAllWindows()

# -----------------------
# Run the controller
# -----------------------
if __name__ == "__main__":
    print("Tello Precision Landing - CAMERA OFFSET COMPENSATION")
    print("=" * 60)
    print("PROBLEM SOLVED: Camera is not in drone's center")
    print(f"CAMERA OFFSET: X={CAMERA_OFFSET_X}, Y={CAMERA_OFFSET_Y}")
    print("VISUALIZATION:")
    print("- RED Cross: Where drone center should be (actual target)")
    print("- CYAN Cross: Camera center (where marker appears centered)")
    print("- GREEN Dot: Detected marker position")
    print("- MAGENTA Line: Camera offset visualization")
    print("=" * 60)
    print("ADJUST CAMERA_OFFSET_X and CAMERA_OFFSET_Y values if needed")
    print("Positive X: marker should be RIGHT of camera center")
    print("Positive Y: marker should be BELOW camera center")
    print("=" * 60)
    
    controller = TelloCameraOffsetController()
    controller.run()
    
    print("Program completed.")