import cv2
import time
import json
import os
import threading
import winsound

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

CONFIG_FILE = os.path.join(
    ROOT_DIR,
    "config.json"
)

DEFAULT_CONFIG = {
    "driver_camera_index": 0,
    "drowsiness_seconds": 2.0,
    "distraction_seconds": 3.0,
    "yawn_limit": 4,

    # Braking starts after 3 violations
    "braking_violation_limit": 3,

    # Police information after 5 total violations
    "police_information_limit": 5,

    # Emergency sound after repeated violations
    "emergency_violation_limit": 2
}


try:

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        CONFIG = DEFAULT_CONFIG.copy()

        CONFIG.update(
            json.load(f)
        )

except Exception:

    CONFIG = DEFAULT_CONFIG.copy()


# ============================================================
# SOUND
# ============================================================

def beep_normal():

    try:

        winsound.Beep(
            1200,
            300
        )

        time.sleep(0.08)

        winsound.Beep(
            1200,
            300
        )

    except Exception:

        try:
            winsound.MessageBeep()

        except Exception:
            pass


def beep_emergency():

    try:

        for _ in range(4):

            winsound.Beep(
                1800,
                250
            )

            time.sleep(0.08)

    except Exception:

        try:
            winsound.MessageBeep()

        except Exception:
            pass


def play_violation_sound(
    emergency=False
):

    try:

        if emergency:

            threading.Thread(
                target=beep_emergency,
                daemon=True
            ).start()

        else:

            threading.Thread(
                target=beep_normal,
                daemon=True
            ).start()

    except Exception:
        pass


# ============================================================
# DRIVER MONITOR
# ============================================================

class DriverMonitor:

    def __init__(self):

        # ====================================================
        # SETTINGS
        # ====================================================

        self.camera_index = int(
            CONFIG.get(
                "driver_camera_index",
                0
            )
        )

        self.drowsiness_limit = float(
            CONFIG.get(
                "drowsiness_seconds",
                2.0
            )
        )

        self.distraction_limit = float(
            CONFIG.get(
                "distraction_seconds",
                3.0
            )
        )

        self.yawn_limit = int(
            CONFIG.get(
                "yawn_limit",
                4
            )
        )

        self.braking_violation_limit = int(
            CONFIG.get(
                "braking_violation_limit",
                3
            )
        )

        self.police_information_limit = int(
            CONFIG.get(
                "police_information_limit",
                5
            )
        )

        self.emergency_violation_limit = int(
            CONFIG.get(
                "emergency_violation_limit",
                2
            )
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.cap = cv2.VideoCapture(
            self.camera_index
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE:

            try:

                self.mp_face_mesh = (
                    mp.solutions.face_mesh
                )

                self.face_mesh = (
                    self.mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                )

                print(
                    "[MEDIAPIPE] Face detector ready."
                )

            except Exception as e:

                print(
                    "[MEDIAPIPE ERROR]",
                    e
                )

        else:

            print(
                "[MEDIAPIPE] Not available."
            )

        # ====================================================
        # YOLO PHONE DETECTOR
        # ====================================================

        self.phone_model = None

        if YOLO_AVAILABLE:

            try:

                model_path = os.path.join(
                    ROOT_DIR,
                    "yolov8n.pt"
                )

                print(
                    "[YOLO] Loading phone detector..."
                )

                if os.path.exists(
                    model_path
                ):

                    self.phone_model = YOLO(
                        model_path
                    )

                else:

                    self.phone_model = YOLO(
                        "yolov8n.pt"
                    )

                print(
                    "[YOLO] Phone detector ready."
                )

            except Exception as e:

                print(
                    "[YOLO ERROR]",
                    e
                )

        # ====================================================
        # TIMERS
        # ====================================================

        self.eye_closed_start = None

        self.distraction_start = None

        self.phone_distraction_start = None

        self.yawn_start = None

        # ====================================================
        # DRIVER STATES
        # ====================================================

        self.drowsiness = False

        self.distraction = False

        self.yawning = False

        self.phone_usage = False

        self.seatbelt = "Detected"

        # ====================================================
        # YAWN STATE
        # ====================================================

        self.yawn_count = 0

        self.last_yawn_time = 0

        self.yawn_episode_active = False

        # ====================================================
        # VIOLATIONS
        # ====================================================

        self.violation_count = 0

        self.violation_flags = {
            "drowsiness": False,
            "distraction": False,
            "yawning": False,
            "phone": False
        }

        # ====================================================
        # SAFETY RESPONSE
        # ====================================================

        self.emergency_alert = False

        # Simulated braking only.
        # This NEVER controls real brakes.
        self.simulated_braking = False

        self.slowdown_level = 0

        self.police_notice = False

        self.last_violation = "None"

        self.last_violation_time = 0

        # ====================================================
        # THREAD LOCK
        # ====================================================

        self.lock = threading.Lock()

    # ========================================================
    # DISTANCE
    # ========================================================

    @staticmethod
    def distance(
        p1,
        p2
    ):

        return (
            (p1[0] - p2[0]) ** 2
            +
            (p1[1] - p2[1]) ** 2
        ) ** 0.5

    # ========================================================
    # EYE ASPECT RATIO
    # ========================================================

    def eye_aspect_ratio(
        self,
        points
    ):

        if len(points) < 6:

            return 1.0

        vertical1 = self.distance(
            points[1],
            points[5]
        )

        vertical2 = self.distance(
            points[2],
            points[4]
        )

        horizontal = self.distance(
            points[0],
            points[3]
        )

        if horizontal == 0:

            return 1.0

        return (
            vertical1 +
            vertical2
        ) / (
            2.0 *
            horizontal
        )

    # ========================================================
    # MOUTH ASPECT RATIO
    # ========================================================

    def mouth_aspect_ratio(
        self,
        points
    ):

        if len(points) < 4:

            return 0.0

        vertical = self.distance(
            points[1],
            points[2]
        )

        horizontal = self.distance(
            points[0],
            points[3]
        )

        if horizontal == 0:

            return 0.0

        return (
            vertical /
            horizontal
        )

    # ========================================================
    # REGISTER VIOLATION
    # ========================================================

    def register_violation(
        self,
        violation_type
    ):

        with self.lock:

            # Prevent frame-by-frame counting
            if self.violation_flags.get(
                violation_type,
                False
            ):

                return

            self.violation_flags[
                violation_type
            ] = True

            self.violation_count += 1

            self.last_violation = (
                violation_type
            )

            self.last_violation_time = (
                time.time()
            )

            print(
                ""
            )

            print(
                "========================================"
            )

            print(
                "[VIOLATION]",
                violation_type
            )

            print(
                "Total violations:",
                self.violation_count
            )

            print(
                "========================================"
            )

            # =================================================
            # SOUND
            # =================================================

            if (
                self.violation_count >=
                self.emergency_violation_limit
            ):

                self.emergency_alert = True

                play_violation_sound(
                    emergency=True
                )

            else:

                play_violation_sound(
                    emergency=False
                )

            # =================================================
            # STAGE 3
            # SIMULATED SLOW BRAKING
            # =================================================

            if (
                self.violation_count >=
                self.braking_violation_limit
            ):

                self.simulated_braking = True

                # Demonstration only.
                #
                # 3 violations = 10%
                # 4 violations = 20%
                # 5 violations = 30%
                #
                # Maximum is 30%.
                #
                # This does NOT control a real vehicle.

                self.slowdown_level = min(
                    30,
                    (
                        self.violation_count -
                        self.braking_violation_limit +
                        1
                    ) * 10
                )

                print(
                    "[DEMO] SIMULATED SLOW BRAKING"
                )

                print(
                    "[DEMO] Slowdown:",
                    self.slowdown_level,
                    "%"
                )

            # =================================================
            # STAGE 4
            # POLICE INFORMATION
            # =================================================

            if (
                self.simulated_braking
                and
                self.violation_count >=
                self.police_information_limit
            ):

                self.police_notice = True

                print(
                    "[DEMO] Driver continued "
                    "ignoring warnings after "
                    "simulated braking."
                )

                print(
                    "[DEMO] POLICE INFORMATION ALERT"
                )

    # ========================================================
    # RESET VIOLATION FLAG
    # ========================================================

    def reset_violation_flag(
        self,
        violation_type
    ):

        with self.lock:

            self.violation_flags[
                violation_type
            ] = False

    # ========================================================
    # PHONE DETECTION
    # ========================================================

    def detect_phone(
        self,
        frame
    ):

        if self.phone_model is None:

            return False

        phone_found = False

        try:

            results = self.phone_model(
                frame,
                verbose=False,
                conf=0.30
            )

            for result in results:

                if result.boxes is None:

                    continue

                for i in range(
                    len(result.boxes)
                ):

                    class_id = int(
                        result.boxes.cls[i]
                        .item()
                    )

                    confidence = float(
                        result.boxes.conf[i]
                        .item()
                    )

                    # COCO class 67:
                    # cell phone

                    if (
                        class_id == 67
                        and
                        confidence >= 0.30
                    ):

                        phone_found = True

                        # Draw phone box
                        try:

                            box = (
                                result
                                .boxes
                                .xyxy[i]
                                .cpu()
                                .numpy()
                                .astype(int)
                            )

                            x1, y1, x2, y2 = box

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                2
                            )

                            cv2.putText(
                                frame,
                                "MOBILE PHONE",
                                (
                                    x1,
                                    max(
                                        25,
                                        y1 - 10
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2
                            )

                        except Exception:
                            pass

                        break

                if phone_found:

                    break

        except Exception as e:

            print(
                "[PHONE DETECTION ERROR]",
                e
            )

        return phone_found

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        if frame is None:

            return frame

        now = time.time()

        # ====================================================
        # MEDIAPIPE FACE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = None

        if self.face_mesh is not None:

            try:

                results = (
                    self.face_mesh.process(
                        rgb
                    )
                )

            except Exception:

                results = None

        # ====================================================
        # FACE FOUND
        # ====================================================

        if (
            results is not None
            and
            results.multi_face_landmarks
        ):

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            h, w = frame.shape[:2]

            points = []

            for landmark in (
                face_landmarks.landmark
            ):

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                points.append(
                    (x, y)
                )

            # =================================================
            # FACE LANDMARKS
            # =================================================

            for x, y in points[::8]:

                if (
                    0 <= x < w
                    and
                    0 <= y < h
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            # =================================================
            # EYE DETECTION
            # =================================================

            left_eye_indices = [
                33,
                160,
                158,
                133,
                153,
                144
            ]

            right_eye_indices = [
                362,
                385,
                387,
                263,
                373,
                380
            ]

            try:

                left_eye = [
                    points[i]
                    for i in
                    left_eye_indices
                ]

                right_eye = [
                    points[i]
                    for i in
                    right_eye_indices
                ]

                left_ear = (
                    self.eye_aspect_ratio(
                        left_eye
                    )
                )

                right_ear = (
                    self.eye_aspect_ratio(
                        right_eye
                    )
                )

                ear = (
                    left_ear +
                    right_ear
                ) / 2.0

            except Exception:

                ear = 1.0

            # =================================================
            # DROWSINESS - 2 SECONDS
            # =================================================

            if ear < 0.20:

                if (
                    self.eye_closed_start
                    is None
                ):

                    self.eye_closed_start = now

                closed_time = (
                    now -
                    self.eye_closed_start
                )

                if (
                    closed_time >=
                    self.drowsiness_limit
                ):

                    if not self.drowsiness:

                        self.drowsiness = True

                        self.register_violation(
                            "drowsiness"
                        )

            else:

                self.eye_closed_start = None

                if self.drowsiness:

                    self.drowsiness = False

                    self.reset_violation_flag(
                        "drowsiness"
                    )

            # =================================================
            # HEAD DIRECTION
            # =================================================

            try:

                nose = points[1]

                left_eye_center = (
                    points[33]
                )

                right_eye_center = (
                    points[263]
                )

                eye_center_x = (
                    left_eye_center[0] +
                    right_eye_center[0]
                ) / 2.0

                eye_distance = abs(
                    right_eye_center[0] -
                    left_eye_center[0]
                )

                if eye_distance > 0:

                    yaw_ratio = (
                        nose[0] -
                        eye_center_x
                    ) / eye_distance

                else:

                    yaw_ratio = 0

            except Exception:

                yaw_ratio = 0

            looking_away = (
                abs(yaw_ratio) > 0.22
            )

            # =================================================
            # LOOKING-AWAY DISTRACTION
            # 3 SECONDS
            # =================================================

            if looking_away:

                if (
                    self.distraction_start
                    is None
                ):

                    self.distraction_start = now

                distraction_time = (
                    now -
                    self.distraction_start
                )

                if (
                    distraction_time >=
                    self.distraction_limit
                ):

                    if not self.distraction:

                        self.distraction = True

                        self.register_violation(
                            "distraction"
                        )

            else:

                self.distraction_start = None

                if (
                    self.distraction
                    and
                    not self.phone_usage
                ):

                    self.distraction = False

                    self.reset_violation_flag(
                        "distraction"
                    )

            # =================================================
            # YAWNING
            # =================================================

            mouth_indices = [
                61,
                291,
                13,
                14
            ]

            try:

                mouth_points = [
                    points[i]
                    for i in
                    mouth_indices
                ]

                mar = (
                    self.mouth_aspect_ratio(
                        mouth_points
                    )
                )

            except Exception:

                mar = 0.0

            mouth_open = (
                mar > 0.48
            )

            # =================================================
            # START NEW YAWN
            # =================================================

            if mouth_open:

                if not self.yawn_episode_active:

                    self.yawn_episode_active = True

                    self.yawn_start = now

            else:

                # Mouth closed = end of
                # current yawn episode

                if self.yawn_episode_active:

                    if (
                        self.yawn_start
                        is not None
                    ):

                        duration = (
                            now -
                            self.yawn_start
                        )

                        if (
                            duration >= 0.6
                            and
                            now -
                            self.last_yawn_time
                            > 2.0
                        ):

                            self.yawn_count += 1

                            self.last_yawn_time = (
                                now
                            )

                            print(
                                "[YAWN] Count:",
                                self.yawn_count
                            )

                            if (
                                self.yawn_count >=
                                self.yawn_limit
                            ):

                                if not self.yawning:

                                    self.yawning = True

                                    self.register_violation(
                                        "yawning"
                                    )

                    self.yawn_episode_active = (
                        False
                    )

                    self.yawn_start = None

            # =================================================
            # PHONE DETECTION
            # =================================================

            phone_found = (
                self.detect_phone(
                    frame
                )
            )

            # =================================================
            # MOBILE PHONE = DISTRACTION
            # =================================================

            if phone_found:

                self.phone_usage = True

                if (
                    self.phone_distraction_start
                    is None
                ):

                    self.phone_distraction_start = (
                        now
                    )

                phone_time = (
                    now -
                    self.phone_distraction_start
                )

                # Phone must be detected
                # continuously for 3 seconds

                if (
                    phone_time >=
                    self.distraction_limit
                ):

                    if not self.distraction:

                        self.distraction = True

                        print(
                            "[PHONE] Driver using "
                            "mobile for 3 seconds."
                        )

                        self.register_violation(
                            "distraction"
                        )

            else:

                self.phone_usage = False

                self.phone_distraction_start = (
                    None
                )

                if not looking_away:

                    if self.distraction:

                        self.distraction = False

                        self.reset_violation_flag(
                            "distraction"
                        )

            # =================================================
            # STATUS
            # =================================================

            if self.drowsiness:

                status_text = "DROWSY"

            elif self.distraction:

                if self.phone_usage:

                    status_text = (
                        "PHONE DISTRACTION"
                    )

                else:

                    status_text = (
                        "DISTRACTED"
                    )

            elif self.yawning:

                status_text = "YAWNING"

            else:

                status_text = "ATTENTIVE"

            # =================================================
            # CAMERA STATUS
            # =================================================

            cv2.putText(
                frame,
                "Drowsiness: "
                +
                (
                    "ALERT"
                    if self.drowsiness
                    else "SAFE"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 0, 255)
                    if self.drowsiness
                    else (0, 255, 0)
                ),
                2
            )

            cv2.putText(
                frame,
                "Distraction: "
                +
                (
                    "ALERT"
                    if self.distraction
                    else "SAFE"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 0, 255)
                    if self.distraction
                    else (0, 255, 0)
                ),
                2
            )

            cv2.putText(
                frame,
                f"Yawns: "
                f"{self.yawn_count}/"
                f"{self.yawn_limit}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 0, 255)
                    if
                    self.yawn_count >=
                    self.yawn_limit
                    else
                    (0, 255, 0)
                ),
                2
            )

            cv2.putText(
                frame,
                f"Violations: "
                f"{self.violation_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 0, 255)
                    if self.violation_count > 0
                    else
                    (0, 255, 0)
                ),
                2
            )

            # =================================================
            # MOBILE PHONE
            # =================================================

            if self.phone_usage:

                cv2.putText(
                    frame,
                    "MOBILE PHONE DETECTED",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            # =================================================
            # BRAKING DEMO
            # =================================================

            if self.simulated_braking:

                cv2.putText(
                    frame,
                    "SIMULATED SLOW BRAKING",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Slowdown: "
                    f"{self.slowdown_level}%",
                    (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

            # =================================================
            # POLICE DEMO
            # =================================================

            if self.police_notice:

                cv2.putText(
                    frame,
                    "POLICE INFORMATION ALERT",
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        return frame

    # ========================================================
    # CAMERA FRAME
    # ========================================================

    def get_frame(self):

        if not self.cap.isOpened():

            self.cap = cv2.VideoCapture(
                self.camera_index
            )

        success, frame = (
            self.cap.read()
        )

        if not success:

            return None

        return self.process_frame(
            frame
        )

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        with self.lock:

            return {

                "drowsiness":
                    bool(
                        self.drowsiness
                    ),

                "distraction":
                    bool(
                        self.distraction
                    ),

                "yawning":
                    bool(
                        self.yawning
                    ),

                "phone_usage":
                    bool(
                        self.phone_usage
                    ),

                "seatbelt":
                    self.seatbelt,

                "yawn_count":
                    int(
                        self.yawn_count
                    ),

                "yawn_limit":
                    int(
                        self.yawn_limit
                    ),

                "violation_count":
                    int(
                        self.violation_count
                    ),

                "last_violation":
                    self.last_violation,

                "emergency_alert":
                    bool(
                        self.emergency_alert
                    ),

                "simulated_braking":
                    bool(
                        self.simulated_braking
                    ),

                "slowdown_level":
                    int(
                        self.slowdown_level
                    ),

                "police_notice":
                    bool(
                        self.police_notice
                    ),

                "drowsiness_seconds":
                    float(
                        self.drowsiness_limit
                    ),

                "distraction_seconds":
                    float(
                        self.distraction_limit
                    ),

                "braking_violation_limit":
                    int(
                        self.braking_violation_limit
                    ),

                "police_information_limit":
                    int(
                        self.police_information_limit
                    )
            }

    # ========================================================
    # COMPATIBILITY
    # ========================================================

    def get_status(self):

        return self.get_state()

    # ========================================================
    # RELEASE
    # ========================================================

    def release(self):

        try:

            if self.cap is not None:

                self.cap.release()

        except Exception:
            pass

        try:

            if self.face_mesh is not None:

                self.face_mesh.close()

        except Exception:
            pass