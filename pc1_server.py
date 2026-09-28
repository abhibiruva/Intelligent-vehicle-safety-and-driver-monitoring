from flask import Flask, jsonify, Response, send_from_directory
import cv2
import os
import time
import threading

from driver_monitor import DriverMonitor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR
)


# ============================================================
# DRIVER MONITOR
# ============================================================

monitor = DriverMonitor()


# ============================================================
# LATEST ROAD DATA FROM PC2
# ============================================================

road_data = {
    "vehicles": 0,
    "people": 0,
    "motorcycles": 0,
    "trucks": 0,
    "hazard": False,
    "connected": False,
    "last_update": 0
}


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "online",
        "server": "PC1",
        "driver_monitor": "running"
    })


# ============================================================
# DRIVER STATUS
# ============================================================

@app.route("/api/status")
def status():

    try:

        # IMPORTANT:
        # driver_monitor.py now contains get_state()
        state = monitor.get_state()

        return jsonify({
            "driver": state,
            "road": road_data,
            "server": {
                "status": "online",
                "timestamp": time.time()
            }
        })

    except Exception as e:

        print(
            "[STATUS ERROR]",
            str(e)
        )

        return jsonify({
            "driver": {},
            "road": road_data,
            "server": {
                "status": "error"
            },
            "error": str(e)
        }), 500


# ============================================================
# ROAD STATUS FROM PC2
# ============================================================

@app.route(
    "/api/road/update",
    methods=["POST"]
)
def road_update():

    global road_data

    try:

        data = {}

        # Flask JSON
        from flask import request

        incoming = request.get_json(
            silent=True
        )

        if incoming:

            data = incoming

        road_data = {
            "vehicles": int(
                data.get(
                    "vehicles",
                    0
                )
            ),

            "people": int(
                data.get(
                    "people",
                    0
                )
            ),

            "motorcycles": int(
                data.get(
                    "motorcycles",
                    0
                )
            ),

            "trucks": int(
                data.get(
                    "trucks",
                    0
                )
            ),

            "hazard": bool(
                data.get(
                    "hazard",
                    False
                )
            ),

            "connected": True,

            "last_update": time.time()
        }

        return jsonify({
            "status": "received"
        })

    except Exception as e:

        print(
            "[ROAD UPDATE ERROR]",
            str(e)
        )

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 400


# ============================================================
# DRIVER CAMERA STREAM
# ============================================================

def generate_driver_frames():

    while True:

        try:

            frame = monitor.get_frame()

            if frame is None:

                time.sleep(0.1)

                continue

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]
            )

            if not success:

                continue

            frame_bytes = (
                buffer.tobytes()
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                +
                frame_bytes
                +
                b"\r\n"
            )

        except Exception as e:

            print(
                "[VIDEO ERROR]",
                str(e)
            )

            time.sleep(0.2)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_driver_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# RESET DEMO VIOLATIONS
# ============================================================

@app.route(
    "/api/reset",
    methods=["POST"]
)
def reset():

    try:

        monitor.violation_count = 0

        monitor.yawn_count = 0

        monitor.emergency_alert = False

        monitor.police_notice = False

        monitor.last_violation = "None"

        monitor.violation_flags = {
            "drowsiness": False,
            "distraction": False,
            "yawning": False,
            "phone": False
        }

        return jsonify({
            "status": "reset"
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown_monitor():

    try:

        monitor.release()

    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print(" INTELLIGENT VEHICLE SAFETY & DRIVER MONITORING")
    print("=" * 60)
    print("")
    print("PC1 SERVER STARTING...")
    print("")
    print("Driver camera : PC1")
    print("Road camera   : PC2")
    print("Dashboard     : http://127.0.0.1:5000")
    print("")
    print("Drowsiness    : 2 seconds")
    print("Distraction   : 3 seconds")
    print("Yawning       : 4 yawns")
    print("Violation     : Sound on every violation")
    print("")
    print("Press CTRL+C to stop.")
    print("=" * 60)
    print("")

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:

        print("\nStopping server...")

    finally:

        shutdown_monitor()