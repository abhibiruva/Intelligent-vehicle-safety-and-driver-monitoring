# 🚛 Intelligent Vehicle Safety & Driver Monitoring

### AI-Powered Real-Time Safety System for Heavy Vehicles

An AI-based prototype designed to improve heavy-vehicle safety by continuously monitoring both the **driver** and the **road environment**.

The system uses two cameras:

- 🎥 Driver camera — monitors driver behaviour
- 🎥 Front/road camera — monitors the road environment

AI models process the camera streams and provide real-time safety information through a web dashboard.

---

## 📌 Problem Statement

Heavy-vehicle drivers may experience fatigue, distraction and unsafe behaviour during long-distance driving.

At the same time, the driver needs awareness of vehicles, pedestrians and other objects present on the road.

Traditional monitoring systems may focus on only one side of the problem.

This project combines:

> **Driver Monitoring + Road Monitoring + Real-Time Safety Alerts**

into a single prototype.

---

# 🎯 Objectives

The main objectives of the project are:

1. Detect driver drowsiness.
2. Detect driver distraction.
3. Detect mobile-phone usage.
4. Detect repeated yawning.
5. Detect unsafe both-hands-up behaviour.
6. Monitor road objects using AI.
7. Generate real-time safety alerts and escalation responses.

---

# 💡 Proposed Solution

The system uses two cameras connected through two computers.

### PC1 – Driver Monitoring

The driver camera monitors:

- Eye closure
- Head direction
- Mouth movement
- Phone usage
- Both-hands-up behaviour

### PC2 – Road Monitoring

The front camera monitors:

- Cars
- Trucks
- Buses
- Motorcycles
- People
- Road hazards

PC2 sends road-monitoring information to PC1 through the local network.

PC1 combines the driver and road information and displays the result on the web dashboard.

---

# 🏗️ System Architecture

```text
                 INTELLIGENT VEHICLE SAFETY SYSTEM
                              │
             ┌────────────────┴────────────────┐
             │                                 │
      DRIVER MONITORING                  ROAD MONITORING
             │                                 │
       Driver Camera                     Road Camera
             │                                 │
       MediaPipe + YOLO                       YOLO
             │                                 │
    ┌────────┼──────────┐              ┌───────┼────────┐
    │        │          │              │       │        │
  Eyes     Head       Mouth          Vehicles People Motorcycles
    │        │          │              │       │        │
    └────────┴──────────┘              └───────┴────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            │
                     SAFETY ENGINE
                            │
                     RISK EVALUATION
                            │
                  ┌─────────┴─────────┐
                  │                   │
                ALERT              RESPONSE
                  │                   │
                  ├── Buzzer          ├── Emergency
                  ├── Dashboard       ├── Simulated Slow-down
                  └── Status          └── Police Information
                            │
                            ▼
                     LIVE DASHBOARD