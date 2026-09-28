import cv2, json, time, os, requests, threading
from ultralytics import YOLO

BASE=os.path.dirname(os.path.abspath(__file__))
CFG=json.load(open(os.path.join(BASE,"config.json"),encoding="utf-8"))
cap=cv2.VideoCapture(int(CFG.get("road_camera_index",0)),cv2.CAP_DSHOW)
if not cap.isOpened(): cap=cv2.VideoCapture(int(CFG.get("road_camera_index",0)))
model=YOLO("yolov8n.pt")
PC1=f"http://{CFG.get('pc1_ip','127.0.0.1')}:{int(CFG.get('pc1_port',5000))}/api/road/update"

# COCO classes used for road safety.
vehicle_names={"car","truck","bus","motorcycle","bicycle"}
last_send=0
print("PC2 road camera running. Press Q to stop.")
while cap.isOpened():
    ok,frame=cap.read()
    if not ok: break
    try:
        result=model.predict(frame,verbose=False,conf=0.40,device="cpu")[0]
        objects=[]; vehicles=0; people=0
        for box in result.boxes:
            cls=int(box.cls[0]); name=model.names[cls]
            conf=float(box.conf[0])
            if name in vehicle_names: vehicles+=1
            if name=="person": people+=1
            x1,y1,x2,y2=map(int,box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{name} {conf:.2f}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
            objects.append({"name":name,"confidence":round(conf,2)})
        # Simple demo hazard rule: person or close/large central object.
        hazard=people>0 or vehicles>=5
        now=time.time()
        if now-last_send>=0.5:
            try: requests.post(PC1,json={"vehicles":vehicles,"people":people,"hazard":hazard,"objects":objects},timeout=.35)
            except Exception: pass
            last_send=now
        cv2.putText(frame,f"ROAD MONITORING | vehicles:{vehicles} people:{people}",(15,30),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        cv2.imshow("PC2 - Road Monitoring (Q to quit)",frame)
    except Exception as e:
        cv2.putText(frame,"YOLO error - check model/install",(15,30),0,.7,(0,0,255),2)
        cv2.imshow("PC2 - Road Monitoring (Q to quit)",frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break
cap.release(); cv2.destroyAllWindows()
