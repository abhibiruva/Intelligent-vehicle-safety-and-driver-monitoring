import cv2, json, os
cfg=json.load(open("config.json"))
idx=int(cfg.get("driver_camera_index",0))
cap=cv2.VideoCapture(idx, cv2.CAP_DSHOW)
print("Camera:", idx, "opened:", cap.isOpened())
while cap.isOpened():
    ok, frame=cap.read()
    if not ok: break
    cv2.imshow("Camera test - press Q",frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break
cap.release(); cv2.destroyAllWindows()
