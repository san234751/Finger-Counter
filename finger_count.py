import cv2
import hand_detection_module
video=cv2.VideoCapture(0)
coun=0
detector=hand_detection_module.hand_detect(min_detection_confidence=0.8)
while True:
    suc,frame=video.read()
    image=detector.findHands(frame)
    lmlist=detector.findPosition(frame,draw=False)
    if len(lmlist)!=0:
        finger=detector.countfinger()
        coun=finger.count(1)
    cv2.putText(frame,f'Finger= {coun}',(20,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow("video",frame)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()