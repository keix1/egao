import cv2
import numpy as np
import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
 
smile_record_count = -1
SMILE_COUNT_SETTING = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = 10
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = None
is_recording = False

while 1:
    ret, src = cap.read()
    dst = np.copy(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor = 1.11, minNeighbors = 3, minSize = (200, 200))
    
    # 認識できた顔の数だけ枠を描画する
    for x, y, w, h in face_rects:
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 顔の部分だけを切り取る
        rect = src[y : y + h, x : x + w]
        # 笑顔の特徴となりうるパーツを認識する
        smile_rects = smile_cascade.detectMultiScale(rect, scaleFactor = 1.11, minNeighbors = 50, minSize = (80, 40))
        
        # 認識できたパーツの数だけ枠を描画する
        for sx, sy, sw, sh in smile_rects:
            cv2.rectangle(dst, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 2)
        
        if len(smile_rects) >= 3: # パーツが３つ以上なら
            smile_record_count = SMILE_COUNT_SETTING
            is_recording = True
            print("smile!")
        
        if is_recording:
            now = datetime.datetime.now()
            if smile_record_count == SMILE_COUNT_SETTING:
                cv2.imwrite(f"smile_image/smile_{str(now)}.jpg", src)
            if not out:
                out = cv2.VideoWriter(f"smile_video/smile_{str(now)}.mp4", fmt, frame_rate, (width, height))
                out.write(src)
            else:
                out.write(src)
    
            smile_record_count -= 1

        if smile_record_count < 0 and is_recording:
            print("recorded!")
            out.release()
            out = None
            is_recording = False
            smile_record_count = -1

    # キャプチャ画面を描画
    cv2.imshow("dst", dst)
    
    # 「q」が押されたら終了
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 
# 終了処理
cap.release()
cv2.destroyAllWindows()