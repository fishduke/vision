#라이브러리 임포트
import cv2
import dlib
import numpy as np

#분류 모델 임포트
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

descs = np.load('models/descs.npy')[()]

#카메라 입력
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print('카메라 에러')
        break
    
    #얼굴 인식 처리 부분
    faces = detector(frame,1)
    faces_count = len(faces)
    
    #유사도 비교 / 분류
    for k, d in enumerate(faces):
        shape = sp(frame, d)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        
        last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

        for name, saved_desc in descs.items():
            dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

        if dist < last_found['dist']:
            last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}
        
        #영상 정보 입력
        cv2.rectangle(frame, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
        cv2.putText(frame, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
    
    #영상 출력
    cv2.imshow('cap',frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

