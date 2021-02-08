# 최신코드
import dlib
import cv2
import glob
from PIL import Image, ImageOps
import numpy as np
import os

print('현재 경로 : ', os.getcwd())
project = os.getcwd()

adr = '/dataset/train/'
adr = project + adr
target_folder = '/dataset/target/'
target_folder = project + target_folder

def make_folder(path):
    try:
        if not(os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("디렉토리 생성 실패")
            raise

count = len(os.listdir(target_folder))

for i in range(count):
    make_folder(adr + os.listdir(target_folder)[i])

fail_get_face = []
fail_to_crop = []

face_detector = dlib.get_frontal_face_detector()

predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

for i in range(count):
    #크롭 카운팅 초기화
    crop_count = 0
    #데이터 로드
    target_folder = target_folder + os.listdir(target_folder)[i] + '/*.jpg'
    face_set = glob.glob(target_folder)

    print('얼굴 사진 수집 경로: ',target_folder)

    for j in range(len(face_set)):
        #이미지 데이터 1개씩 로드
        img = cv2.imread(face_set[j])
        #얼굴 디텍트
        faces = face_detector(img)
        
        #디텍트 된 얼굴 수에 따른 반응 설정
        if len(faces) >= 1:
            print("얼굴 검출 성공 =>", face_set[j])
            crop_count += 1

            for k, d in enumerate(faces):
                try:
                    crop = img[d.left():d.right(), d.top():d.bottom()]
                    crop = cv2.resize(crop, (440, 400), interpolation = cv2.INTER_AREA)

                    save_image = adr + os.listdir(adr)[i] + '/' + str(crop_count) + '.jpg'
                    cv2.imwrite(save_image, crop)
                
                except:
                    print("crop하기 어려운 사진:", face_set[j])
                    fail_to_crop.append(face_set[j])
                    fail_to_crop.append('\n')
                    pass                           

        elif len(faces) == 0:
                print('얼굴 검출 실패 =>',face_set[i])
                fail_get_face.append(face_set[i])
                fail_get_face.append('\n')
                pass
        else:
            pass

    print(len(face_set),'장의 사진 중',crop_count,'개의 얼굴 검출, Crop 완료')
    break

#실패한 결과 저장
fail_history = project + '/fail_history/'
fail2 = fail_history + 'fail_to_crop.txt'
fail3 = fail_history + 'fail_get_face.txt'

make_folder(fail_history)

with open(fail2, 'w') as f:
    for line in fail_to_crop:
        f.write(line)
with open(fail3, 'w') as f:
    for line in fail_get_face:
        f.write(line)

if os.path.isdir(fail3) and os.path.isdir(fail2) == True:
    print('실패 결과 리스트 작성 완료')
elif os.path.isdir(fail3) == True:
    print('얼굴 검출 실패 리스트 작성 완료')
elif os.path.isdir(fail2) == True:
    print('학습용 얼굴 데이터 저장 실패 리스트 작성 완료')
else:
    print('실패 결과 데이터 없음')


#Crop 코드 종료
print('AI 학습 데이터 셋 구축 완료')