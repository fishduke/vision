# 최신코드
import dlib
import cv2
import glob
from PIL import Image, ImageOps
import numpy as np
import openface
import os

project = os.getcwd()

print('현재 경로 : ', os.getcwd())

count = int(input('몇 명을 등록하시나요? ex)4 :'))
adr = input('현재 위치에서 학습 데이터를 저장할 폴더 경로를 알려주세요 ex) /dataset/train/ :')
adr = project + adr
# adr = '/dataset/train/'
try:
    if not(os.path.isdir(adr)):
        os.makedirs(os.path.join(adr))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("디렉토리 생성 실패")
        raise

target_folder = input('이미지를 가져올 폴더 경로를 알려주세요 ex) /dataset/face/ :')
target_folder = project + target_folder


predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_aligner = openface.AlignDlib('./shape_predictor_68_face_landmarks.dat')

fail_get_face = []
fail_to_crop = []

for i in range(count):
    face_detector = dlib.get_frontal_face_detector()
    crop_count = 0

    set_path = input('등록할 사람의 폴더명을 지정해주세요 ex)고길동 :')
    os.chdir(adr)

    try:
        if not(os.path.isdir(set_path)):
            os.makedirs(os.path.join(set_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("디렉토리 생성 실패")
            raise
    os.chdir(project)

    data_path = adr + set_path

    #데이터 로드
    face_set = glob.glob(project+target_folder+set_path+'/*.jpg')
    # print(target_folder+set_path)
    print(len(face_set),'face_set')


    for i in range(len(face_set)):
        #이미지 데이터 1개씩 로드
        img = cv2.imread(face_set[i])
        
        #얼굴 디텍트
        faces = face_detector(img,1)
        
        #디텍트 된 얼굴 수에 따른 반응 설정
        if len(faces) >= 1:
            print("얼굴 검출 성공 =>", face_set[i])

            # 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시
            for k, d in enumerate(faces):
                # k 얼굴 인덱스
                # d 얼굴 좌표
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
                .format(k, d.left(), d.top(), d.right(), d.bottom()))

                shape = predictor(img, d)
                # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))

                #랜드마크 리스트 생성
                landmark_list = []

                #코에 빨강 찍기
                for p in shape.parts():
                    landmark_list.append([p.x, p.y])
                    if p.x == shape.parts()[30].x and p.y == shape.parts()[30].y:
                        # cv2.circle(img, (p.x, p.y), 1, (0, 0, 255), -1)
                        face_center_x = shape.parts()[30].x
                        face_center_y = shape.parts()[30].y
                        nose = shape.parts()[30]
                    else:
                        pass
                        # cv2.circle(img, (p.x, p.y), 1, (0, 255, 0), -1)

                try:
                    crop = img[faces[i].top():faces[i].bottom(),
                    faces[i].left():faces[i].right()]
                except:
                    print("crop하기 어려운 사진:", face_set[i])
                    fail_to_crop.append(face_set[i])
                    pass
                
                alignedFace = face_aligner.align(532, img, d,
                landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)    

                file_name = str(crop_count)+".jpg"
                data_path_a = data_path + '/' + file_name
                alignedFace = cv2.resize(alignedFace, (440, 400), interpolation = cv2.INTER_AREA)

                cv2.imwrite(data_path_a, alignedFace)
                crop_count += 1

        elif len(faces) == 0:
                print('얼굴 검출 실패 =>',face_set[i])
                fail_get_face.append(face_set[i])
                pass
        else:
            pass

    print(len(face_set),'장의 사진 중',crop_count,'개의 얼굴 검출, Crop 완료')

os.chdir(project)

#실패한 결과 저장
fail_path = project + '/fail_history/'
fail_crop = project + 'fail_to_crop.txt'
fail_get = project + 'fail_get_face.txt'

try:
    if not(os.path.isdir(fail_path)):
        os.makedirs(os.path.join(fail_path))
except OSError as e:
    if e.errno != errno.EEXIST:
        print('fail_history Directory exist')
        raise

with open(fail_crop, 'w') as f:
    for line in fail_to_crop:
        f.write(line)
        f.write('\n')
with open(fail_get, 'w') as f:
    for line in fail_get_face:
        f.write(line)
        f.write('\n')

if os.path.isdir(fail_get) and os.path.isdir(fail_crop) == True:
    print('실패 결과 리스트 작성 완료')
elif os.path.isdir(fail_get) == True:
    print('얼굴 검출 실패 리스트 작성 완료')
elif os.path.isdir(fail_crop) == True:
    print('학습용 얼굴 데이터 저장 실패 리스트 작성 완료')
else:
    print('실패 결과 데이터 없음')



    
#Crop 코드 종료
print('AI 학습 데이터 셋 구축 완료')