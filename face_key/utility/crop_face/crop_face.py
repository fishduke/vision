import dlib
import cv2
import glob
import os

#함수
def make_folder(path):
    try:
        if not(os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("디렉토리 생성 실패")
            raise

def delete_file(file):
    if os.path.isfile(file):
        os.remove(file)

def write_fail(path, lines):
    if len(lines) >= 1:
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)
    else:
        pass

#경로 설정
project = os.getcwd()

face_detector = dlib.get_frontal_face_detector()
fail_get_face = []
fail_to_crop = []

fail_history = project + '/fail_history/'
dataset = project + '/dataset/'
fail2 = fail_history + 'fail_to_crop.txt'
fail3 = fail_history + 'fail_get_face.txt'
make_folder(fail_history)
make_folder(dataset)

file = './fail_history/fail_get_face.txt'
delete_file(file)
file = './fail_history/fail_to_crop.txt'
delete_file(file)


adr = '/dataset/train/'
adr = project + adr
target_folder = '/dataset/target/'
target_folder = project + target_folder

#타겟 폴더들을 학습 폴더로 복사
count = len(os.listdir(target_folder))

for i in range(count):
    make_folder(adr + os.listdir(target_folder)[i])

for i in range(count):
    #크롭 카운팅 초기화
    crop_count = 0
    #데이터 로드
    target_folder = target_folder + os.listdir(target_folder)[i] + '/*.jpg'
    face_set = glob.glob(target_folder)

    print('얼굴 사진 수집 경로: ', target_folder)

    for j in range(len(face_set)):
        print(j,'j')
    # 5장 사진만 테스트용
    # for j in range(5):
        #이미지 데이터 1개씩 로드
        img = cv2.imread(face_set[j])

        #얼굴 디텍트
        faces = face_detector(img)
        print(faces,'얼굴 ROI')
        
        #디텍트 된 얼굴 수에 따른 반응 설정
        if len(faces) >= 1:
            print("얼굴 검출 성공 =>", face_set[j])
            print("얼굴 검출 수 =>", len(faces))
            crop_count += 1

            for k, d in enumerate(faces):
                try:
                    crop = img[d.top():d.bottom(),d.left():d.right()]
                    crop = cv2.resize(crop, (440, 400), interpolation = cv2.INTER_AREA)
                    save_image = adr + os.listdir(adr)[i] + '/' + str(crop_count) + '.jpg'
                    print('이미지 저장', save_image)
                    # img = cv2.rectangle(img, (d.left(), d.top()),(d.right(), d.bottom()), (255,255,0),2)
                    cv2.imwrite(save_image, crop)
                
                except:
                    print("crop하기 어려운 사진:", face_set[j])
                    fail_to_crop.append(face_set[j])
                    fail_to_crop.append('\n')
                    pass                           

        elif len(faces) == 0:
                print('검출 실패 =>',face_set[i])
                fail_get_face.append(face_set[i])
                fail_get_face.append('\n')
                pass
        else:
            pass

    print(len(face_set),'장의 사진 중',crop_count,'개의 얼굴 검출, Crop 완료')
    break

#실패한 결과 저장
write_fail(fail2, fail_to_crop)
write_fail(fail3, fail_get_face)

if os.path.isfile(fail3) and os.path.isfile(fail2) == True:
    print('실패 결과 리스트 작성 완료')
elif os.path.isfile(fail3) == True:
    print('얼굴 검출 실패 리스트 작성 완료')
elif os.path.isfile(fail2) == True:
    print('학습용 얼굴 데이터 저장 실패 리스트 작성 완료')
else:
    print('실패 결과 데이터 없음')


#Crop 코드 종료
print('AI 학습 데이터 셋 구축 완료')