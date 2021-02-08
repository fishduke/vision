import sys
import os
import dlib  #dlib 패키지 
import glob  #이미지 파일 얻기위한 패키지 

'''
매개변수가 3개(파이썬 파일까지)가 아니면 
제대로 입력하라고 메시지 출력후 종료 
제대로 입력이라면 
$ python face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces
뭐 이런식으로...
'''
if len(sys.argv) <= 0:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# 첫번째(파이썬 코드 뺀) 매개변수로 68 얼굴 랜드마크 학습된 모델 데이터 
# predictor_path = sys.argv[1]
predictor_path = './shape_predictor_68_face_landmarks.dat'
# 두번재 매개변수로는 랜드마크를 적용할 이미지들 모아둔 폴더
# faces_folder_path = sys.argv[2]
faces_folder_path = './Dataset/face'

# 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
detector = dlib.get_frontal_face_detector()
# 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성 
predictor = dlib.shape_predictor(predictor_path)

# 화면 표시용 윈도 실행 
win = dlib.image_window()

# 두번째 매개변수로 지정한 폴더를 싹다 뒤져서 jpg파일을 찾는다. 
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    # 파일에서 이미지 불러오기 
    img = dlib.load_rgb_image(f)

    # 윈도 화면 지우고 
    win.clear_overlay()
    
    # 현재 이미지 그리고 
    win.set_image(img)

    # 얼굴 인식 두번째 변수 1은 업샘플링을 한번 하겠다는 얘기인데
    # 업샘플링을하면 더 많이 인식할 수 있다고 한다.
    # 다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
    # 그냥 1이면 될 듯.
    dets = detector(img, 1)

    # 인식된 얼굴 개수 출력 
    print("Number of faces detected: {}".format(len(dets)))

    # 이제부터 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시할 것이다. 
    for k, d in enumerate(dets):
        # k 얼굴 인덱스
        # d 얼굴 좌표
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        # 인식된 좌표에서 랜드마크 추출 
        shape = predictor(img, d)
    
        # shape는 "full_object_detection" 클래스이다. 
        # part(0)부터 part(67)까지 총 68개의 X,Y 좌표를 가지고 있다. 
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # 사진에 추출한 랜드마크 좌표를 그려준다. 
        # 참고로 add_overlay에 full_object_dection 좌표를 넣어주면 기본 파랑색(0,0,255)으로 그린다. 
        win.add_overlay(shape)
    
    # 마지막으로 인식한 얼굴 좌표를 그려준다.(list of rectagles)
    # opencv 할때 주로하던 얼굴에 상자 그리기와 동일 
    # 참고로 add_overlay에 list of rectagles를 넣어주면 기본 빨간색(255,0,0)으로 그린다. 
    win.add_overlay(dets)

    # 키가 누릴때까지 대기
    dlib.hit_enter_to_continue()