import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_multiple_img(images, rows = 5, cols=5):
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(10,10))
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

test_dir = input('테스트할 이미지가 저장된 폴더 경로를 입력하세요. ex) /dataset/test_folder :')
dir = os.getcwd()
test_dir = dir + test_dir
batch_size = 256

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=batch_size, target_size=(220,200), color_mode='rgb',class_mode='categorical', shuffle=False, seed = 0)

additional_model = load_model('./pretrained_VGG_weight.hdf5')
yhat_test = additional_model.predict(test_generator, batch_size=batch_size)

test_result = []
show_result = []
print('발견된 테스트 사진 파일 수 : ', len(yhat_test))

count = len(yhat_test)

for i in range(count):

    img, b = test_generator.next()
    test_result = []
    test_result.append(yhat_test[i][0]*100)
    test_result.append(yhat_test[i][1]*100)
    test_result.append(yhat_test[i][2]*100)
    test_result.append(yhat_test[i][3]*100)

    a = test_result.index(max(test_result))

    # 분류 결과 출력
    if a == 0:
        result = 'A: %.1f%%' % (yhat_test[i][0]*100)
    elif a == 1:
        result = 'B: %.1f%%' % (yhat_test[i][1]*100)
    elif a == 2:
        result = 'C: %.1f%%' % (yhat_test[i][2]*100)
    elif a == 3:
        result = 'D: %.1f%%' % (yhat_test[i][3]*100)
    show_result.append(result)


total_images = len(img)
images = {show_result[i]: img[i] for i in range(total_images)}

display_multiple_img(images, 4, 3)