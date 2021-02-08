from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras import optimizers, initializers, regularizers, metrics
import math
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from keras import models, layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import math
import shutil
import dlib
from keras.callbacks import EarlyStopping
from keras.applications import ResNet50


train_datagen = ImageDataGenerator(rescale=1./255,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)


batch_size = 512
dir = os.getcwd()
# train_dir = os.path.join('/home/gpulab/Dev/classification/whoisit/dataset/train')
# test_dir = os.path.join('/home/gpulab/Dev/classification/whoisit/dataset/test')
train_dir = input('학습용 이미지 폴더 경로를 지정하세요 ex) /dataset/train :')
train_dir = dir + train_dir


train_generator = train_datagen.flow_from_directory(train_dir, batch_size=batch_size, target_size=(220,200), color_mode='rgb',class_mode='categorical',subset="training")
validation_generator = train_datagen.flow_from_directory(train_dir, batch_size=batch_size, target_size=(220,200), color_mode='rgb',class_mode='categorical',subset="validation")


#다중 GPU 사용 모델 불러오기
#사용할 GPU 수 정의
a = [0,1,2,3,4,5,6,7]
b = []

gpu_num = int(input('사용할 GPU수를 정해주세요 ex)1 :'))
for i in range(gpu_num):
    b.append(a[i])

b = str(b)
b = b.replace('[','')
b = b.replace(']','')


os.environ["CUDA_VISIBLE_DEVICES"] = b

strategy = tf.distribute.MirroredStrategy()
print('장치의 수: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():  
    initial_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(220,200,3), pooling='max', classes=4)
    x = initial_model.output
    x = Dense(4, activation='softmax')(x)
    additional_model = Model(initial_model.input, x)
    additional_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

    checkpoint = [
    tf.keras.callbacks.EarlyStopping(patience=10, mode='auto',monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', 
            mode='min', save_best_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')]

history = additional_model.fit(train_generator, 
        steps_per_epoch=train_generator.samples // batch_size,                      
        epochs=100, 
        validation_data=validation_generator, 
        validation_steps=validation_generator.samples // batch_size, 
        callbacks=[checkpoint])

print("-- Evaluate --")
scores = additional_model.evaluate_generator(validation_generator, steps=5)
print("%s: %.2f%%" %(additional_model.metrics_names[1], scores[1]*100))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()
 
plt.show()

