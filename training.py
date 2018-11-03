from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
# CPU만 사용함. 에폭시 당 10초 정도

# AVX/FMA를 쓰지 않게 만들어서, 에러 안띄우기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

# 간단한 sequential 모델 리턴한는 함수
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  return model

# 기본 모델 만들고, fitting하고, 확인하고, 마무리하고.
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
model.summary()


# HDF5파일로 저장 (keras 기본 확장자)
model.save('my_model.h5')
# 저장한 거 불러오는 방법
#new_model = keras.models.load_model('my_model.h5')
#new_model.summary()
