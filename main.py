from __future__ import absolute_import, division, print_function
import os
import sys
import tensorflow as tf
import training
import getImageMyOPENCV as imgs
from tensorflow import keras

#sin cos과 같은 연속 문자에 대한 보완 필요
def digit(a):
  b = []
  for e in a:
    if e >='0' and e <= '9':
      b.append(int(e))
    elif e != ' ' and e != ',':
      b.append(e)
  return b

#프로그램 메인 루틴
def main():
  try:
    ### 학습모델 불러오기
    model = tf.keras.models.load_model('my_model.h5')

  except (FileNotFoundError, OSError) as e:
    model = training.train()
  result = []
  ocrs = imgs.ReadImg(sys.argv[1])
  trainImg = []
  for e in ocrs:
    trainImg.append(e.data)
    result.append(model.predict(e.data).argmax())

  print('predict: ', result)
  print('Is predict right? (1.right, 2.wrong)')
  right = input()
  if int(right) != 1:
    print('Input Answer: ')
    ans = input()
    darr = digit(ans)
    for idx, e in enumerate(trainImg):
      model.fit(e, [darr[idx]])
    model.save('my_model.h5')


if __name__ == "__main__":
  main() 
