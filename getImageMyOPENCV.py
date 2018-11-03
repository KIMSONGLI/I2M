import cv2
import numpy as np
import tensorflow as tf
import time

## 시간측정 시작
stime = time.time()

### 학습모델 불러오기
model = tf.keras.models.load_model('my_model.h5')
# model.summary()

### 유효영역 자르기
img = cv2.bitwise_not( cv2.imread('sample.bmp', 0) )
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
img = np.uint8( cv2.connectedComponents(img)[1] )
idx = np.where(img)
img = img[np.min(idx[0]):np.max(idx[0])+1, np.min(idx[1]):np.max(idx[1])+1]

### 글자 클래스
class ocr:
    def __init__(self, Newimg, labelValue):
        self.points = np.argwhere(Newimg==labelValue)
        self.makeNewimage()
        self.ID = model.predict(self.resize28(self.image).reshape(1,-1))
    def makeNewimage(self):
    ## points의 좌표로 새로운 이미지 만들어서 저장
        cRange = np.array([[e1, e2] for (e1, e2) in self.points])
        dr = np.amax(cRange, 0)
        ul = np.amin(cRange, 0)
        self.image = np.zeros((dr[0]-ul[0]+1,dr[1]-ul[1]+1))
        self.image[cRange[:,0]-ul[0], cRange[:,1]-ul[1]] = 1
        self.position = ( np.around(np.mean(cRange[:,0])), np.around(np.mean(cRange[:,1])) )
    @staticmethod
    def resize28(image):
    ## 28*28사이즈로 줄이기 : 혹시나 그냥 따로 넣을 이미지도 있을까봐 static으로..
        rowSize, colSize = image.shape
        resized = np.zeros((28,28))
        if rowSize>colSize:
            shortLength = int(colSize*20/rowSize)
            resized[4:24, int(13-shortLength/2):int(13-shortLength/2)+shortLength] = cv2.resize( image, (shortLength,20) )
        else:
            shortLength = int(rowSize*20/colSize)
            resized[int(13-shortLength/2):int(13-shortLength/2)+shortLength, 4:24] = cv2.resize( image, (20,shortLength) )
        return resized
    @classmethod
    def labeled(cls, img):
    ## 글자 목록 뽑아내기
        ocrList = []
        for labelValue in range(1, 1+np.max(img)):
            eachString = cls(img, labelValue)
            ocrList.append(eachString)
        return ocrList

### 글자 분할, 왼쪽부터 정렬
ocrList = ocr.labeled(img)
ocrList.sort(key=lambda ORC: ORC.position[1])

## 시간 출력
print('소요 시간 :', time.time()-stime, '초')

### 잘라낸 이미지별로 저장하기
iter = 0
for classed_string in ocrList:
    iter += 1
    name = 'CLASSED{}_{}.bmp'.format(iter, np.argmax(classed_string.ID) )
    print(name, ['{:.2f}'.format(item) for item in classed_string.ID.tolist()[0]])
    cv2.imwrite(name, 255*classed_string.image/np.max(classed_string.image) )

### 화면에 적당히 색깔 넣어서 보이기
hue_img = np.uint8(143*img/len(ocrList))
white_img = 255*np.ones_like(hue_img)
labeled_img = cv2.merge([hue_img, white_img, white_img])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[hue_img==0] = 255
cv2.imwrite('CLASSED.bmp', labeled_img)
cv2.imshow('CLASSED.bmp', labeled_img)
cv2.waitKey()



##################################################################################################
# i 입력하는 법 : 
# sin 입력하는 법 : 글자 전부 인식시키고 "s"를 찾음. 리스트에서 바로 다음번에 i가 나오는지 체크함.

#bounding box를 합치는 방법
#(a) 모폴로지
#bounding box를 검정 배경 이미지에 흰색 채워진 박스로 그린 다음, dilate를 통해 영역을 넓혀 가까운 박스들끼리 이어붙이기
#(b) 바운딩박스 경계선간 거리
#bounding box 경계선 간 거리가 가까운 박스들끼리 합치기
#(c) 바운딩박스들의 Union 계산
#bounding box에 or연산을 해서 너비, 높이가 일정값 수준일때만 합쳐주기