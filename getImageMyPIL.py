from PIL import Image
import PIL.ImageOps
import numpy as np
import tensorflow as tf
import time

## 시간측정 시작
stime = time.time()

### 학습모델 불러오기
model = tf.keras.models.load_model('my_model.h5')
# model.summary()

### 유효영역 자르기
img = PIL.ImageOps.invert(Image.open('sample.bmp')).convert("1")
Newimg = np.asarray(img.crop(img.getbbox()))

### 글자 클래스
class ocr:
    def __init__(self, Newimg, rowSize,colSize, idxR,idxC):
        self.points = set()
        self.findContour(Newimg, rowSize,colSize, idxR,idxC)
        self.makeNewimage()
        self.ID = model.predict(self.resize28(self.image).reshape(1,-1))
    def findContour(self, Newimg, rowSize,colSize, idxR,idxC):
    ## 입력받은 이미지에서 연속된 점들의 처음 그룹을 points(set)로 저장
        if (idxR, idxC) not in self.points:
            self.points.add( (idxR, idxC) )
            if idxC+1 < colSize:
                if Newimg[idxR, idxC+1]:
                    self.findContour(Newimg, rowSize,colSize, idxR,idxC+1)
                if idxR+1<rowSize and Newimg[idxR+1, idxC+1]:
                    self.findContour(Newimg, rowSize,colSize, idxR+1,idxC+1)
                if idxR-1>=0 and Newimg[idxR-1, idxC+1]:
                    self.findContour(Newimg, rowSize,colSize, idxR-1,idxC+1)
            if idxR+1<rowSize and Newimg[idxR+1, idxC]:
                self.findContour(Newimg, rowSize,colSize, idxR+1,idxC)
            if idxR-1>=0 and Newimg[idxR-1, idxC]:
                self.findContour(Newimg, rowSize,colSize, idxR-1,idxC)
            if idxC-1 >= 0:
                if Newimg[idxR, idxC-1]:
                    self.findContour(Newimg, rowSize,colSize, idxR,idxC-1)
                if idxR+1<rowSize and Newimg[idxR+1, idxC-1]:
                    self.findContour(Newimg, rowSize,colSize, idxR+1,idxC-1)
                if idxR-1>=0 and Newimg[idxR-1, idxC-1]:
                    self.findContour(Newimg, rowSize,colSize, idxR-1,idxC-1)
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
    ## 28*28사이즈로 줄이기
        rowSize, colSize = image.shape
        temp_im = Image.fromarray(image)
        resized = np.zeros((28,28))
        if rowSize>colSize:
            shortLength = int(colSize*20/rowSize)
            temp_im = temp_im.resize( (shortLength,20) )
            resized[4:24, int(13-shortLength/2):int(13-shortLength/2)+shortLength] = np.asarray(temp_im)
        else:
            shortLength = int(rowSize*20/colSize)
            temp_im = temp_im.resize( (20,shortLength) )
            resized[int(13-shortLength/2):int(13-shortLength/2)+shortLength, 4:24] = np.asarray(temp_im)
        return resized
    @classmethod
    def many(cls, Newimg):
    ## 글자 목록 뽑아내기
        ocrList = []
        rowSize, colSize = Newimg.shape
        idxs = np.argwhere(Newimg)
        idxs = list(map(tuple, idxs[idxs[:,1].argsort()]))
        while True:
            eachString = cls(Newimg, rowSize,colSize, idxs[0][0],idxs[0][1])
            ocrList.append(eachString)
            idxs = [e for e in idxs if e not in eachString.points]
            if not idxs:
                break
        return ocrList

### 불연속 기준으로 글자 분류
ocrList = ocr.many(Newimg)

## 시간 출력
print('소요 시간 :', time.time()-stime, '초')

### 잘라낸 이미지별로 저장하기
iter = 0
for classed_string in ocrList:
    iter += 1
    name = 'CLASSED{}_{}.bmp'.format(iter, np.argmax(classed_string.ID) )
    print(name, ['{:.2f}'.format(item) for item in classed_string.ID.tolist()[0]])
    Image.fromarray(255*classed_string.image).convert('RGB').save(name)
