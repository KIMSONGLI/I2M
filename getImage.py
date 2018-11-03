from PIL import Image
import numpy as np
import time

stime = time.time()

# 유효영역 자르기
img = Image.open('sample.bmp').convert("1")
arr = np.invert( np.asarray(img) )
idx = np.argwhere(arr)
dr = np.amax(idx, 0)
ul = np.amin(idx, 0)
Newimg = arr[ul[0]:dr[0]+1, ul[1]:dr[1]+1]

# 함수 : 연속 영역 색인
def findContour(classSet, Newimg, row,col, idxR,idxC):
    if (idxR, idxC) not in classSet:
        classSet.add( (idxR, idxC) )
        if idxC+1 < col:
            if Newimg[idxR, idxC+1]: classSet = findContour(classSet, Newimg, row,col, idxR,idxC+1)
            if idxR+1<row and Newimg[idxR+1, idxC+1]: classSet = findContour(classSet, Newimg, row,col, idxR+1,idxC+1)
            if idxR-1>=0 and Newimg[idxR-1, idxC+1]: classSet = findContour(classSet, Newimg, row,col, idxR-1,idxC+1)
        if idxR+1<row and Newimg[idxR+1, idxC]:
            classSet = findContour(classSet, Newimg, row,col, idxR+1,idxC)
        if idxR-1>=0 and Newimg[idxR-1, idxC]:
            classSet = findContour(classSet, Newimg, row,col, idxR-1,idxC)
        if idxC-1 >= 0:
            if Newimg[idxR, idxC-1]: classSet = findContour(classSet, Newimg, row,col, idxR,idxC-1)
            if idxR+1<row and Newimg[idxR+1, idxC-1]: classSet = findContour(classSet, Newimg, row,col, idxR+1,idxC-1)
            if idxR-1>=0 and Newimg[idxR-1, idxC-1]: classSet = findContour(classSet, Newimg, row,col, idxR-1,idxC-1)
    return classSet

# 비연속 영역 분류
row, col = Newimg.shape
classList = []
classPosition = []
#classPointList = []
idxs = np.argwhere(Newimg)
idxs = list(map(tuple, idxs[idxs[:,1].argsort()]))
while True :
    classSet = set()
    classSet = findContour(classSet, Newimg, row,col, idxs[0][0],idxs[0][1])

    cRange = np.array([[e1, e2] for (e1, e2) in classSet])
    dr = np.amax(cRange, 0)
    ul = np.amin(cRange, 0)
    eachString = np.zeros((dr[0]-ul[0]+1,dr[1]-ul[1]+1))
    eachString[cRange[:,0]-ul[0], cRange[:,1]-ul[1]] = 1
    classList.append( eachString )
    classPosition.append( (np.around(np.mean(cRange[:,0])), np.around(np.mean(cRange[:,1]))) )
    #classPointList.append( classSet.copy() )

    idxs = [e for e in idxs if e not in classSet]
    if not idxs:
        break

print('소요 시간 :', time.time()-stime, '초')


# 잘라낸 이미지별로 저장하기
iter = 0
for cls in classList:
    iter += 1
    Image.fromarray(cls*255).convert('RGB').save('CLASSED{}.bmp'.format(iter))
