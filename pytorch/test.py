import os
import cv2

inFile = open('F:/wflw/WFLW_annotations/list_98pt_rect_attr_train_test/test.txt','r')

srcImgRoot = 'F:/wflw/WFLW_images/'

dstImgRoot = 'F:/wflw/WFLW_annotations/list_98pt_rect_attr_train_test/'

outFile = open('F:/wflw/WFLW_annotations/wrong.txt','w')

allLines = inFile.readlines()

index = 0
for line in allLines:
    splitsLine = line.split(' ')
    curImgPath = splitsLine[-1][:-1]
    print(curImgPath)
    imgFullPath = srcImgRoot + curImgPath
    img = cv2.imread(imgFullPath)
    height = img.shape[0]
    width = img.shape[1]

    add_top = 300
    add_bottom = 300
    add_left = 300
    add_right = 300
    img = cv2.copyMakeBorder(img, add_top, add_bottom, add_left, add_right, cv2.BORDER_CONSTANT, (0, 0, 0))


    # for i in range(0,196,2):
    #     cv2.circle(img, (round(float(splitsLine[i]) + 300), round(float(splitsLine[i+1])) + 300) ,1,(0,0,255),2)
    #
    # cv2.imshow("img", img)
    #cv2.waitKey(0)

    upperleftX = int(splitsLine[196]) +300
    upperleftY = int(splitsLine[197]) +300

    lowerRightX = int(splitsLine[198]) +300
    lowerRightY = int(splitsLine[199]) +300

    #扩边
    diffX = lowerRightX - upperleftX
    diffY = lowerRightY - upperleftY

    #cv2.rectangle(img, (int(upperleftX), int(upperleftY)), (int(lowerRightX), int(lowerRightY)), (0, 0, 255), 2)

    upperleftX = upperleftX - diffX * 0.2
    lowerRightX = lowerRightX + diffX * 0.2

    upperleftY = upperleftY - diffY * 0.05
    lowerRightY = lowerRightY + diffY * 0.15


    #将矩形框调整为正方形
    newDiffX = lowerRightX - upperleftX
    newDiffY = lowerRightY - upperleftY

    if newDiffX < newDiffY:#需左右扩边
        pad = (newDiffY-newDiffX)/2
        upperleftX = upperleftX - pad
        lowerRightX = lowerRightX + pad
    if newDiffX > newDiffY:
        pad = (newDiffX - newDiffY) / 2
        upperleftY = upperleftY - pad
        lowerRightY = lowerRightY + pad

    side = int(lowerRightX - upperleftX)
    scale = 112 / side
    newheight = (height + 600) * scale
    newwidth = (width + 600) * scale
    newImg = cv2.resize(img, (round(newwidth), round(newheight)))

    newupperleftX = upperleftX * scale - 8
    newupperleftY = upperleftY * scale - 8

    # cv2.rectangle(img, (int(upperleftX), int(upperleftY)),
    #               (int(lowerRightX), int(lowerRightY )), (255, 255, 0), 2)
    # cv2.imshow("img", img)
    #
    # cv2.rectangle(newImg, (int(upperleftX* scale - 8), int(upperleftY* scale - 8)), (int(lowerRightX* scale + 8), int(lowerRightY* scale + 8)), (255,255,0), 2)
    # cv2.imshow("img2", newImg)
    # cv2.waitKey(0)

    landmarkList = []
    for i in range(0, 196, 2):
        landmarkList.append((float(splitsLine[i]) + 300 - upperleftX) * scale + 8)
        landmarkList.append((float(splitsLine[i + 1]) + 300 - upperleftY) * scale + 8)

    roiFace = newImg[int(newupperleftY):int(newupperleftY) + 128, int(newupperleftX):int(newupperleftX) + 128]
    #useImg = cv2.resize(roiFace,(112,112))
    dstImgPath = curImgPath[:-4] + '_' + str(index) + '.jpg'

    cv2.imwrite(dstImgRoot + dstImgPath, roiFace)


    outFile.write(dstImgPath + ' ')
    for i in range(len(landmarkList)):
        outFile.write(str('%.3f'%landmarkList[i]) + ' ')

    outFile.write('\n')
    index += 1
    #cv2.rectangle(img, (int(upperleftX), int(upperleftY)), (int(lowerRightX), int(lowerRightY)), (255,255,0), 2)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    # for i in range(0,196,2):
    #    cv2.circle(useImg, (round(float(landmarkList[i])), round(float(landmarkList[i+1]))) ,1,(0,255,0),2)
    # cv2.imshow("region", useImg)
    # cv2.waitKey(0)
    # pass


inFile.close()
outFile.close()