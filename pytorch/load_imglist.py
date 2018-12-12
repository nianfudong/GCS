# -*- coding:utf-8 -*
import torch.utils.data as data

import cv2
import os
import os.path
import random
import numpy as np
from PIL import Image, ImageEnhance
import math


def twicePadding(thisImg, thisLandmarks):
    imgshape = thisImg.shape
    width = imgshape[1]
    height = imgshape[0]

    add_top = int(height)
    add_bottom = int(height)
    add_left = int(width)
    add_right = int(width)
    resImg = cv2.copyMakeBorder(thisImg, add_top, add_bottom, add_left, add_right, cv2.BORDER_CONSTANT, (0, 0, 0))
    for i in range(len(thisLandmarks)):
        thisLandmarks[i][0] += add_left
        thisLandmarks[i][1] += add_top
    return [resImg, thisLandmarks]


def flipImg(imageInfo):
    thisImg = imageInfo[0]
    thisLandmarks = imageInfo[1]
    imgshape = thisImg.shape

    width = imgshape[1]
    height = imgshape[0]
    random.seed()
    ix = random.randint(0, 1)
    dstlandmarks = []
    if ix == 1:  ###水平翻转
        dstimg = cv2.flip(thisImg, 1)
        fliplandmarks = []
        for i in range(len(thisLandmarks)):
            curlandmark_x = imgshape[1] - 1 - thisLandmarks[i][0]
            curlandmark_y = thisLandmarks[i][1]
            fliplandmarks.append([curlandmark_x, curlandmark_y])
        # 重新排列顺序

        def flip_landmark(re_landmark, or_landmark, sequence):
            for i in range(sequence[1], sequence[0] - 1, -1):
                re_landmark.append(or_landmark[i])
            return re_landmark

        def copy_landmark(re_landmark, or_landmark, sequence):
            for i in range(sequence[0], sequence[1] + 1):
                re_landmark.append(or_landmark[i])
            return re_landmark

        dstLandmarks = flip_landmark(dstlandmarks, fliplandmarks, [0, 32])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [42, 46])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [47, 50])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [33, 37])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [38, 41])

        dstLandmarks = copy_landmark(dstLandmarks, fliplandmarks, [51, 54]) #顺序不变

        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [55, 59])

        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [68, 72])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [73, 75])

        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [60, 64])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [65, 67])

        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [76, 82])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [83, 87])


        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [88, 92])

        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [93, 95])
        dstLandmarks = flip_landmark(dstLandmarks, fliplandmarks, [96, 97])

    else:  # 不水平翻转
        dstimg = thisImg
        dstLandmarks = thisLandmarks
    return [dstimg, dstLandmarks]


def rotateImage(imageInfo):
    angle = [-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5, 0,2, 1,2,3,4,5,6,7,8,9,11, 10,11,12,13,14,15]
    random.seed()
    ix = random.randint(0, len(angle) - 1)

    thisImg = imageInfo[0]
    thisLandmarks = imageInfo[1]

    thisAngle = angle[ix]
    imgshape = thisImg.shape
    width = imgshape[1]
    height = imgshape[0]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), thisAngle, 1)
    # 第三个参数：变换后的图像大小
    resimg = cv2.warpAffine(thisImg, M, (width, height))
    reslandmarks = []
    for i in range(len(thisLandmarks)):
        curlandmark_x = thisLandmarks[i][0]
        curlandmark_y = thisLandmarks[i][1]
        dst_x = curlandmark_x * M[0][0] + curlandmark_y * M[0][1] + M[0][2]
        dst_y = curlandmark_x * M[1][0] + curlandmark_y * M[1][1] + M[1][2]
        reslandmarks.append([dst_x, dst_y])
    # for i in range(len(reslandmarks)):
    #     pointx = float(reslandmarks[i][0])
    #     pointy = float(reslandmarks[i][1])
    #     cv2.circle(resimg, (int(pointx), int(pointy)), 1, (255, 255, 0), 2)
    # cv2.imshow('rotateImg', resimg)
    # cv2.waitKey(0)
    return [resimg, reslandmarks]


def getROIimg(imageInfo):
    thisImg = imageInfo[0]
    thisLandmarks = imageInfo[1]
    # 框的八个方向偏移增强

    random.seed()
    ix = random.randint(0, 11)
    random.seed()
    iy = random.randint(0, 11)
    #print(ix)
    #print(iy)
    roiLandmarks = []   # xyxyxyx -> xxxxxyyyyyy
    for i in range(len(thisLandmarks)):
        roiLandmarks.append(thisLandmarks[i][0] - ix)
    for i in range(len(thisLandmarks)):
        roiLandmarks.append(thisLandmarks[i][1] - iy)

    finalimg = thisImg[iy:iy+112, ix:ix+112]
    #print(finalimg.shape)
    roiimage = Image.fromarray(finalimg)
    roilabel = np.array(roiLandmarks, dtype=np.float32)
    return roiimage, roilabel


def dataAug(imageInfo):
    # thisImg = imageInfo[0]
    # thisLandmarks = imageInfo[1]
    augrotatesample = rotateImage(imageInfo)
    augflipsample = flipImg(augrotatesample)
    return getROIimg(augflipsample)


def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如
    return dst


def default_loader(path, target):
    # print(path)
    img = cv2.imread(path)

    random.seed()
    change_image = random.randint(0, 1)
    color_image = random.randint(0, 1)
    brightness_image = random.randint(0, 1)
    contrast_image = random.randint(0, 1)
    Sharpness = random.randint(0, 1)
    if change_image:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if color_image:
            random_factor = np.random.uniform(0.8, 1.3)  # / 10.  # 随机因子
            img = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
        if brightness_image:
            random_factor = np.random.uniform(0.6, 1.5)  # 随机因子
            img = ImageEnhance.Brightness(img).enhance(random_factor)  # 调整图像的亮度
        if contrast_image:
            random_factor = np.random.uniform(0.5, 1.8)  # 随机因1子
            img = ImageEnhance.Contrast(img).enhance(random_factor)  # 调整图像对比度

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img)
    pairLandmarks = []#(x1,y1) (x2,y2) (x3,y3)
    for i in range(len(target) // 2):
        pairLandmarks.append([target[2 * i], target[2 * i + 1]])
    sample = [img, pairLandmarks]
    # print (len(pairLandmarks))

    # bigsample = twicePadding(sample)

    return dataAug(sample)


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, loader=default_loader):
        self.root      = root
        self.transform = transform
        self.loader    = loader
        with open(fileList, 'r') as file:
            self.lines = file.readlines()
        random.shuffle(self.lines)

    def __getitem__(self, index):
        curLine = self.lines[index]
        splitLineContents = curLine.split(' ')
        imgPath=splitLineContents[0]
        #print(imgPath)
        landmarks = splitLineContents[1:-1]
        for id in range(len(landmarks)):
            landmarks[id] = float(landmarks[id])

        #print (len(landmarks))
        #try:
        img, label = self.loader(os.path.join(self.root, imgPath), landmarks)
        #except:
            #print(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.lines)
