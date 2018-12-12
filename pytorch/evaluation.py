import os
import math

#inFileTruth = open('F:/wflw/WFLW_annotations/valGroundtruth.txt','r')

#inFilePred = open('F:/wflw/modelv1/results/modelmse/msepredVal.txt','r')

inFileTruth = open('F:/wflw/WFLW_annotations/testGroundtruth.txt','r')

inFilePred = open('F:/wflw/modelv1/results/modelgcs/gcswingpredTest_1.5.txt','r')

allTruth = inFileTruth.readlines()
allPred = inFilePred.readlines()

allError = 0

failureNum = 0

for ix in range(len(allTruth)):
    curTruthLine = allTruth[ix]
    curTruthLandmark = curTruthLine.split(' ')[1:-1]

    curPredLine = allPred[ix]
    curPredLandmark = curPredLine.split(' ')[1:-1]

    leftx = float(curTruthLandmark[120])
    lefty = float(curTruthLandmark[121])

    rightx = float(curTruthLandmark[144])
    righty =  float(curTruthLandmark[145])

    norm = math.sqrt((rightx-leftx) * (rightx-leftx) + (righty - lefty) * (righty - lefty))

    error = 0
    for i in range(0,98):
        predX = float(curPredLandmark[i*2])
        predY = float(curPredLandmark[i * 2 + 1])
        truthX = float(curTruthLandmark[i * 2])
        truthY = float(curTruthLandmark[i * 2 + 1])
        dist = math.sqrt((truthX-predX) * (truthX-predX) + (truthY - predY) * (truthY - predY))
        normdist = dist / norm
        error += normdist
        if normdist > 0.1:
            failureNum += 1
    error = error/98.0
    allError += error

print("mean error is: " + str(allError / 2500.0))
print("failure rate is: " + str(failureNum / (2500.0 * 98.0)))





