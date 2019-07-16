import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#--------------------------------------------------------------------------------

def main ():

    #runTest()
    runTrain()

#--------------------------------------------------------------------------------

def runTrain():

    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RESNET50 = 'RES-NET-50'
    SE_RESNET50 = 'SE-RES-NET-50'
    SE_DenseNet121 = 'SE-DENSE-NET-121'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    #---- Path to the directory with images
    pathDirData = './database'

    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train.txt'
    pathFileVal = './dataset/val.txt'
    pathFileTest = './dataset/test.txt'

    #---- Neural network parameters: type of the network, is it pre-trained
    #---- on imagenet, number of classes
    nnArchitecture = SE_DenseNet121
    nnIsTrained =   False
    nnClassCount = 14

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100

    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    #checkpoint = './models/m-21042019-055416.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)

    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

def runTest():

    pathDirData = './database'
    pathFileTest = './dataset/test.txt'
    nnArchitecture = 'RES-NET-50'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = './models/m-200419-resnet50.pth.tar'

    timestampLaunch = ''

    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
