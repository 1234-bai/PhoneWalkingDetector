import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
from tqdm import tqdm
 
TRAIN_RATIO = [8, 1, 1]
rootPath = 'datasets/phoneData/phoneExtension/'
jpgPath = 'images/'
labelPath = 'labels/'
annotationPath = 'Annotations/'
dataSetPath = 'D:/_NewCode/PythonPro/yolov5_phone/datasets/phone/'
 
class Divider:
    def __init__(self, jpgPath, labelPath, proportion_train_val_test = [8,1,1]):
        self.labelPath = labelPath
        self.proportion = proportion_train_val_test
        self.initialImgSetPath = jpgPath
        self.totalImgNames = self.__gainJpgFiles(self.initialImgSetPath)
        self.countImg = len(self.totalImgNames)
        self.indexList = range(self.countImg)
        self.totalProportion = sum(self.proportion)
        self.countTest = int(self.countImg * self.proportion[2] / self.totalProportion)
        self.countVal = int(self.countImg * self.proportion[1] / self.totalProportion)
        self.countTrain = self.countImg - self.countTest - self.countVal
        self.trainValIndexList = random.sample(self.indexList, self.countImg - self.countTest)
        self.valIndexList = random.sample(self.trainValIndexList, self.countVal)

    def __gainJpgFiles(self, path):
        fileList = []
        dir_list = os.listdir(path)
        for i in dir_list:
            if i.endswith('.jpg'):
                fileList.append(i)
        return fileList

    def __checkDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def divideDataSet(self, dividedDataSetPath):
        self.__checkDir(dividedDataSetPath)
        imagePath = dividedDataSetPath + 'images/'
        # self.__checkDir(imagePath)
        labelPath = dividedDataSetPath + 'labels/'
        # self.__checkDir(labelPath)
        trainImgPath = imagePath + 'train/'
        self.__checkDir(trainImgPath)
        trainLabelPath = labelPath + 'train/'
        self.__checkDir(trainLabelPath)
        valImgPath = imagePath + 'val/'
        self.__checkDir(valImgPath)
        valLabelPath = labelPath + 'val/'
        self.__checkDir(valLabelPath)
        testImgPath = imagePath + 'test/'
        self.__checkDir(testImgPath)
        testLabelPath = labelPath + 'test/'
        self.__checkDir(testLabelPath)
        for i in self.indexList:
            imgName = self.totalImgNames[i]
            imgLabelName = imgName[:-4] + '.txt'
            # print(str(i+1)+':'+imgName+','+imgLabelName)
            if i in self.trainValIndexList:
                if i in self.valIndexList:
                    copyfile(self.initialImgSetPath + imgName, valImgPath + imgName)
                    copyfile(self.labelPath + imgLabelName, valLabelPath + imgLabelName)
                else:
                    copyfile(self.initialImgSetPath + imgName, trainImgPath + imgName)
                    copyfile(self.labelPath + imgLabelName, trainLabelPath + imgLabelName)
            else:
                copyfile(self.initialImgSetPath + imgName, testImgPath + imgName)
                copyfile(self.labelPath + imgLabelName, testLabelPath + imgLabelName)
        infoText = open(dividedDataSetPath+'info.txt', 'w')
        infoText.write(str(self.countImg) + ' ' + str(self.countTest) + ' ' + str(self.countVal) + ' ' + str(self.countTrain))

    def divideTxt(self, dataSetTextPath):
        if not os.path.exists(dataSetTextPath):
            os.makedirs(dataSetTextPath)
        valText = open(dataSetTextPath+'val.txt', 'w')
        trainText = open(dataSetTextPath+'train.txt', 'w')
        testText = open(dataSetTextPath+'test.txt', 'w')
        for i in tqdm(self.indexList):
            imgName = './images/' + self.totalImgNames[i] + '\n'
            if i in self.trainValIndexList:
                if i in self.valIndexList:
                    # 写到val.txt里
                    valText.write(imgName)
                else:
                    # 写到train.txt里
                    trainText.write(imgName)
            else:
                # 写到test.txt里
                testText.write(imgName)
        valText.close()
        trainText.close()
        testText.close()
        infoText = open(dataSetTextPath+'info.txt', 'w')
        infoText.write(str(self.countImg) + ' ' + str(self.countTest) + ' ' + str(self.countVal) + ' ' + str(self.countTrain))
        infoText.close()

if __name__ == "__main__":
    divider = Divider(rootPath + jpgPath, rootPath + labelPath, TRAIN_RATIO)
    divider.divideTxt(rootPath)
    # divider.divideDataSet(dataSetPath)