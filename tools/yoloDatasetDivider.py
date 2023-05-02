import os
import random
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
 
TRAIN_RATIO = [6, 2, 2]
rootPath = 'D:/QianXiaoYi/Pictures/Data/phone/train_with_anoations/phone/'
jpgPath = 'images/'
labelPath = 'labels/'
annotationPath = 'Annotations/'
dataSetPath = 'D:/_NewCode/PythonPro/yolov5_phone/datasets/phone/'
 


def gainFiles(path, suffix = 'jpg'):
    fileList = []
    dir_list = os.listdir(path)
    for i in dir_list:
        if i.endswith('.'+suffix):
                fileList.append(i)
    return fileList

def checkDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clearFile(path):
    open(path, "w").close()

def openAppendFile(path):
    clearFile(path)
    return open(path, 'a')

class Divider:
    def __init__(self, jpgPath, labelPath, proportion_train_val_test = [8,1,1]):
        self.labelPath = labelPath
        self.proportion = proportion_train_val_test
        self.initialImgSetPath = jpgPath
        self.totalImgNames = gainFiles(self.initialImgSetPath)
        self.countImg = len(self.totalImgNames)
        self.indexList = range(self.countImg)
        self.totalProportion = sum(self.proportion)
        self.countTest = int(self.countImg * self.proportion[2] / self.totalProportion)
        self.countVal = int(self.countImg * self.proportion[1] / self.totalProportion)
        self.countTrain = self.countImg - self.countTest - self.countVal
        self.trainValIndexList = random.sample(self.indexList, self.countImg - self.countTest)
        self.valIndexList = random.sample(self.trainValIndexList, self.countVal)

    def divideDataSet(self, dividedDataSetPath):
        self.checkDir(dividedDataSetPath)
        imagePath = dividedDataSetPath + 'images/'
        # self.__checkDir(imagePath)
        labelPath = dividedDataSetPath + 'labels/'
        # self.__checkDir(labelPath)
        trainImgPath = imagePath + 'train/'
        checkDir(trainImgPath)
        trainLabelPath = labelPath + 'train/'
        checkDir(trainLabelPath)
        valImgPath = imagePath + 'val/'
        checkDir(valImgPath)
        valLabelPath = labelPath + 'val/'
        checkDir(valLabelPath)
        testImgPath = imagePath + 'test/'
        checkDir(testImgPath)
        testLabelPath = labelPath + 'test/'
        checkDir(testLabelPath)
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
        with open(dividedDataSetPath+'info.txt', 'w') as f:
            f.write(str(self.countImg) + ' ' + str(self.countTest) + ' ' + str(self.countVal) + ' ' + str(self.countTrain))

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
        with open(dataSetTextPath+'info.txt', 'w') as f:
            f.write(str(self.countImg) + ' ' + str(self.countTest) + ' ' + str(self.countVal) + ' ' + str(self.countTrain))


class DirDivider:
    def __init__(self, ratio = [8, 1, 1]):
        totalProportion = sum(ratio)
        self.countTest = ratio[2] / totalProportion
        self.countVal = ratio[1] / totalProportion
        self.countTrain = 1.0 - self.countTest - self.countVal
    
    def divide(self, dirPath, lanelNames, labelCountThres = None):
        dirPath = Path(dirPath)
        valFile = openAppendFile(dirPath / 'val.txt')
        trainFile = openAppendFile(dirPath / 'train.txt')
        testFile = openAppendFile(dirPath / 'test.txt')
        infoFile = openAppendFile(dirPath / 'info.txt')
        for cls, label in enumerate(lanelNames):
            totalImgNames = gainFiles(str(dirPath / 'labels' / label), 'txt')
            if labelCountThres is None:
                indexList = range(len(totalImgNames))
            else:
                num = labelCountThres[cls]
                if num is None or num < 0:
                    indexList = range(len(totalImgNames))
                else:
                    indexList = random.sample(range(len(totalImgNames)), num)
            countImg = len(indexList)
            countTest = int(countImg * self.countTest)
            countVal = int(countImg * self.countVal)
            countTrain = countImg - countTest - countVal
            trainValIndexList = random.sample(indexList, countImg - countTest)
            valIndexList = random.sample(trainValIndexList, countVal)
            for i in tqdm(indexList, desc=label):
                imgName = './images/' + label + '/' + totalImgNames[i].split('.')[0] + '.jpg\n'
                if i in trainValIndexList:
                    if i in valIndexList:
                        # 写到val.txt里
                        valFile.write(imgName)
                    else:
                        # 写到train.txt里
                        trainFile.write(imgName)
                else:
                    # 写到test.txt里
                    testFile.write(imgName)
            infoFile.write(f'{label}: {countImg} {countTest} {countVal} {countTrain}\n')
        valFile.close()
        trainFile.close()
        testFile.close()
        infoFile.close()
        

if __name__ == "__main__":
    # divider = Divider(rootPath + jpgPath, rootPath + labelPath, TRAIN_RATIO)
    # divider.divideTxt(rootPath)
    # divider.divideDataSet(dataSetPath)
    divider = DirDivider()
    divider.divide(
        'datasets/actionData/yolo', 
        ['Call', 'PlayWithOneHand', 'PlayWithTwoHands', 'Stand', 'Sit'],
        [250, 250, None, None, None]
    )