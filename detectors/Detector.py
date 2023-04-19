# from abc import abstractclassmethod, abstractproperty, abstractmethod

class Detector:
    def __init__(self, class_names):
        self.__class_names = class_names

    # @abstractmethod
    def detectSingleImage(self, im0, conf_thres, mode = 'image', isNew = True, line_thickness = 2):
        pass

    @property
    def class_names(self):
        return self.__class_names
    
    def getLabel(self, cls):
        return self.__class_names[cls]
