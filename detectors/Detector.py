class Detector:
    def __init__(self, class_names):
        self.__class_names = class_names

    def detectSingleImage(self, im0, conf_thres, mode = 'image', isNew = True, line_thickness = 2):
        pass

    def getNames(self):
        return self.__class_names
    
    def getLabel(self, cls):
        # return ['phoneWalking', 'call', 'other'][cls]
        return self.__class_names[cls]