from .DataWriter import DataWriter
import cv2

class YoloFileWriter(DataWriter):

    def __init__(self, size=1024) -> None:
        super().__init__(None, size)

    def update(self, args):
        videoWriter = None
        while True:
            savePath, mode, cap, videoInfo, isNew = self.read()
            if savePath is None: break
            '''
                img : HWC, BGR
            '''
            if mode == 'image':
                cv2.imwrite(savePath, cap)
            else:   # stream or vedio
                if isNew: # 是一个新的视频或者第一个视频
                    if videoWriter is not None:
                        videoWriter.release() # release previous video writer
                        videoWriter = None
                    if videoInfo: #video
                        fps = videoInfo[0]
                        w = int(videoInfo[1])
                        h = int(videoInfo[2])
                    else:   # stream
                        fps, w, h = 30, cap.shape[1], cap.shape[0]
                    videoWriter = cv2.VideoWriter(str(savePath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                assert(videoWriter != None)
                videoWriter.write(cap) # 是前一个视频的下一帧

    def save(self, savePath, mode, cap, videoInfo, isNew):
        super().save((savePath, mode, cap, videoInfo, isNew))
    
    def toEnd(self):
        self.save(*((None,) * 5))
