import torch.multiprocessing as mp

from libs.yolov5 import loadData

class DataReader:

    def __init__(self, source, batch_size, queue_size = 1024, **args) -> None:
        self.source = source
        self.args = args 
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize = queue_size)

    def start(self):
        p = mp.Process(target=self.update, args=())
        p.start()
        self.worker = p
        return self
    
    def update(self):
        dataset = loadData(self.source, **(self.args))
        items = []
        count = 0
        for path, im0s, vid_info, infoStr in dataset:
            if dataset.mode == 'stream':
                path = path[0] + '.mp4'
                im0s = im0s[0]
            items.append((path, im0s, vid_info, infoStr))
            count += 1
            if count == self.batch_size:
                self.queue.put(items.copy())
                items.clear()
                count = 0
        if len(items):
            self.queue.put(items.copy())
        self.queue.put(None)

    def read(self):
        return self.queue.get()

    def running(self):
        # indicate that the thread is still running
        return not self.queue.empty()

    def count(self):
        # indicate the remaining images
        return self.queue.qsize()
    
    def stop(self):
        # indicate that the thread should be stopped
        self.worker.join()

    def terminate(self):
        # directly terminate
        self.worker.terminate()

    def clear_queues(self):
        self.clear(self.queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    @property
    def mode(self):
        return self.mode
    