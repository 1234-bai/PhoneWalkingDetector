
import torch.multiprocessing as mp


class DataWriter:

    def __init__(self, saver, size = 1024) -> None:
        self.saver = saver
        self.queue = mp.Queue(maxsize = size)

    def start(self, **args):
        p = mp.Process(target=self.update, args=(args,))
        p.start()
        self.worker = p
        return self

    def update(self, args):
        while True:
            item = self.read()
            if item is None: break
            self.saver(item, **args)

    def save(self, item):
        self.queue.put(item)

    def read(self):
        return self.queue.get(timeout=1)

    def running(self):
        # indicate that the thread is still running
        return not self.queue.empty()

    def count(self):
        # indicate the remaining images
        return self.queue.qsize()
    
    def stop(self):
        # indicate that the thread should be stopped
        self.save(None)
        self.worker.join()

    def terminate(self):
        # directly terminate
        self.worker.terminate()

    def clear_queues(self):
        self.clear(self.queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def toEnd(self):
        self.save(None)
