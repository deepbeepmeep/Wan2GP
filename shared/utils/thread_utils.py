# based on FramePack https://github.com/lllyasviel/FramePack

import time
import traceback

from threading import Thread, Lock


class Listener:
    task_queue = []
    lock = Lock()
    thread = None
    
    @classmethod
    def _process_tasks(cls):
        while True:
            task = None
            with cls.lock:
                if cls.task_queue:
                    task = cls.task_queue.pop(0)
                    
            if task is None:
                time.sleep(0.001)
                continue
                
            func, args, kwargs, thread_name = task
            current_name = None
            try:
                if thread_name:
                    current_name = cls.thread.name
                    cls.thread.name = thread_name
                func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))

                # print(f"Error in listener thread: {e}")
            finally:
                if current_name is not None:
                    cls.thread.name = current_name
    
    @classmethod
    def add_task(cls, func, *args, thread_name=None, **kwargs):
        with cls.lock:
            cls.task_queue.append((func, args, kwargs, thread_name))

        if cls.thread is None:
            cls.thread = Thread(target=cls._process_tasks, daemon=True)
            cls.thread.start()


def async_run(func, *args, thread_name=None, **kwargs):
    Listener.add_task(func, *args, thread_name=thread_name, **kwargs)


class FIFOQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()

    def push(self, cmd, data = None):
        with self.lock:
            self.queue.append( (cmd, data) )

    def pop(self):
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None

    def top(self):
        with self.lock:
            if self.queue:
                return self.queue[0]
            return None

    def next(self):
        while True:
            with self.lock:
                if self.queue:
                    return self.queue.pop(0)

            time.sleep(0.001)


class AsyncStream:
    def __init__(self):
        self.input_queue = FIFOQueue()
        self.output_queue = FIFOQueue()
