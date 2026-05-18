# based on FramePack https://github.com/lllyasviel/FramePack

import time
import traceback

from threading import Thread, Lock


class _TaskRunner:
    def __init__(self, runner_name="default"):
        self.runner_name = str(runner_name or "default")
        self.thread_name = self.runner_name.replace("_", " ").strip().title() or "Default"
        self.task_queue = []
        self.lock = Lock()
        self.thread = None

    def _process_tasks(self):
        while True:
            task = None
            with self.lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    
            if task is None:
                time.sleep(0.001)
                continue
                
            func, args, kwargs, thread_name = task
            current_name = None
            thread = self.thread
            try:
                if thread_name and thread is not None:
                    current_name = thread.name
                    thread.name = thread_name
                func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))

                # print(f"Error in listener thread: {e}")
            finally:
                if current_name is not None and thread is not None:
                    thread.name = current_name

    def add_task(self, func, *args, thread_name=None, **kwargs):
        with self.lock:
            self.task_queue.append((func, args, kwargs, thread_name))
            thread = None if self.thread is not None else Thread(target=self._process_tasks, daemon=True, name=self.thread_name)
            if thread is not None:
                self.thread = thread

        if thread is not None:
            thread.start()


class Listener:
    runners = {}
    lock = Lock()

    @classmethod
    def _get_runner(cls, runner_name="default"):
        runner_name = str(runner_name or "default")
        with cls.lock:
            runner = cls.runners.get(runner_name, None)
            if runner is None:
                runner = _TaskRunner(runner_name)
                cls.runners[runner_name] = runner
            return runner

    @classmethod
    def add_task(cls, func, *args, runner_name="default", thread_name=None, **kwargs):
        cls._get_runner(runner_name).add_task(func, *args, thread_name=thread_name, **kwargs)


def async_run(func, *args, thread_name=None, **kwargs):
    Listener.add_task(func, *args, thread_name=thread_name, **kwargs)


def async_run_in(runner_name, func, *args, thread_name=None, **kwargs):
    Listener.add_task(func, *args, runner_name=runner_name, thread_name=thread_name, **kwargs)


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
