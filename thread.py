import threading
import time
import os

class mythread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self);
    def run(self):
        os.system(" sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' ")


th = mythread();
th.start();
th.join()