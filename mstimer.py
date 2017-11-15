import time

class MSTimer(object):
    def __init__(self):
        self.start_timer()
        return
    def start_timer(self):
        self.start = time.time()
        return
    def stop_timer(self):
        elapsed = time.time()
        elapsed = elapsed - self.start
        return elapsed