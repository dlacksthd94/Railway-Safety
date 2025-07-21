import os
import time

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self           # (optional) so you can read .elapsed later
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        elapsed = self.format_hms(self.elapsed)
        print(f"[{self.label}]\t elapsed: {elapsed}")
            
    def format_hms(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60               # still a float now
        return f"{h:02d}:{m:02d}:{s:06.3f}"   # e.g. 00:01:02.357
