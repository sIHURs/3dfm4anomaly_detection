import time, torch

class SpanTimer:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.wall_marks = {}
        self.cuda_marks = {}

    def mark(self, name: str):
        self.wall_marks[name] = time.perf_counter()
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e.record()
            self.cuda_marks[name] = e

    def span(self, start: str, end: str):
        wall = self.wall_marks[end] - self.wall_marks[start]
        if self.use_cuda:
            torch.cuda.synchronize()
            cuda_ms = self.cuda_marks[start].elapsed_time(self.cuda_marks[end])
            return wall, cuda_ms / 1000.0
        return wall, None