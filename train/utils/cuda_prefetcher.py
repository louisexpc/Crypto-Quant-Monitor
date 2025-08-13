# ---- cuda_prefetcher.py ----
import torch

class CudaPrefetcher:
    """把 DataLoader 迭代出的 batch 用另一條 CUDA stream 預先搬上 GPU。"""
    def __init__(self, loader, device="cuda", non_blocking=True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.non_blocking = non_blocking
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            Xb, yb = batch
            Xb = Xb.to(self.device, non_blocking=self.non_blocking)
            yb = yb.to(self.device, non_blocking=self.non_blocking)
            self.next_batch = (Xb, yb)

    def __iterme__(self):  # 避免和 python 的 __iter__ 衝突，看你喜歡怎麼命名
        return self

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            return None
        self._preload()
        return batch
