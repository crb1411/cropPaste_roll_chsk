from collections import OrderedDict

try:
    import openslide
except Exception:
    openslide = None

class SlideLRU:
    def __init__(self, max_open=100, open_fn=None, shared_val=None):
        assert max_open >= 1
        self.max_open = max_open
        self.shared_val = shared_val
        self.open_fn = open_fn or (lambda p: openslide.OpenSlide(p))
        self.cache = OrderedDict()  # path -> handle

    def get(self, path: str):
        # 命中：移到队尾并返回
        if path in self.cache:
            self.cache.move_to_end(path, last=True)
            return self.cache[path]
        # 未命中：打开并放入
        h = self.open_fn(path)
        self.cache[path] = h
        self.cache.move_to_end(path, last=True)
        # 超限：淘汰最旧并关闭
        if len(self.cache) > self.max_open:
            _, old = self.cache.popitem(last=False)
            self._safe_close(old)
        if self.shared_val is not None:
            self.shared_val.value = len(self.cache)
        return h

    def close(self, path: str):
        h = self.cache.pop(path, None)
        if h is not None:
            self._safe_close(h)

    def close_all(self):
        for _, h in list(self.cache.items()):
            self._safe_close(h)
        self.cache.clear()

    @staticmethod
    def _safe_close(h):
        close = getattr(h, "close", None)
        if callable(close):
            close()

# ===== 用法 =====
# lru = SlideLRU(max_open=8)  # 默认用 openslide.OpenSlide
# slide = lru.get("/data/a.svs")
# img = slide.read_region((0,0), 0, (512,512))
# lru.close_all()

if __name__ == "__main__":
    lru = SlideLRU(max_open=8)
    svs_path = "/data/work/ruijin/data/CMU-21744683275045_2864_1.svs"
    slide = lru.get(svs_path)
    print(len(lru.cache))
    img = slide.read_region((0,0), 0, (512,512))
    lru.close(svs_path)
    print(len(lru.cache))
    
    
    