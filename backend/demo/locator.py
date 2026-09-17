"""
locator.py — 위치 판정 모듈 (시연·평가·FastAPI 공용)

프레임 → YOLO 탐지 → bbox 크기로 거리 추정(핀홀) → 최근 N프레임 다수결 → 상태/안내문
  상태: 'none'(탐지 없음) / 'far'(탐지, 거리 > near_m) / 'near'(거리 <= near_m → 안내)
  안내: 문장이 바뀔 때 즉시, 같은 문장은 announce_every 초마다 한 번

사용:
    from locator import Locator
    loc = Locator('final_best.pt')
    out = loc.process(frame, t_sec)      # dict: landmark, distance_m, state, confidence, text, announce, detections
"""
import time
from collections import deque, Counter
import numpy as np
from ultralytics import YOLO

# 실제 높이/너비(m) — 거리 추정용. 정수기는 안내 대상에서 제외.
REAL_SIZE = {'2_class': (0.20, 0.13), '4_class': (0.20, 0.13), 'front_door': (2.5, 4.4), 'rear_door': (2.1, 1.8)}
DISPLAY = {'2_class': '2강의실', '4_class': '4강의실', 'front_door': '앞문', 'rear_door': '뒷문'}


class Locator:
    def __init__(self, weights, f_norm=0.85, near_m=2.5, window=3, conf=0.5, imgsz=640, device=None,
                 announce_every=5.0, targets=tuple(REAL_SIZE)):
        self.model = YOLO(str(weights))
        self.f_norm, self.near_m, self.window = f_norm, near_m, window
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.announce_every, self.targets = announce_every, set(targets)
        self.buf = deque(maxlen=window)            # (landmark or None, distance or None)
        self.last_text, self.last_announce_t = None, -1e9

    # ---------- 거리: d = f_norm * real * W / px ----------
    def _distance(self, name, x1, y1, x2, y2, W, H):
        h_real, w_real = REAL_SIZE[name]
        clipped = y1 < 2 or y2 > H - 2
        px = (x2 - x1) if clipped else (y2 - y1)
        real = w_real if clipped else h_real
        return float(self.f_norm * real * W / max(px, 1.0))

    def detect(self, frame):
        H, W = frame.shape[:2]
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        dets = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            name = r.names[int(c)]
            if name not in self.targets: continue
            x1, y1, x2, y2 = map(float, b)
            dets.append({'name': name, 'conf': float(s), 'box': (x1, y1, x2, y2),
                         'distance_m': self._distance(name, x1, y1, x2, y2, W, H)})
        return dets

    def process(self, frame, t=None):
        return self.update(self.detect(frame), t)

    def update(self, dets, t=None):
        """탐지 결과(dets)만으로 판정 갱신 — 평가 시 캐시된 탐지를 윈도우 크기별로 재생할 때 사용"""
        t = time.time() if t is None else t
        # 프레임 대표: 가장 가까운(=가장 큰) 탐지. 여러 랜드마크가 보이면 가까운 것이 안내 대상.
        top = min(dets, key=lambda d: d['distance_m']) if dets else None
        self.buf.append((top['name'], top['distance_m'], top['conf']) if top else (None, None, 0.0))

        votes = Counter(n for n, _, _ in self.buf if n)
        if votes:
            landmark, n_votes = votes.most_common(1)[0]
            ds = [d for n, d, _ in self.buf if n == landmark]
            cs = [c for n, _, c in self.buf if n == landmark]
            distance = float(np.median(ds)); confidence = float(np.mean(cs))
            state = 'near' if distance <= self.near_m else 'far'
        else:
            landmark, distance, confidence, state = None, None, 0.0, 'none'

        text = f'앞쪽에 {DISPLAY[landmark]}이 있습니다' if state == 'near' else None
        announce = False
        if text is not None and (text != self.last_text or t - self.last_announce_t >= self.announce_every):
            announce, self.last_announce_t = True, t
        if text != self.last_text: self.last_text = text

        return {'t': t, 'landmark': landmark, 'landmark_kr': DISPLAY.get(landmark), 'distance_m': distance,
                'state': state, 'confidence': confidence, 'text': text, 'announce': announce, 'detections': dets}

    def reset(self):
        self.buf.clear(); self.last_text, self.last_announce_t = None, -1e9
