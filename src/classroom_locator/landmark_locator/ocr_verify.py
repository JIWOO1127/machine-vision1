"""
ocr_verify.py — 근접 시 표지판 글자로 YOLO 판정 검증 (팀원 OCR 로직 축약판)

원본: https://github.com/JIWOO1127/machine-vision1/blob/feature/seojiwoo-detection/seojiwoo/core/ocr_verify.py
      (서지우님 작업물) — 로직은 원본과 동일.

  crop → 확대 → EasyOCR(ko,en) → 한글/숫자 조각 합치기 → 판정
    반환: 'confirm' (숫자 일치) / 'reject' (회의실이거나 다른 숫자) / 'unknown' (읽은 글자 없음)

  pip install easyocr
  단독 테스트: python -m classroom_locator.landmark_locator.ocr_verify 사진.jpg [모델경로]
"""
import re, time
import cv2, numpy as np

_KO_DIGIT = re.compile(r'[0-9가-힣]')
SIGN_DIGIT = {'2_class': '2', '3_class': '3', '4_class': '4'}   # 3_class 는 YOLO 클래스 없이 OCR 로만 구분


class OcrVerifier:
    def __init__(self, gpu=True, min_conf=0.3, upscale_to=200):
        import easyocr
        self.reader = easyocr.Reader(['ko', 'en'], gpu=gpu, verbose=False)
        self.min_conf, self.upscale_to = min_conf, upscale_to

    def read(self, frame, box, pad=0.15):
        x1, y1, x2, y2 = box; H, W = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        x1, y1 = int(max(0, x1 - w * pad)), int(max(0, y1 - h * pad))
        x2, y2 = int(min(W, x2 + w * pad)), int(min(H, y2 + h * pad))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return []
        s = self.upscale_to / max(1, crop.shape[0])
        if s > 1: crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        res = self.reader.readtext(crop, allowlist='0123456789강의실회Class Room')   # 허용 문자 제한으로 오독 감소
        res = sorted([r for r in res if r[2] >= self.min_conf], key=lambda r: r[0][0][0])
        return [r[1] for r in res]

    def verify(self, frame, box, yolo_class):
        texts = self.read(frame, box)
        joined = ' '.join(t for t in texts if _KO_DIGIT.search(t)).replace(' ', '')
        if not joined: return 'unknown', joined
        if '회의' in joined or 'meeting' in joined.lower(): return 'reject', joined
        digits = re.findall(r'\d', joined)
        want = SIGN_DIGIT.get(yolo_class)
        if not digits: return 'unknown', joined          # "강의실"만 읽힘 → 판단 보류(팀원 로직과 동일)
        return ('confirm' if want in digits else 'reject'), joined


def build_verifier(gpu: bool | None = None) -> 'OcrVerifier | None':
    """OcrVerifier를 안전하게 생성합니다. easyocr 미설치/초기화 실패 시 경고만
    남기고 None을 반환해서, 호출부가 OCR 없이(YOLO 판정만으로) 계속 동작할 수
    있게 합니다 (pipeline/grid_tracker.py, pipeline/core.py 공용)."""
    import logging
    try:
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        return OcrVerifier(gpu=gpu)
    except Exception as e:  # noqa: BLE001 - OCR은 선택 기능, 실패해도 나머지 판정은 계속되어야 함
        logging.getLogger('classroom_locator').warning(f'OcrVerifier 초기화 실패, OCR 없이 계속 진행: {e}')
        return None


def read_digit(verifier: 'OcrVerifier | None', frame, box, digits_allowed: str = '234') -> str | None:
    """표지판 bbox 글자를 읽어서 digits_allowed 중 한 글자만 뚜렷하게 읽히면
    그 숫자를 반환합니다 (여러 개거나 하나도 없으면 None - 애매하면 판단 보류).
    """
    digit, _ = read_sign_info(verifier, frame, box, digits_allowed)
    return digit


def read_sign_info(
    verifier: 'OcrVerifier | None', frame, box, digits_allowed: str = '234'
) -> tuple[str | None, bool]:
    """표지판 bbox를 한 번만 OCR로 읽어서 (읽힌 숫자, 회의실 여부)를 함께
    반환합니다. 2_class/4_class 표지판과 생김새가 거의 동일한 회의실
    표지판("O 회의실")이 YOLO에 room2/room4로 오분류되는 경우를 걸러내기
    위함 - `OcrVerifier.verify()`의 '회의' 감지 로직과 같은 기준을 쓰되,
    숫자 추출과 한 번의 `read()` 호출로 같이 처리한다."""
    if verifier is None or box is None:
        return None, False
    texts = verifier.read(frame, box)
    joined = ''.join(texts)
    is_meeting = '회의' in joined or 'meeting' in joined.lower()
    digits = {ch for text in texts for ch in text if ch in digits_allowed}
    digit = next(iter(digits)) if len(digits) == 1 else None
    return digit, is_meeting


if __name__ == '__main__':
    import sys
    from ultralytics import YOLO
    img = cv2.imread(sys.argv[1]); v = OcrVerifier()
    m = YOLO(sys.argv[2] if len(sys.argv) > 2 else 'final_best.pt')
    r = m.predict(img, conf=0.4, verbose=False)[0]
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        name = r.names[int(c)]
        if name in SIGN_DIGIT:
            t0 = time.perf_counter(); out = v.verify(img, tuple(map(float, b)), name)
            print(name, '→', out, f'{(time.perf_counter()-t0)*1000:.0f} ms')
