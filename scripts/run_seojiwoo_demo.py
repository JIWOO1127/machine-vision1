#!/usr/bin/env python
"""서지우님(machine-vision1) 판정 로직으로 구동하는 시연 스크립트.

원본: external/seojiwoo_core/demo_live.py — 로직은 원본과 동일하며,
      classroom_locator.seojiwoo 패키지에서 Locator/Navigator/OcrVerifier를
      불러오고 기본 모델 경로를 이 프로젝트의 models/yolo/에 맞췄다.

사용 예:
    # route만 지정 (로고 단계 없이 바로 시작, 5클래스 final_best.pt로 충분)
    python scripts/run_seojiwoo_demo.py --source data/raw/test_video.mp4 ^
        --route front_door 4_class 2_class rear_door --ocr

    # 로고 찾기부터 시작하는 전체 시나리오 (6클래스 모델 필요, 기본값으로 자동 사용)
    python scripts/run_seojiwoo_demo.py --source 0 --nav --ocr --tts

키: q 종료, 스페이스 일시정지
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# src/ 를 import 경로에 추가 (다른 scripts/*.py와 동일한 패턴)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_locator.seojiwoo import Locator  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "final_best.pt"
DEFAULT_LOGO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "final_best_v2_with_logo.pt"

ap = argparse.ArgumentParser()
ap.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='주 모델 (표지판·문): 5클래스 v1 권장')
ap.add_argument('--logo_weights', default=None, help='로고 전용 모델 (6클래스). --nav 사용 시 기본값 자동 적용')
ap.add_argument('--source', required=True, help='파일 경로 / 스트림 URL / 카메라 번호')
ap.add_argument('--stride', type=int, default=6, help='N프레임마다 1회 추론 (30fps 영상, 6 → 5회/초)')
ap.add_argument('--window', type=int, default=5)
ap.add_argument('--min_votes', type=int, default=4)
ap.add_argument('--sign_conf', type=float, default=0.35, help='표지판 conf (OCR 검증과 함께 쓰므로 낮게)')
ap.add_argument('--no_approach', action='store_true', help='접근 추세 조건 끄기')
ap.add_argument('--route', nargs='*', default=None, help='동선 순서, 예: front_door 4_class 2_class rear_door')
ap.add_argument('--near_m', type=float, default=3.0)
ap.add_argument('--f_norm', type=float, default=0.85)
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--device', default=None)
ap.add_argument('--save', action='store_true')
ap.add_argument('--tts', action='store_true', help='pyttsx3 음성 안내 (pip install pyttsx3)')
ap.add_argument('--ocr', action='store_true', help='근접 시 표지판 글자 검증 (pip install easyocr)')
ap.add_argument('--nav', action='store_true', help='시나리오 내비게이션 (로고→4→2→뒷문, 방향 안내 포함)')
ap.add_argument('--wall_m', type=float, default=1.2)
ap.add_argument('--merge_signs', action='store_true', help='2/4 표지판을 하나로 합쳐 투표, 숫자는 OCR (--ocr 필요)')
ap.add_argument('--font', default='C:/Windows/Fonts/malgun.ttf')
args = ap.parse_args()

# --nav는 logo 클래스가 있는 6클래스 모델이 필요 → 별도로 안 주면 기본 v2 모델 사용
if args.nav and args.logo_weights is None:
    args.logo_weights = str(DEFAULT_LOGO_WEIGHTS)

verifier = None
if args.ocr:
    from classroom_locator.seojiwoo.ocr_verify import OcrVerifier
    verifier = OcrVerifier(gpu=(args.device != 'cpu'))
loc = Locator(args.weights, logo_weights=args.logo_weights, verifier=verifier, merge_signs=(args.merge_signs or args.nav), targets=('2_class', '4_class', 'front_door', 'rear_door', 'logo'), f_norm=args.f_norm, near_m=args.near_m, window=args.window, min_votes=args.min_votes,
              conf=args.conf, sign_conf=args.sign_conf, approach_only=not args.no_approach, route=args.route, device=args.device)
nav = None
if args.nav:
    from classroom_locator.seojiwoo import Navigator
    nav = Navigator(loc, wall_m=args.wall_m)
src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
font = ImageFont.truetype(args.font, 34) if Path(args.font).exists() else ImageFont.load_default()
font_s = ImageFont.truetype(args.font, 22) if Path(args.font).exists() else ImageFont.load_default()

# ---------- TTS (별도 스레드, 발화 중이면 새 요청은 버림) ----------
speak_q = []
if args.tts:
    import pyttsx3
    def _tts_worker():
        eng = pyttsx3.init(); eng.setProperty('rate', 170)
        while True:
            if speak_q:
                txt = speak_q.pop(0); eng.say(txt); eng.runAndWait()
            else: time.sleep(0.05)
    threading.Thread(target=_tts_worker, daemon=True).start()

def draw(frame, out, infer_ms):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(img)
    for det in out['detections']:
        x1, y1, x2, y2 = det['box']; near = det['distance_m'] <= args.near_m
        col = (0, 200, 90) if near else (255, 170, 0)
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        d.text((x1, max(0, y1 - 26)), f"{det['name']} {det['conf']:.2f}  {det['distance_m']:.1f}m", font=font_s, fill=col)
    # 상단 상태 바
    d.rectangle([0, 0, W, 62], fill=(0, 0, 0))
    if out.get('nav_text'):
        msg = out['nav_text']; col = (0, 230, 110) if out.get('nav_announce') or out.get('done') else (255, 220, 120)
        d.text((14, H - 62), f"step {out['step_idx']}: {out['step']}  target={out['target']}  steer={out.get('steer')}", font=font_s, fill=(200, 200, 200))
    elif out['state'] == 'near':   msg, col = out['text'], (0, 230, 110)
    elif out['state'] == 'far':  msg, col = f"{out['landmark_kr']} 감지 · {out['distance_m']:.1f}m 앞" + (f"  ({out['reason']})" if out.get('reason') else ''), (255, 190, 0)
    else:                        msg, col = '복도 이동 중', (170, 170, 170)
    d.text((14, 12), msg, font=font, fill=col)
    if out.get('ocr'): d.text((14, H - 34), f"OCR {out['ocr']}", font=font_s, fill=(120, 200, 255))
    nxt = out.get('route_next'); d.text((W - 360, 20), f"{infer_ms:.0f} ms  w={args.window}/{args.min_votes}" + (f"  next:{nxt}" if nxt else ''), font=font_s, fill=(200, 200, 200))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

writer = None
if args.save:
    out_path = Path(args.source).with_name(Path(args.source).stem + '_demo.mp4') if not str(src).startswith('http') and not isinstance(src, int) else Path('live_demo.mp4')
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

i, last, infer_ms, paused = 0, None, 0.0, False
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok:
            fails = globals().get('fails', 0) + 1; globals()['fails'] = fails
            if fails > 30: print('스트림 끊김, 종료'); break
            time.sleep(0.1); continue
        globals()['fails'] = 0
        t = i / fps
        if i % args.stride == 0 or last is None:
            t0 = time.perf_counter(); last = nav.update(frame, t) if nav else loc.process(frame, t); infer_ms = (time.perf_counter() - t0) * 1000
            say = last.get('nav_text') if nav and last.get('nav_announce') else (last['text'] if not nav and last['announce'] else None)
            if say:
                print(f"[{t:6.1f}s] 🔊 {say}")
                if args.tts and not speak_q: speak_q.append(say)
        vis = draw(frame, last, infer_ms)
        if writer: writer.write(vis)
        cv2.imshow('landmark demo', vis); i += 1
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord(' '): paused = not paused

cap.release(); cv2.destroyAllWindows()
if writer: writer.release(); print('저장:', out_path)
