"""
demo_live.py — 시연용: 영상 파일 또는 폰 카메라 스트림에 판정 결과를 오버레이

  python demo_live.py --weights final_best.pt --source test.mp4 --save            # 녹화 영상 → 오버레이 mp4 저장
  python demo_live.py --weights final_best.pt --source http://192.168.0.5:4747/video   # DroidCam / IP Webcam
  python demo_live.py --weights final_best.pt --source 0 --tts                    # 노트북 웹캠 + 음성

키: q 종료, 스페이스 일시정지
"""
import argparse, time, threading
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from locator import Locator

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--source', required=True, help='파일 경로 / 스트림 URL / 카메라 번호')
ap.add_argument('--stride', type=int, default=6, help='N프레임마다 1회 추론 (30fps 영상, 6 → 5회/초)')
ap.add_argument('--window', type=int, default=3)
ap.add_argument('--near_m', type=float, default=2.5)
ap.add_argument('--f_norm', type=float, default=0.85)
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--device', default=None)
ap.add_argument('--save', action='store_true')
ap.add_argument('--tts', action='store_true', help='pyttsx3 음성 안내 (pip install pyttsx3)')
ap.add_argument('--font', default='C:/Windows/Fonts/malgun.ttf')
args = ap.parse_args()

loc = Locator(args.weights, f_norm=args.f_norm, near_m=args.near_m, window=args.window, conf=args.conf, device=args.device)
src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
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
    if out['state'] == 'near':   msg, col = out['text'], (0, 230, 110)
    elif out['state'] == 'far':  msg, col = f"{out['landmark_kr']} 감지 · {out['distance_m']:.1f}m 앞", (255, 190, 0)
    else:                        msg, col = '복도 이동 중', (170, 170, 170)
    d.text((14, 12), msg, font=font, fill=col)
    d.text((W - 300, 20), f"{infer_ms:.0f} ms/frame  window={args.window}", font=font_s, fill=(200, 200, 200))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

writer = None
if args.save:
    out_path = Path(args.source).with_name(Path(args.source).stem + '_demo.mp4') if not str(src).startswith('http') and not isinstance(src, int) else Path('live_demo.mp4')
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

i, last, infer_ms, paused = 0, None, 0.0, False
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok: break
        t = i / fps
        if i % args.stride == 0 or last is None:
            t0 = time.perf_counter(); last = loc.process(frame, t); infer_ms = (time.perf_counter() - t0) * 1000
            if last['announce']:
                print(f"[{t:6.1f}s] 🔊 {last['text']}")
                if args.tts and not speak_q: speak_q.append(last['text'])
        vis = draw(frame, last, infer_ms)
        if writer: writer.write(vis)
        cv2.imshow('landmark demo', vis); i += 1
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord(' '): paused = not paused

cap.release(); cv2.destroyAllWindows()
if writer: writer.release(); print('저장:', out_path)
