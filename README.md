# classroom-locator

YOLO(객체 탐지) + OCR(문자 인식)로 강의실 표지판/문/로고를 인식해 복도 위
**현재 위치를 추적**하고 다음 목적지까지 안내하는 프로젝트입니다.

- `src/classroom_locator/` — 핵심 판정 로직. 웹캠/영상 재생으로 CLI에서 바로 실행
- `webapp/` — 같은 판정 로직을 실시간 카메라 + 사진/영상 업로드로 쓰는 폰 브라우저용 웹 UI

## 폴더 구조

```
classroom-locator/
├── .vscode/                     # VSCode 실행/설정 (launch.json, settings.json 등)
├── configs/
│   ├── default.yaml             # detector/localization/realtime 등 전역 설정
│   └── locations.yaml           # 위치별 grid_cell/snap_distance_m/안내 문구
│                                 # (root CLI와 webapp이 이 파일 하나를 공용으로 읽음)
├── data/
│   ├── raw/                     # 원본 사진/영상 (배치 모드 입력)
│   ├── processed/                # 전처리된 이미지
│   ├── annotations/              # YOLO 학습용 라벨(txt) 등
│   └── dataset/                  # 학습용 데이터셋(train/valid/test)
├── models/
│   ├── yolo/landmark_best.pt     # 표지판·문 전용 (5클래스) - root CLI와 webapp이 detector.weights로 참조
│   ├── yolo/logo_best.pt         # 로고 전용 - detector.logo_weights로 참조 (2모델 구성, seojiwoo/core 최종 방식)
│   └── ocr/                      # OCR 관련 모델/가중치 (필요 시)
├── notebooks/                    # 데이터 탐색, 실험용 주피터 노트북
├── src/classroom_locator/        # 핵심 파이썬 패키지
│   ├── detection/                 # YOLO 등 객체 탐지 모듈 (추상화)
│   ├── landmark_locator/          # 실측 물리 크기 기반 거리 추정 + 근접 시 OCR 정정 (Locator)
│   ├── localization/               # 인식 결과 → 위치 매칭 / 지도 좌표 로직
│   ├── pipeline/                   # 실시간·배치 파이프라인 조립, 격자 위치 추적(grid_tracker), 위치 잠금
│   ├── utils/                      # 공용 유틸 (이미지, 로깅)
│   └── config.py                   # 설정 로더
├── scripts/                      # CLI 실행 스크립트 (진입점)
│   ├── run_realtime.py           # 웹캠/영상 실시간 실행
│   ├── run_batch.py              # 사진 폴더 배치 실행
│   ├── run_landmark_demo.py      # landmark_locator 단독 시연 (오버레이 영상 저장, TTS 등)
│   └── train_yolo.py             # YOLO 학습 스크립트
├── webapp/                       # 폰 브라우저용 웹 UI
│   ├── frontend/                 # React + Vite (촬영/영상 업로드/실시간 카메라 + TTS 음성 안내)
│   └── backend/                  # Flask 서버
│       ├── app.py                 # API 라우팅 (/api/analyze, /api/analyze-frame 등)
│       ├── services/
│       │   ├── vision_bridge.py   # src/classroom_locator 판정 로직 ↔ 프론트 JSON 응답 어댑터
│       │   └── locations.py       # /api/locations CRUD 전용 (판정 로직과 무관한 별도 저장소)
│       └── data/locations.json    # 위 CRUD가 쓰는 데이터
├── tests/                        # pytest 테스트
├── outputs/                      # 실행 결과물 (로그, 결과 이미지/JSON)
└── docs/                         # 설계 문서, 진행 로그(setup_log.md)
```

## 시작하기 (가상환경)

conda 가상환경 이름은 **`classroom-locator`**입니다. (설치 과정에서 겪었던
문제들과 상세 이력은 [docs/setup_log.md](docs/setup_log.md) 참고)

```powershell
conda create -n classroom-locator python=3.11 -y
conda activate classroom-locator

cd classroom-locator
pip install -r requirements.txt

# GPU(NVIDIA) 있으면 CUDA 빌드로 덮어쓰기 (CUDA 버전은 nvidia-smi로 확인)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

루트의 이 `requirements.txt` 하나로 `src/`의 CLI 스크립트와 `webapp/`이 공용으로
쓰는 탐지·거리추정·OCR 로직까지 전부 커버됩니다. 웹 서버(Flask)를 띄우려면
아래 "웹 UI 실행하기"에서 `webapp/backend/requirements.txt`를 추가로 설치해야
합니다.

```bash
python scripts/run_batch.py --input data/raw --config configs/default.yaml
python scripts/run_realtime.py --camera 0 --config configs/default.yaml
```

## 실행 스크립트 옵션

### `scripts/run_realtime.py` — 웹캠/영상 실시간 실행

```bash
python scripts/run_realtime.py --camera 0
python scripts/run_realtime.py --source data/raw/test_video.mp4
python scripts/run_realtime.py --source data/raw/test_video.mp4 --snapshot --grid-map
```

| 인자 | 설명 |
|---|---|
| `--config PATH` | 설정 파일 경로 (기본 `configs/default.yaml`) |
| `--camera N` | 카메라 인덱스 (설정 파일 값 덮어쓰기) |
| `--source PATH` | 웹캠 대신 재생할 영상 파일 경로. `--camera`보다 우선함 |
| `--snapshot` / `--no-snapshot` | 위치가 바뀔 때마다 그 순간 화면을 `outputs/results/realtime_snapshots/`에 저장할지 여부 (기본 꺼짐, 설정 파일의 `realtime.save_snapshots`로도 조정 가능) |
| `--grid-map` / `--no-grid-map` | 격자 지도 위치 창 표시 여부 (기본 꺼짐, 설정 파일의 `realtime.show_grid_map`으로도 조정 가능) |

### `scripts/run_batch.py` — 사진 폴더 배치 처리

```bash
python scripts/run_batch.py --input data/raw --config configs/default.yaml
```

| 인자 | 설명 |
|---|---|
| `--input PATH` | 처리할 이미지 폴더 (하위 폴더까지 재귀적으로 읽음) |
| `--config PATH` | 설정 파일 경로 |

### `scripts/run_landmark_demo.py` — landmark_locator 단독 시연

```bash
python scripts/run_landmark_demo.py --source 0 --tts
python scripts/run_landmark_demo.py --source data/raw/test_video.mp4 --save
python scripts/run_landmark_demo.py --source data/raw/test_video.mp4 --ocr --route front_door 4_class 2_class rear_door
python scripts/run_landmark_demo.py --source data/raw/test_video.mp4 --nav --ocr --tts   # 로고→벽→4→3→2→뒷문 시나리오
```

| 인자 | 설명 |
|---|---|
| `--weights PATH` | 주 모델 - 표지판·문 전용 5클래스 (기본 `models/yolo/landmark_best.pt`) |
| `--logo_weights PATH` | 로고 전용 모델 (기본 `models/yolo/logo_best.pt`, 2모델 구성이 기본값) |
| `--source PATH` | 파일 경로 / 스트림 URL / 카메라 번호 (필수) |
| `--stride N` | N프레임마다 1회 추론 (기본 6, 30fps 영상 기준 초당 5회) |
| `--window N` / `--min_votes N` | 최근 N프레임 중 몇 표 이상 같아야 확정할지 (기본 5 / 4) |
| `--sign_conf F` | 표지판(2_class/4_class) confidence 임계값 (OCR 검증과 함께 쓰므로 낮게, 기본 0.35) |
| `--no_approach` | 접근 추세(거리가 줄어드는 중인지) 확인 조건 끄기 |
| `--route NAME...` | 동선 순서 고정, 예: `front_door 4_class 2_class rear_door` (`--nav`와 함께 쓰지 않음) |
| `--near_m` / `--f_norm` / `--conf` | 근접 판정 거리(m), 거리 추정 보정 계수, 기본 confidence |
| `--device` | `cuda`/`cpu` 등 강제 지정 (기본 자동 감지) |
| `--save` | 오버레이 결과를 mp4로 저장 |
| `--tts` | pyttsx3로 음성 안내 (`pip install pyttsx3` 필요) |
| `--ocr` | 근접 시 표지판 글자 OCR 검증 (`pip install easyocr` 필요) |
| `--nav` | 시나리오 내비게이션 (로고→벽→4강의실→3강의실→2강의실→뒷문, 좌/우/직진 방향 안내). `--route`보다 우선하며 `--merge_signs`를 자동으로 켬 |
| `--wall_m` | `--nav` 전용: 로고 벽에 도달했다고 볼 거리(m), 기본 1.2 |
| `--merge_signs` | 2_class/4_class를 하나로 합쳐 투표하고 숫자는 OCR로만 구분 (`--ocr` 필요) |
| `--font PATH` | 한글 표시용 폰트 경로 (기본 맑은 고딕) |

### `scripts/train_yolo.py` — YOLO 학습

```bash
python scripts/train_yolo.py --data data/dataset/data.yaml --model yolov8n.pt --epochs 100
```

| 인자 | 설명 |
|---|---|
| `--data PATH` | YOLO 데이터셋 yaml 경로 |
| `--model NAME` | 베이스 모델/가중치 (기본 `yolov8n.pt`) |
| `--epochs N` | epoch 수 (기본 100) |
| `--imgsz N` | 입력 이미지 크기 (기본 640) |
| `--project PATH` | 결과 저장 위치 (기본 `outputs/results`) |
| `--name NAME` | 실행 이름 (기본 `sign_detector`) |

## 웹 UI 실행하기

### 1. 백엔드 추가 설치 (최초 1회)

```powershell
conda activate classroom-locator
pip install -r webapp/backend/requirements.txt
```

루트 `requirements.txt`에는 없는 `Flask`, `flask-cors`, `imageio-ffmpeg`(결과
영상을 폰에서 재생되는 형식으로 변환)이 추가로 설치됩니다.

### 2. 프론트엔드 빌드 (최초 1회, `App.jsx`를 고칠 때마다 다시)

```powershell
cd webapp/frontend
npm install
npm run build
```

`npm run build`를 다시 안 하면 `webapp/frontend/dist/`가 예전 코드로 남아서,
소스를 고쳐도 브라우저에는 반영되지 않습니다.

### 3. 서버 실행

```powershell
cd webapp/backend
python app.py
```

기본으로 `http://localhost:5000`에서 뜨고, `cloudflared`(`.tools/cloudflared.exe`
또는 시스템 PATH)가 있으면 폰 등 외부 기기에서 접속 가능한 임시 주소도 콘솔에
같이 출력됩니다:

```
[tunnel] 외부 접속 주소: https://xxxx.trycloudflare.com/?key=xxxxxxxx
```

이 링크를 폰 브라우저로 열면 됩니다. 서버를 켤 때 브라우저 창을 자동으로
새로 열지는 않으니, 위 주소를 직접 열어야 합니다.

| 환경변수 | 설명 |
|---|---|
| `PORT` | 로컬 포트 (기본 5000) |
| `DEMO_ACCESS_KEY` | 외부(터널) 접속 시 요구할 키. 안 정해주면 실행할 때마다 자동 생성돼서 콘솔에 출력됨 |
| `DISABLE_TUNNEL=1` | Cloudflare 터널을 켜지 않고 로컬(`http://localhost:PORT`)에서만 실행 |

실시간 카메라·사진·영상 분석 전부 `webapp/backend/services/vision_bridge.py`를
거쳐 `src/classroom_locator`의 동일한 위치 판정 로직(`LandmarkGridPositionTracker`)을
씁니다 - 즉 위치 데이터(`configs/locations.yaml`)와 판정 규칙은 CLI
(`run_realtime.py --grid-map`)와 웹 UI가 완전히 동일합니다.
