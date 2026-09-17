# classroom-locator

YOLO(객체 탐지) + OCR(문자 인식)을 이용해 강의실 표지판 사진을 분석하고,
인식된 표지판 텍스트를 미리 정의한 위치 정보와 매칭하여 **현재 위치를 추적**하는 프로젝트입니다.

- 웹캠으로 실시간 프레임을 받아 처리하는 모드
- 미리 촬영한 사진 폴더를 배치로 처리하는 모드

두 가지를 모두 지원하도록 구성했고, YOLO/OCR 라이브러리는 아직 미정이어도
바로 개발을 시작할 수 있도록 **추상 인터페이스 + 어댑터(구현체) 교체 구조**로 설계했습니다.

## 폴더 구조

```
classroom-locator/
├── .vscode/                 # VSCode 실행/설정 (launch.json, settings.json 등)
├── configs/                 # 설정 파일 (yaml)
│   ├── default.yaml         # detector/ocr backend, 임계값 등 전역 설정
│   └── locations.yaml       # "표지판 텍스트 → 위치" 매핑 데이터
├── data/
│   ├── raw/                 # 원본 사진/영상 (배치 모드 입력)
│   ├── processed/           # 전처리된 이미지
│   ├── annotations/         # YOLO 학습용 라벨(txt) 등
│   ├── dataset/             # Roboflow에서 export한 학습용 데이터셋(train/valid/test)
│   └── eval/                # 영상 성능 평가용 정답(ground truth) 타임라인 CSV
├── models/
│   ├── yolo/                # YOLO 가중치 파일 (.pt 등)
│   └── ocr/                 # OCR 관련 모델/가중치 (필요 시)
├── notebooks/                # 데이터 탐색, 실험용 주피터 노트북
├── src/classroom_locator/    # 핵심 파이썬 패키지
│   ├── detection/            # YOLO 등 객체 탐지 모듈 (추상화)
│   ├── ocr/                  # OCR 모듈 (추상화)
│   ├── localization/         # 인식 텍스트 → 위치 매칭 로직
│   ├── pipeline/             # 실시간/배치 파이프라인 조립, 위치 잠금 로직
│   ├── utils/                 # 공용 유틸 (이미지, 로깅, 지도 시각화 등)
│   ├── evaluation.py          # 영상 기반 성능 평가(F1/혼동행렬/latency/FPS/커버리지)
│   └── config.py              # 설정 로더
├── scripts/                  # CLI 실행 스크립트 (진입점)
│   ├── run_realtime.py       # 웹캠/영상 실시간 실행 + 성능 평가 모드
│   ├── run_batch.py          # 사진 폴더 배치 실행
│   ├── train_yolo.py         # YOLO 학습 스크립트
│   ├── visualize_map.py      # locations.yaml 좌표 기반 2D 평면도 생성
│   └── evaluate.py           # 배치(사진) 결과 정확도 평가 스크립트
├── tests/                     # pytest 테스트
├── outputs/                   # 실행 결과물 (로그, 결과 이미지/JSON, 평가 리포트)
└── docs/                      # 아키텍처/설계 문서, 진행 로그(setup_log.md)
```

## 설계 포인트: 라이브러리 교체 가능한 구조

YOLO 버전(YOLOv8/YOLOv5 등)과 OCR 라이브러리(EasyOCR/PaddleOCR/Tesseract 등)를
아직 정하지 않았다는 점을 고려해서, `detection`과 `ocr` 모듈은 각각

- `base.py` : 추상 클래스(인터페이스)
- `xxx_detector.py` / `xxx_reader.py` : 실제 구현체
- `__init__.py` : `get_detector(config)` / `get_ocr_reader(config)` 팩토리 함수

로 나눠져 있습니다. `configs/default.yaml`에서 `backend` 값만 바꾸면
코드 수정 없이 다른 라이브러리로 교체할 수 있습니다.

```yaml
detector:
  backend: ultralytics   # or "yolov5"
  weights: models/yolo/best.pt

ocr:
  backend: easyocr       # or "tesseract", "paddleocr"
  lang: ["ko", "en"]
```

## 시작하기

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

```bash
python scripts/run_batch.py --input data/raw --config configs/default.yaml
python scripts/run_realtime.py --camera 0 --config configs/default.yaml
```

## 실행 스크립트 옵션

### `scripts/run_realtime.py` — 웹캠/영상 실시간 실행 + 성능 평가

```bash
python scripts/run_realtime.py --camera 0
python scripts/run_realtime.py --source data/raw/test_video.mp4
python scripts/run_realtime.py --source data/raw/test_video.mp4 --no-snapshot
python scripts/run_realtime.py --source data/raw/test_video.mp4 --eval data/eval/ground_truth_example.csv
```

| 인자 | 설명 |
|---|---|
| `--config PATH` | 설정 파일 경로 (기본 `configs/default.yaml`) |
| `--camera N` | 카메라 인덱스 (설정 파일 값 덮어쓰기) |
| `--source PATH` | 웹캠 대신 재생할 영상 파일 경로. `--camera`보다 우선함 |
| `--snapshot` / `--no-snapshot` | 위치가 바뀔 때마다 그 순간 화면을 `outputs/results/realtime_snapshots/`에 저장할지 여부 (기본 켜짐, 설정 파일의 `realtime.save_snapshots`로도 조정 가능) |
| `--eval PATH` | 성능 평가 모드. 정답 타임라인 CSV(`start_sec,end_sec,location`) 경로를 넘기면 화면 표시 대신 F1/혼동행렬, 응답한 것 중 정확도, 전환 반응 속도(Latency), 처리 속도(FPS), 커버리지를 계산해서 출력하고 `outputs/results/eval_report_*.json`으로 저장함. `--source` 필수 |

### `scripts/run_batch.py` — 사진 폴더 배치 처리

```bash
python scripts/run_batch.py --input data/raw --config configs/default.yaml
```

| 인자 | 설명 |
|---|---|
| `--input PATH` | 처리할 이미지 폴더 (하위 폴더까지 재귀적으로 읽음) |
| `--config PATH` | 설정 파일 경로 |

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

### `scripts/visualize_map.py` — 2D 평면도 생성

```bash
python scripts/visualize_map.py --highlight room3
```

| 인자 | 설명 |
|---|---|
| `--locations PATH` | 위치 데이터 파일 (기본 `configs/locations.yaml`) |
| `--highlight NAME` | 현재 위치로 빨간색 강조할 location name |
| `--output PATH` | 저장 경로 (기본 `outputs/results/map.png`) |

## 위치 매핑 데이터 만들기

`configs/locations.yaml`에 강의실 표지판에 적힌 텍스트(예: "공학관 301호")를
실제 위치 정보(건물, 층, 좌표 등)와 매핑해두면, OCR 인식 결과를 이 목록과
비교(정확/유사 매칭)해서 현재 위치를 판별합니다. 자세한 포맷은 파일 내 주석 참고.
