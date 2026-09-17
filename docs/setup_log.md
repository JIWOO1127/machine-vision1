# 개발 환경 구성 & 학습 진행 로그

이 프로젝트를 진행하면서 실제로 겪은 이슈와 해결 과정을 시간순으로 기록합니다.
팀원이 같은 환경을 구성하거나 같은 문제를 겪을 때 참고용입니다.

## 1. conda 환경 구성

- Windows 사용자 계정명에 한글(`한국전파진흥협회`)이 포함되어 있어, conda 설치 시
  기본 경로(`%USERPROFILE%\miniconda3`)가 비-ASCII 경로라서 설치가 거부됨.
  → **`C:\Users\Public\Miniconda3`** (ASCII 경로)에 설치해서 해결.
- `conda init powershell`이 사용자 프로필(`Documents\WindowsPowerShell\profile.ps1`)에
  자동으로 쓰지 못함 → 수동으로 conda hook 블록을 프로필에 추가.
- PowerShell 실행 정책이 기본값(Restricted)이라 프로필 스크립트가 자동 실행되지 않음
  → `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`로 해결
  (관리자 권한 불필요, 사용자 범위만 변경).
- Anaconda 기본 채널(`pkgs/main`, `pkgs/r`, `pkgs/msys2`) 이용약관 동의 필요
  → `conda tos accept --override-channels --channel <채널>` 로 동의 처리.

```powershell
conda create -n classroom-locator python=3.11 -y
conda activate classroom-locator
```

## 2. GPU(torch) 설치

- GPU: NVIDIA GeForce RTX 4050 Laptop (VRAM 6GB), 드라이버 CUDA 13.3 지원.
- PyPI 기본 `pip install torch`만 하면 버전이 안 맞을 수 있어서, PyTorch 공식 인덱스에서
  CUDA 12.4 빌드로 명시 설치.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

- **주의**: `pip install -r requirements.txt`(ultralytics 포함)를 실행하면 ultralytics의
  의존성 해석 과정에서 torch가 버전 핀 없는 최신 버전(예: 2.14.0)으로 자동 업그레이드되고,
  torchvision은 따라가지 못해 버전이 어긋나는 문제가 발생했음. 증상:
  ```
  RuntimeError: operator torchvision::nms does not exist
  ```
  → torch/torchvision을 다시 **서로 호환되는 버전 쌍**으로 재설치해서 해결:
  ```powershell
  pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
  ```
  **교훈**: `requirements.txt`를 pip install한 뒤에는 항상 torch/torchvision 버전이
  깨지지 않았는지 확인 후 학습을 실행할 것. 확인 방법은 아래 "환경 점검" 참고.

## 3. 나머지 패키지 설치

```powershell
pip install -r requirements.txt
```

현재 `requirements.txt`에서 활성화된 백엔드:
- YOLO: `ultralytics`
- OCR: `easyocr`

## 4. 라벨링 툴

- **LabelImg**: PyQt5 최신 버전과 호환이 깨져 실행 자체가 안 됨
  (`TypeError: drawLine()... unexpected type 'float'`, 유지보수 중단된 프로젝트라 발생하는
  고질적 버그). 사용하지 않기로 함.
- **Label Studio**: 로컬 설치/실행이 안 됨 (원인 미상, 재시도 안 함).
- **최종 선택: Roboflow** (웹 기반, 설치 불필요). Object Detection(Bounding Box) 프로젝트로
  생성, 클래스는 `room2`, `room3`, `room4`로 라벨링 중 (복도 등 표지판이 명확하지 않은 경우는
  별도 클래스 없이 라벨을 아예 달지 않는 방식으로 처리).
- Export 포맷은 **YOLOv8** 사용 (`ultralytics` 라이브러리 및 `scripts/train_yolo.py`와 호환).

## 5. 데이터셋 배치

Roboflow에서 받은 export(zip 압축 해제한 폴더)는 [data/dataset/](../data/dataset/)에
통째로 넣습니다. Roboflow가 만들어주는 `data.yaml`은 그대로 쓰지 못하는 경우가 있어
아래처럼 수정해서 사용 중:

```yaml
train: train/images
val: train/images   # 데이터가 매우 적을 때 임시 조치. 데이터가 늘어나면 valid/images로 교체
# test: test/images
```

**현재 상태(2026-09-15 기준)**: `room2`, `room4` 클래스만 라벨링 완료, 사진 6장뿐.
→ 학습 결과 성능 수치는 무의미하며, 파이프라인이 정상 동작하는지 확인하는
스모크 테스트 용도로만 사용 중. `room3` 라벨링 및 사진 추가 확보 예정.

### v3 업데이트 (같은 날, room3 추가 + 사진 30장으로 확장)

Roboflow에서 `room3` 라벨링을 마치고 사진을 30장으로 늘려 재 export (`labelimg.v3i.yolov8`).
이번에도 valid/test 폴더 없이 30장 전부 train으로만 내려받아져서, `val`은 여전히
`train/images`와 동일하게 임시 설정. `data/dataset/`를 통째로 교체.

## 6. 학습 실행

```powershell
conda activate classroom-locator
cd classroom-locator
python scripts/train_yolo.py --data data/dataset/data.yaml --model yolov8n.pt --epochs 50
```

결과는 `outputs/results/sign_detector/weights/best.pt`에 저장될 것으로 예상했지만,
실제로는 ultralytics(8.4.152 기준)가 `--project outputs/results`를 `runs/detect/` 아래에
중첩시켜서 아래 경로에 저장됨:
```
runs/detect/outputs/results/sign_detector-N/weights/best.pt
```
(N은 실행할 때마다 자동 증가) 경로가 예상과 다르니 학습 후 `find`나 탐색기로 실제
`weights/best.pt` 위치를 확인할 것.

### 첫 스모크 테스트 결과 (2026-09-15)

- 데이터: room2 6장 전부, room4 4장 (val=train 그대로, 총 6장 사진에 bbox 10개)
- 50 epoch, GPU(RTX 4050) 정상 사용, 에러 없이 완료 → **파이프라인 자체는 검증됨**
- 성능 수치는 무의미함 (val=train, 사진 극소수):
  - room2: P=0, R=0, mAP50=0 (전혀 학습 안 됨)
  - room4: P=0.003, R=0.75, mAP50=0.745
- 이 가중치는 실사용하지 않고 `models/yolo/`로 옮기지 않음. room3 라벨링 +
  사진 추가 확보 후 재학습 필요.

### 두 번째 학습 결과 (v3 데이터셋, 30장/3클래스, 2026-09-15)

`runs/detect/outputs/results/sign_detector-4/weights/best.pt`, 100 epoch, GPU 사용.

```
all:   P=0.871  R=1.0   mAP50=0.977  mAP50-95=0.798
room2: P=0.853  R=1.0   mAP50=0.988
room3: P=0.840  R=1.0   mAP50=0.948
room4: P=0.918  R=1.0   mAP50=0.995
```

수치는 크게 좋아졌지만 **여전히 val=train(동일한 30장)이라 이건 "학습 데이터를 얼마나
잘 외웠는지"에 가깝고, 처음 보는 사진에 대한 진짜 성능이 아님**을 유의할 것. 사진이
더 늘어나면 Roboflow에서 train/valid를 실제로 분리해서(예: 24/6) 재 export해야
신뢰할 수 있는 수치가 나옴. 다음 확인은 학습에 쓰지 않은 새 사진으로 `yolo predict`
돌려보는 것을 추천.

## 환경 점검용 스니펫

torch/torchvision 버전이 꼬였는지 확인할 때 사용:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__)"
```

## 7. 복도 지도 시각화

room2/room3/room4가 일자 복도(4-3-2 순서, 총 16m)에 배치된다는 것을 반영해
[configs/locations.yaml](../configs/locations.yaml)의 좌표를 갱신하고, 이를 그림으로
확인할 수 있는 도구를 추가함:

- `src/classroom_locator/utils/map_viz.py` : 좌표 기반으로 복도 그림을 그리는 함수
- `scripts/visualize_map.py --highlight room3` : CLI로 지도 PNG 생성
  (`outputs/results/map.png`에 저장, `--highlight`로 넘긴 위치를 빨간색으로 강조)

**현재 한계**: 표지판이 실제로 인식된 지점(room2/3/4)만 강조 가능. 표지판 사이
중간 지점은 아직 표현 못 함 — 두 표지판이 동시에 보일 때 bbox 크기 비율로
보간(interpolation)하는 기능을 추후 추가 예정 (여러 표지판이 한 프레임에 잡히는
직선 복도 특성을 활용).

## 8. 영상 재생 테스트 + OCR 제거 (2026-09-15)

`scripts/run_realtime.py`에 `--source <영상경로>` 옵션을 추가해서, 웹캠 대신
녹화된 영상 파일로도 파이프라인을 테스트할 수 있게 함 (`cv2.VideoCapture`가
카메라 인덱스든 파일 경로든 그대로 받음).

바탕화면에 있던 테스트 영상으로 1차 실행했을 때 두 가지 문제 발견:

1. **OCR이 CPU로 실행됨**: `easyocr_reader.py`에 `gpu=False`가 하드코딩되어
   있었음 → `torch.cuda.is_available()`로 자동 감지하도록 수정.
2. **더 근본적인 문제**: room2/room3 표지판에 박스가 쳐져도 화면 위쪽 "현재 위치"
   텍스트가 계속 room4로 고정되는 버그 발생. 원인은 `core.py`가 여전히
   "탐지 → OCR로 글자 읽기 → locations.yaml과 텍스트 유사도 매칭" 방식으로 동작하고
   있었기 때문. YOLO를 이미 room2/room3/room4로 직접 분류하도록 학습해놨는데
   OCR을 또 거치면서, OCR이 실패/저신뢰 판정하면 `result.match`가 None이 되어
   이전 매칭 결과(room4)가 화면에 그대로 남아있던 것.

   **해결**: `core.py`를 재작성해서 OCR을 완전히 빼고, 탐지된 YOLO 클래스 이름으로
   `locations.yaml`을 바로 조회하도록 변경. 한 프레임에 여러 표지판이 동시에
   보이면(직선 복도 특성상 흔함) **bbox 면적이 가장 큰(=가장 가까운) 표지판**을
   현재 위치로 선택 (이전에 논의했던 "여러 강의실이 동시에 보일 때" 해결 방식을
   그대로 구현). `batch_pipeline.py`의 결과 json에서도 `ocr_texts` 필드 제거.

   OCR 관련 모듈(`ocr/`, `SignMatcher`)은 삭제하지 않고 그대로 남겨둠 —
   나중에 학습 안 된 새 강의실까지 일반화하려면 다시 필요할 수 있음.

3. **화면 잘림**: 폰 세로 영상 해상도가 그대로 `cv2.imshow`에 들어가서 창이
   모니터 밖으로 넘침 → `realtime_pipeline.py`에 `_resize_for_display()` 추가,
   가로/세로 중 큰 쪽이 900px 넘으면 비율 유지하며 축소 후 표시 (탐지 자체는
   원본 해상도로 수행하므로 정확도엔 영향 없음).

## 9. OCR 기반 방식으로 전체 롤백 (2026-09-15)

room2/room3/room4 개별 클래스 방식(mAP50 0.977)을 버리고, 원래 설계(YOLO
단일 `sign` 클래스 + OCR 텍스트 매칭)로 완전히 되돌림. 학습 안 된 새 강의실도
재학습 없이 `locations.yaml`에 항목만 추가하면 인식 가능하게 하기 위함 — 단,
OCR 오인식/저조도 조건에 성능이 좌우될 수 있음을 감수하는 결정.

**Roboflow에서 다시 라벨링하지 않고, 이미 그려둔 bbox는 그대로 두고 클래스
번호만 일괄 변경**하는 방식으로 처리함 (재작업 최소화):

```bash
# data/dataset/train/labels/*.txt 의 각 줄 맨 앞 클래스 번호(0/1/2)를 모두 0으로 치환
sed -i 's/^[0-9]\+ /0 /' *.txt
```

`data/dataset/data.yaml`도 `nc: 1, names: ['sign']`로 수정 후 재학습.

**주의**: 이 변경은 로컬 `data/dataset/`에만 반영됨. **Roboflow 프로젝트 자체는
여전히 room2/room3/room4 클래스로 남아있으므로**, 나중에 Roboflow에서 다시
export하면 이 병합이 사라지고 3클래스로 되돌아감. 계속 OCR 방식을 쓸 거라면
Roboflow 프로젝트의 클래스 자체를 `sign` 하나로 정리하는 걸 고려할 것.

되돌린 코드:
- `core.py`: OCR reader + `SignMatcher` 기반 매칭으로 복원 (bbox 크기 기반
  직접 조회 로직 제거)
- `batch_pipeline.py`: 결과 json에 `ocr_texts` 필드 복원
- `configs/default.yaml`: `target_classes: ["sign"]`로 복원
- `realtime_pipeline.py`의 3초 위치 잠금 로직은 매칭 방식과 무관하게 그대로
  유지됨 (OCR 매칭이 간헐적으로 실패해도 화면이 덜 흔들리는 효과는 유지)

**재학습 중 문제**: 첫 시도에서 `RuntimeError: CUDA error: device-side assert
triggered` 발생. 원인은 `data/dataset/train/labels.cache`가 이전(3클래스) 학습
때 캐시된 상태로 남아있었던 것 — 라벨 txt는 고쳤지만 캐시가 갱신 안 돼서
클래스 개수 불일치로 GPU에서 assert 발생. `labels.cache` 삭제 후 재시도하니
정상 진행됨. **교훈**: `data/dataset/` 라벨을 스크립트로 직접 수정한 뒤에는
`train/labels.cache`(및 `valid/labels.cache`)를 지우고 학습할 것.

**결과** (`sign_detector-6`, 100 epoch, 단일 sign 클래스):
```
P=0.996  R=1.0  mAP50=0.995  mAP50-95=0.803
```
가중치를 `models/yolo/best.pt`로 교체 완료. (여전히 val=train 30장 그대로라
수치 자체는 낙관적임 — 실제 신뢰도는 새 영상으로 다시 확인 필요)

## 9-1. "room4만 계속 인식되는" 버그 원인 규명 (2026-09-15)

OCR 기반으로 되돌린 뒤 영상 테스트를 반복했는데, room2/room3 표지판이 화면에
분명히 잡히는데도 "현재 위치"가 계속 room4에 고정되는 문제가 재현됨. 처음엔
"OCR이 글자를 아예 못 읽는다"고 추정했지만, 실제로는 **세 가지 문제가 겹친
것**으로 확인됨.

**조사 과정에서 먼저 발견한 부수 버그**: 원인 파악을 위해 OCR 원본 인식 결과를
DEBUG 로그로 남기도록 코드를 추가했는데, `configs/default.yaml`의
`logging.level: DEBUG`로 바꿔도 로그가 전혀 안 찍혔음. 원인은
`realtime_pipeline.py`/`batch_pipeline.py`가 `setup_logger()`를 호출할 때
`log_dir`만 넘기고 `level`은 안 넘겨서, 설정 파일 값이 무시되고 항상 기본값
INFO로 고정되어 있었던 것. 두 파일 모두 `level=log_config.get("level", "INFO")`를
넘기도록 수정해서 해결.

**로그가 보이게 된 뒤 확인한 진짜 원인 3가지:**

1. **EasyOCR이 "4강의실"을 "4"와 "강의실"처럼 별개 조각으로 쪼개서 인식.**
   기존 코드는 이 조각들을 각각 따로 `locations.yaml`과 비교했는데, 숫자만
   ("2", "3", "4") 놓고 보면 "2강의실"과 유사도가 40%대라 임계값(70)을 못
   넘고, 반대로 "강의실"이라는 단어만 떼서 비교하면 room2/room3/room4 **셋
   다 유사도 86%로 동일하게 높게 나옴** (전부 "OO강의실"이라 단어만으론 구분
   불가). 동점일 때 코드가 `score > best.score`(엄격한 부등호)로 비교해서
   먼저 나온 후보를 유지하는데, `locations.yaml`에 room4가 첫 번째라
   **"강의실"만 읽힐 때마다 항상 room4로 매칭**되고 있었음.
2. **표지판에 영어 부제("Meeting Room", "Class Room" 등)도 같이 있어서**,
   조각들을 단순 공백으로 합칠 때 이 노이즈가 섞여 유사도를 희석시킴.
3. **OCR이 "실"을 "신"으로 오인식**하는 경우가 잦아서(폰트/블러 영향으로
   추정), 기존 임계값 70은 실제 노이즈를 감안하면 너무 엄격했음.

**해결한 방법 4가지** (모두 `core.py`/`easyocr_reader.py`/`configs/default.yaml`
에 반영됨):

1. `easyocr_reader.py`가 반환하는 텍스트 조각을 bbox의 왼쪽 x좌표 기준으로
   정렬해서 반환 → 합칠 때 올바른 읽기 순서(왼쪽→오른쪽) 보장
2. 한글 또는 숫자가 하나라도 포함된 조각만 남기고, 영어로만 된 조각(노이즈)은
   버림
3. 조각을 따로따로 비교하지 않고 **하나로 합쳐서("4 강의실") 통으로 한 번만
   비교** — 숫자 정보가 항상 반영되게 함
4. `match_threshold`를 70 → 55로 낮춤 (정답 위치는 합쳐진 문자열 기준 60~90%,
   오답 위치는 40%대로 나와서 20%p 이상 차이가 있어 55는 안전한 값으로 확인)

부가적으로 `upscale_for_ocr()`를 추가해서 멀리서 찍혀 작은 crop도 OCR이 더 잘
읽도록 확대 처리함.

**적용 후**: 같은 영상으로 재테스트해서 room4 → room3 → room2로 정확히,
깜빡임 없이 전환되는 것을 확인함.

**향후 개선 아이디어 (미구현)**:
- 한 위치에 대해 여러 프레임의 OCR 결과를 누적해서 다수결로 판단하면 일회성
  오인식에 덜 민감해질 수 있음
- PaddleOCR 등 다른 OCR 백엔드와 정확도 비교해볼 수 있음 (`requirements.txt`에
  이미 옵션으로 있음)
- 표지판이 기울어져 찍히는 경우가 많다면 crop 후 원근 보정(perspective
  correction)을 추가하면 인식률이 더 좋아질 수 있음
- 지금은 임계값을 낮춰서 우회했지만, 데이터가 더 모이면 OCR 오인식 패턴 자체를
  분석해서(예: "실"↔"신" 같은 흔한 오인식 쌍을 aliases에 미리 추가) 근본적으로
  보완할 수 있음

## 9-2. "2/3강의실 인식 후 이후 계속 4강의실로 인식되는" 잔존 버그 (2026-09-16)

9-1에서 고친 "강의실만 읽히면 room4로 쏠리는" 버그가, **숫자+단어가 같이 읽힐
때만** 고쳐져 있었고, **숫자 없이 "강의실"이라는 단어만 단독으로 읽히는
프레임**에서는 여전히 재발했음. 실제 사용 중 "2강의실/3강의실은 잘 인식되는데,
그 이후로는 전부 4강의실로 인식된다"는 형태로 나타남 — room2/room3를 지나고
나서 표지판과의 각도가 애매해지거나 숫자 부분이 흐릿해지면 "강의실"만 읽히는
경우가 잦아지고, 그때마다 `locations.yaml`에서 첫 번째로 나열된 room4로
쏠렸던 것.

**근본 원인**: [sign_matcher.py](../src/classroom_locator/localization/sign_matcher.py)의
`match()`가 동점(같은 최고 점수를 여러 위치가 공유)일 때 `score > best.score`
(엄격한 부등호)라서 먼저 발견된 후보를 그대로 유지 — 즉 "동점이면 임의로
첫 번째를 선택"하는 구조적 결함이었음. 9-1에서는 "숫자+단어 조합"으로 점수
차이를 벌려서 우회했을 뿐, 동점 자체를 처리하는 로직은 그대로였던 것.

**해결**: `match()`를 수정해서, 최고 점수가 **서로 다른 위치 사이에 동점으로
걸리면 그 프레임은 매칭 실패(`None`)로 처리**하도록 변경. 애매한 추측으로
엉뚱한 위치를 확정하기보다, 판단을 보류하고(화면은 이전 위치 유지) 다음
프레임에서 숫자까지 명확히 읽힐 때 갱신되도록 함.

```python
# 검증: "강의실"만 단독으로 들어오면 이제 None (이전엔 room4로 잘못 확정됨)
matcher.match(["강의실"])  # -> None
matcher.match(["3 강의실"])  # -> room3 (정상)
```

기존 `tests/test_localization.py` 3개 테스트도 모두 통과 확인.

## 9-3. 지도를 1D 복도 선에서 2D 평면도로 확장 (2026-09-16)

앞문/뒷문/정수기가 기존 room2/3/4 일자 복도와 같은 층이지만 같은 직선 위에
있지는 않아서("2D 평면도가 필요"), `map_viz.py`를 `coords.x`만 쓰던 1D
방식에서 `coords.x`+`coords.y`를 모두 쓰는 진짜 2D 평면도로 재작성.
`draw_corridor_map()` → `draw_floor_map()`으로 이름도 변경 (더 이상 "복도"만
그리는 게 아니라서). room4-room3-room2 구간은 여전히 알아볼 수 있게 선으로
연결하고, 나머지 위치(문/정수기 등)는 독립된 점으로만 표시.

**좌표 확정 (2026-09-16)**: `front_door: (-2, 5)`, `rear_door: (16, 5)`,
`water_dispenser: (5, 3)`로 `configs/locations.yaml`에 반영 완료. 6개 지점
전부 `scripts/visualize_map.py`로 2D 평면도에 정상 표시됨.

## 10. 글자 없는 물체(앞문/뒷문/정수기) 지원 추가 (2026-09-15)

앞문/뒷문/정수기처럼 글자가 없는 물체는 OCR로 판별할 수 없어서, 강의실
표지판(OCR 방식)과 다르게 **YOLO가 이미지 자체를 별도 클래스로 직접 분류**하도록
설계 확장 (처음엔 비상구로 검토했다가, 최종적으로 앞문/뒷문/정수기/표지판(2,3,4강의실)
4종류로 확정):

- YOLO 클래스가 `sign`이면 → 기존처럼 OCR로 글자 읽어서 위치 판별 (2/3/4강의실)
- YOLO 클래스가 그 외(`front_door`/`back_door`/`water_dispenser`)면 → 클래스
  이름이 곧 `locations.yaml`의 위치 이름이므로 OCR 없이 바로 조회 (room2/3/4를
  직접 분류했던 예전 방식과 동일한 원리를, sign 클래스와 공존하는 형태로 재도입)
- 여러 물체가 동시에 보이면(강의실 표지판 + 앞문 등 섞여도) bbox가 가장
  큰(가까운) 것을 현재 위치로 선택 — 기존 로직을 확장해서 재사용

**변경 파일**: `core.py`(`OCR_CLASS_NAME` 상수로 분기, 후보를 모아서 bbox
크기로 최종 선택하도록 재작성), `configs/locations.yaml`(`front_door`/
`back_door`/`water_dispenser` 항목 추가, `coords`/`guidance`는 아직 실측 전이라
`null`), `configs/default.yaml`(`target_classes: ["sign", "front_door",
"back_door", "water_dispenser"]`).

**아직 안 된 것**: 이 3개 클래스로 학습된 가중치가 없음 — 앞문/뒷문/정수기
사진을 찍어서 Roboflow에 각각 새 클래스로 라벨링 → export → `data/dataset/`에
반영 → 재학습해야 실제로 탐지됨. 그리고 `locations.yaml`의 각 위치 좌표(`coords`)와
안내 문구(`guidance`)도 실측 후 채워야 지도/안내 메시지에 반영됨.

## 11. 새 통합 데이터셋으로 교체 (machine_vision 2.v1i.yolov8, 2026-09-15)

다운로드 폴더의 `machine_vision 2.v1i.yolov8`에 227장(train 161 / valid 45 /
test 21, 처음으로 제대로 된 train/valid 분리가 있는 export)짜리 라벨링된
데이터가 있었음. 원래 클래스는 `['2_class', '4_class', 'front_door',
'rear_door', 'water_dispenser']` (5개, 2강의실/4강의실을 각각 별도 클래스로
라벨링한 상태)였는데, 지금 아키텍처(강의실 표지판은 `sign` 단일 클래스로
탐지 후 OCR로 판별)에 맞게 `2_class`/`4_class`를 `sign` 하나로 병합.

**클래스 인덱스 재매핑** (bbox 좌표는 그대로 두고 맨 앞 클래스 번호만 변경):
```
0(2_class) -> 0(sign)
1(4_class) -> 0(sign)
2(front_door)     -> 1(front_door)
3(rear_door)      -> 2(rear_door)
4(water_dispenser)-> 3(water_dispenser)
```
`sed`로 train/valid/test의 모든 라벨 txt 파일에 일괄 적용. `data/dataset/`
전체를 이 새 데이터셋으로 교체하고 `data.yaml`도 `nc: 4,
names: ['sign', 'front_door', 'rear_door', 'water_dispenser']`로 수정.

**참고**: 원래 프로젝트에서 "뒷문"을 `back_door`로 이름 지었었는데, 이 export의
Roboflow 프로젝트에서는 `rear_door`로 되어 있어서, 매번 이름을 바꿔야 하는
수고를 덜기 위해 **우리 쪽 설정(`configs/locations.yaml`,
`configs/default.yaml`)을 `rear_door`에 맞춰 변경**함 (반대로 하지 않음 —
Roboflow 쪽 클래스명이 앞으로도 계속 이걸로 나올 것이므로).

**데이터 불균형 주의**: 라벨 개수가 `sign` 86장, `front_door` 50장,
`rear_door` 64장인데 비해 **`water_dispenser`는 전체 3장뿐**(train 2, valid 1,
test 0). 이대로면 정수기 탐지 성능이 낮을 가능성이 높음 — 정수기 사진을
추가로 더 찍어서 보강하는 걸 권장.

### 학습 결과 (`sign_detector-7`, 100 epoch, 2026-09-15)

**드디어 진짜 train/valid 분리로 학습한 첫 결과** (val=train이 아니라서 이번
수치는 신뢰할 수 있음):
```
전체:            P=0.919  R=0.965  mAP50=0.995  mAP50-95=0.842
sign:            mAP50=0.995
front_door:      mAP50=0.995
rear_door:       mAP50=0.995
water_dispenser: mAP50=0.995 (단, valid에 1장뿐이라 통계적으로 큰 의미는 없음)
```
가중치 `models/yolo/best.pt`로 교체 완료. `water_dispenser`는 데이터가 워낙
적어서(3장) 이 점수를 그대로 믿기보다 사진을 더 모아서 검증할 필요 있음.

## 12. 영상 기반 성능 평가 기능 추가 (2026-09-16)

같은 데이터셋으로 YOLO / YOLO+OCR / VPR / ArUco 등 여러 방식을 비교하기 위해,
영상 + 정답 타임라인으로 5개 지표를 자동 계산하는 기능 추가:
1. 클래스별 F1 / 혼동행렬
2. 응답한 것 중 정확도
3. 전환 반응 속도(Latency) — 새 구간 진입 후 정답으로 갱신될 때까지 걸린 시간
4. 처리 속도(FPS/추론 시간)
5. 커버리지(응답률)

**새로 만든 파일:**
- `src/classroom_locator/pipeline/locking.py`: 위치 잠금(3초 유지) 로직을
  `LocationLocker` 클래스로 분리 — `realtime_pipeline.py`(화면 표시)와
  `evaluation.py`(평가)가 완전히 같은 규칙으로 동작하게 하기 위함 (따로
  구현하면 평가 결과가 실제 화면 동작과 미묘하게 달라질 위험이 있어서 공용화함)
- `src/classroom_locator/evaluation.py`: 정답 CSV 로더 + 영상을 처음부터
  끝까지 돌리며 지표 계산 + 리포트 출력/JSON 저장
- `data/eval/ground_truth_example.csv`: 정답 타임라인 예시 (형식:
  `start_sec,end_sec,location`, location 비우면 "전환구간"으로 채점 제외)

**CLI**: `scripts/run_realtime.py`에 `--eval <정답CSV경로>` 추가 (`--source`
필수). 지정하면 화면 표시 대신 평가를 수행하고 콘솔 출력 + `outputs/results/
eval_report_<시각>.json`으로 저장.

**설계 메모**:
- Latency는 영상 자체의 시간(frame_index/fps)을 기준으로 계산 — 평가 스크립트가
  화면 표시 없이 최대한 빠르게 프레임을 처리하기 때문에, wall-clock 시간을 쓰면
  "실제로 재생했을 때"와 다른 값이 나옴. 반면 처리 속도(FPS)는 반대로
  wall-clock(실제 연산 시간)을 씀 — 이건 실제 계산 부하를 재는 것이라 영상
  재생 속도와 무관해야 하기 때문.
- F1 계산 시 "무응답"도 하나의 예측값으로 취급해서 recall에 반영함 (애매하면
  응답을 안 하는 것도 "그 클래스를 놓친 것"으로 채점) — 커버리지/응답
  정확도로 "왜 recall이 낮은지"(오답이 많아서 vs 응답을 안 해서)를 따로
  구분해서 볼 수 있게 함.

**첫 실행 테스트**: 기존 테스트 영상(19.1초)으로 예시 정답 CSV를 만들어 실행
→ 기능 자체는 정상 동작 확인. 단, 예시 CSV의 타이밍은 실제 영상을 보지 않고
이전 로그 기반으로 대충 추정한 것이라 room4 구간이 실제와 안 맞아 F1=0으로
나옴 — **실제 평가하려면 영상을 보면서 CSV 타이밍을 직접 맞춰야 함.**

## 13. requirements.txt / README 실제 환경 기준으로 정리 (2026-09-16)

`requirements.txt`를 범위 지정(`>=`)에서 **실제 conda 환경(`classroom-locator`)에
설치된 정확한 버전으로 고정**함 (`pip freeze` 기준). torch/torchvision은
CPU 빌드를 기본으로 두고, GPU 있는 PC는 설치 후 CUDA 빌드로 덮어쓰라는 명령을
주석으로 안내 (이전에 겪었던 "ultralytics 설치 시 torch가 핀 없이 최신으로
자동 업그레이드되며 torchvision과 버전이 어긋나는 문제"를 애초에 방지하기 위함).

`README.md`에 conda 환경 이름(`classroom-locator`) 명시하고, 지금까지 추가된
모든 스크립트 인자(`run_realtime.py`의 `--source`/`--snapshot`/`--eval` 등,
`run_batch.py`, `train_yolo.py`, `visualize_map.py`)를 표로 정리해서 추가함.
폴더 구조 트리에도 `data/dataset/`, `data/eval/`, `evaluation.py` 등 최근
추가된 것들을 반영.

## 14. YOLO 직접분류 vs OCR 성능 비교 실험 (2026-09-16)

`test sample.v1i.yolov8`(100장, 111개 물체) 테스트셋으로 평가해보니 OCR
경로(room2/room4)가 표지판이 멀리서 찍힌 사진에서 recall이 매우 낮음을 확인
(room2 18.5%, room4 8.0%, `--per-object` 모드 기준). 원인 진단(`_diagnose_miss.py`
스크립트, 이후 삭제) 결과 OCR이 아예 글자 영역을 못 찾는 경우가 대부분이었고,
표지판 bbox 크기를 비교해보니 학습 데이터(평균 21.8%) 대비 이 테스트셋
(평균 6.0%)이 훨씬 멀리서 찍힌 사진들이라는 게 원인으로 확인됨.

**"OCR을 끄고 YOLO 직접분류만으로 하면 얼마나 나아지는지" 비교 실험**을
진행함 — room2/room4를 `sign`으로 합치지 않고 원래 클래스(`2_class`/`4_class`)
그대로 살려서 별도 데이터셋(`data/dataset_yolo_only/`, machine_vision 2 export의
원본 라벨 그대로, room3는 없음)으로 재학습.

**결과 비교** (`--per-object` 모드, 동일 테스트셋):
```
              OCR 방식 recall   YOLO 직접분류 recall
room2              0.185              0.481
room4              0.080              0.760
front_door         0.667              0.741
rear_door          0.938              0.875
Macro F1           0.557              0.821
```
YOLO 직접분류가 특히 room4에서 압도적으로 좋음 (글자를 못 읽어도 표지판
생김새만으로 구분 가능하기 때문). 오분류(확신 있게 틀림)도 OCR 방식보다 적음
(room4->room2 1건 외엔 전부 무응답).

**현재 상태 (둘 다 유지, 비교만 하고 전환은 안 함):**
- 기존 OCR 방식: `models/yolo/best.pt` + `configs/default.yaml` (기본, 안 건드림)
- YOLO 직접분류 실험: `models/yolo/best_yolo_only.pt` + `configs/yolo_only.yaml` (신규)
  - `core.py`의 `OCR_CLASS_NAME = "sign"` 분기 덕분에, 클래스 이름만 room2/room4로
    바꿔서 학습하면 코드 수정 없이 OCR을 자동으로 건너뜀 (target_classes만 다르게
    설정하면 됨)
- **주의**: `data/dataset_yolo_only`엔 room3 라벨이 없어서 이 모델은 room3를
  전혀 인식하지 못함. room3까지 비교하려면 room3 사진을 이 데이터셋에도
  추가해서 재학습 필요.

## 15. final_best.pt 추가 비교 + evaluate_test_sample.py에 --raw-detector 모드 추가 (2026-09-16)

사용자가 `models/yolo/final_best.pt`(YOLO 단독, 클래스명이 `2_class`/`4_class`
그대로인 모델)를 추가로 가져와서 비교 요청. 클래스 이름이 `locations.yaml`
기준(room2/room4)과 달라서 기존 `pipeline.process_image()` 경로로는 위치
조회가 안 됨 → `evaluate_test_sample.py`에 **`--raw-detector`** 모드 추가:
OCR/위치 매칭(`core.py`)을 거치지 않고 `pipeline.detector.detect()`의 raw
클래스 이름을 `DEFAULT_CLASS_MAP`으로 이름만 맞춰서 바로 채점. 전용 설정
`configs/final_best.yaml`도 생성.

**3개 모델 비교 결과 (test_sample, `--per-object` 모드, Macro F1):**
```
YOLO+OCR (best.pt)                : 0.557  (room2 R=0.185, room4 R=0.080)
YOLO 단독 v1 (best_yolo_only.pt)   : 0.821  (room2 R=0.481, room4 R=0.760)
YOLO 단독 v2 (final_best.pt)       : 0.850  (room2 R=0.593, room4 R=0.800)  <- 현재 가장 좋음
```
세 모델/설정 다 남겨두고 **비교만 계속하는 것으로 결정** (기본값 전환은 안 함).
`configs/default.yaml`(OCR), `configs/yolo_only.yaml`(v1),
`configs/final_best.yaml`(v2)로 언제든 다시 비교 가능.

## 16. distance_data 캘리브레이션 + machine-vision1 위치추정 로직 이식 (2026-09-17)

**OCR 완전 배제 확정**: `configs/default.yaml`을 YOLO 단독 방식으로 전환.
`LocatorPipeline.__init__`도 `target_classes`에 `"sign"`이 없으면 OCR
리더/매처를 아예 초기화하지 않도록 수정 (`core.py`) — 시작 시간이
2.65초로 단축됨(EasyOCR 로딩 생략).

**distance_data 캘리브레이션**: 카카오톡으로 받은 `distance_data`
폴더(`front_*.jpg`/`rear_*.jpg`/`sign_*.jpg`, 파일명이 실제 거리(m))로
`scripts/calibrate_distance.py` 작성 — YOLO로 bbox를 뽑아 "크기 비율(bbox
변길이/프레임 변길이) <-> 실제 거리" 대응표를 만듦
(`data/eval/distance_calibration.csv`). OpenCV가 한글 중첩 경로를 못 읽는
문제가 있어 `cv2.imdecode(np.fromfile(...))`로 우회. `sign` 클래스는 원본
사진 해상도(4284x5712)가 학습 이미지보다 훨씬 커서 리사이즈 시 작아져
conf_threshold(0.5)에 안 걸림 — 캘리브레이션용으로만 conf 0.1로 낮춰서
추출(원거리 3.6m 이상은 신뢰도가 노이즈 수준<1%까지 떨어져 진짜 탐지 불가
확인, front 3개/rear 5개/sign 3개 = 총 11개 데이터 확보).

**machine-vision1 프로젝트 위치추정 로직 이식**: 사용자가 공유한
`github.com/JIWOO1127/machine-vision1`(같은 조 다른 팀원의 프로젝트로 추정,
웹서버+ArUco+Faster R-CNN/YOLO+PaddleOCR 혼합 구조)의 `VisualLocationEstimator`
(`backend/services/visual_localization.py`)를 분석 후 이식:
- 핀홀 카메라 모델로 bbox 크기 → 거리 추정 (원본은 실측 물리 크기(mm)+화각
  필요 → 우리는 `distance_data` 경험적 캘리브레이션으로 대체, 실측 불필요)
- bbox 좌우 위치 → 방향각(bearing) 추정
- 지도를 격자로 나눠 각 격자점의 "예상 거리/각도 vs 실측치" 오차를 비용으로
  계산해서 최소 비용 지점을 현재 위치로 선택 (랜드마크 1개는 "후보"만,
  2개 이상 동시 확정 시 "확정(localized)")
- 이전 프레임 위치와의 근접성을 사전확률처럼 반영해 안정화
- v1 단순화: surface-visibility(정면에서만 보임) 각도 제약, 명시적 zone
  정의, optical flow는 이식 안 함 (필요시 추가 가능)

**새 파일**: `src/classroom_locator/localization/visual_localization.py`
(`VisualLocationEstimator`, `LocalizationResult`), `data/distance_calibration.json`
(클래스별 k_height/k_width), `scripts/calibrate_distance.py`,
`scripts/estimate_position.py` (사진 한 장으로 테스트).

**map_viz.py 확장**: `draw_floor_map(..., estimated_position=(x,y))`로
임의 연속 좌표를 별 마커로 표시 가능해짐 (기존 `highlight`는 정해진
위치 이름만 강조 가능했음).

**detector class_name_map 추가**: 학습 데이터셋마다 클래스명이 제각각이라
(`2_class`/`4_class` vs `room2`/`room4`) `YoloDetector`에
`class_name_map` 설정을 추가해서 원본 클래스명을 locations.yaml 기준
이름으로 바꿔치기 (`detection/yolo_detector.py`, `detection/__init__.py`).

**모델 재선정**: `configs/default.yaml`을 `best_yolo_only.pt`에서
`final_best.pt`로 교체 (test_sample 평가에서 더 좋았음, 15번 항목 참고).
`class_name_map`으로 `2_class`→`room2`, `4_class`→`room4` 매핑.

**검증**: 표지판+뒷문이 동시에 찍힌 샘플 사진으로 단일 랜드마크(뒷문만
인식) 테스트 → "candidates" 상태로 추정 좌표(10.5, 3.0) 산출, 지도에
별 마커로 정상 표시 확인. **2개 이상 랜드마크 동시 확정("localized") 케이스는
아직 실제 데이터로 검증 못함** — 다음에 여러 물체가 동시에 잘 잡히는
사진/영상으로 추가 검증 필요.

## 17. 격자(grid) 위치 추적 + 실시간 표시 창 추가 (2026-09-17)

**새 요구사항**: 위치 추정을 VisualLocationEstimator(연속 좌표+삼각측량)보다
단순하게, "가로 14 x 세로 4칸(한 칸 1.8m) 격자 + 좌회전/우회전만 있는 고정
시연 경로" 전제로 구현. 랜드마크별로 "이 거리(snap_distance_m) 이내면 이
격자점 근처"라는 규칙.

**만든 것**:
- `Location`에 `grid_cell`(격자 교차점 좌표, 칸 번호 아님), `snap_distance_m`
  필드 추가 (`location_map.py`), `locations.yaml`에 `grid:` 섹션(cols/rows/
  cell_size_m) 추가
- `GridPositionTracker`(`pipeline/grid_tracker.py`): 임계값 이내로 탐지되면,
  "계산된 실제 거리와 가장 가까운 격자 교차점"을 전체 격자점 중에서 찾아
  현재 위치로 갱신 (랜드마크 좌표로 순간이동이 아니라 거리에 따라 점점
  가까워지는 식). 여러 랜드마크 동시 조건 만족 시 distance/snap_distance_m
  비율이 가장 작은 것 우선
- `render_grid_image()`: 격자를 OpenCV 이미지로 그림. **왼쪽 아래가 원점**이
  되도록 y축을 뒤집어서 그림 (이미지 좌표계는 원래 위가 원점이라 변환 필요)
- `realtime_pipeline.py`에 통합: 처리되는 프레임마다 격자 위치 갱신하고
  별도 창("classroom-locator - 격자 위치")에 실시간 표시
- `run_realtime.py`에 `--grid-map`/`--no-grid-map` 플래그 추가
  (`realtime.show_grid_map` 설정과 연동, 스냅샷 플래그와 동일한 패턴)

**한글 렌더링 버그 발견 및 수정**: 기존 메인 화면의 안내 문구가
`cv2.putText`로 그려지고 있었는데, **OpenCV의 `putText`는 한글을 지원하지
않아서 계속 깨진 글자로 표시되고 있었을 가능성이 높음**. `utils/image_utils.py`에
`put_korean_text()`(PIL 기반, machine-vision1 프로젝트의 `draw_location_overlay`
방식과 동일)를 만들어서 그리드 창과 메인 화면 안내 문구 둘 다 이걸로 교체.

**좌표 확정** (사용자가 지정): 왼쪽 아래 원점 기준
- 앞문(front_door): (0, 4), 임계값 3.0m
- 4강의실(room4): (3, 0), 임계값 2.5m
- 2강의실(room2): (14, 0), 임계값 2.5m (앞문/4강의실과 같은 "강의실" 규칙 적용)
- 뒷문(rear_door): (14, 4), 임계값 3.0m (앞문과 동일 — 사용자 확인 완료)
- room3는 이번 시연 경로에 없어서 grid_cell 미정 (TODO로 남겨둠)

**검증**: `_nearest_point_by_distance()`를 각 랜드마크의 실제 임계값 범위
내에서 테스트 → 거리가 늘어날수록 랜드마크에서 자연스럽게 멀어지는 격자점을
정확히 찾음 확인. 다만 임계값보다 훨씬 큰 거리를 강제로 넣으면(비현실적
범위) 전혀 엉뚱한 방향의 격자점으로 튈 수 있음을 확인 — 실사용 범위(항상
snap_distance_m 이내에서만 갱신됨)에서는 문제 없음.

`testvideo.mp4`(바탕화면)로 전체 통합 테스트 → 앞문→4강의실→뒷문 순서로
정상 전환, 에러 없이 끝까지 재생 확인.

**경로 제약 추가**: "가장 가까운 격자점"을 전체 15x5 격자가 아니라
**x축(y=0) + y축(x=0) + x=cols축(오른쪽 끝)** 이 세 직선(L자/U자형 경로,
점 23개)으로만 제한하도록 `GridPositionTracker.__init__` 수정. 이전엔
임계값을 벗어난 큰 거리를 강제로 넣으면 경로에서 벗어난 엉뚱한 격자점으로
튀는 걸 확인했었는데(예: room4 기준 target=5 -> (0,4)), 이제는 항상 경로
위의 점으로만 나옴 (23개 점 모두로 검증 완료).

## 18. seojiwoo_core 외부 코드 이식 (2026-09-17)

`github.com/JIWOO1127/machine-vision1`의 `feature/seojiwoo-detection` 브랜치,
`seojiwoo/core` 폴더 전체를 `external/seojiwoo_core/`에 이식함 (같은 조
팀원 서지우님 작업물). 우리 패키지(`src/classroom_locator/`)와 임포트
구조가 달라서 통합하지 않고 독립 폴더로 그대로 둠 — 자세한 내용은
`external/seojiwoo_core/README.md` 참고.

**이식한 파일**: `locator.py`(거리추정+프레임투표 판정), `navigator.py`
(시나리오 상태머신+방향안내), `ocr_verify.py`(근접 시에만 OCR 재확인),
`demo_live.py`(오버레이+TTS 실행 스크립트), `requirements.txt`.

**중요 발견**: 우리 `models/yolo/final_best.pt`(5클래스)가 이 저장소의
`v1_best.pt`와 해시까지 완전히 동일함 — 즉 우리가 갖고 있던 게 구버전이었고,
저장소의 현재 `final_best.pt`는 **`logo` 클래스가 추가된 6클래스 최신
버전**임을 확인. 이 최신 모델을
`external/seojiwoo_core/final_best_v2_with_logo.pt`로 같이 가져옴
(Navigator의 find_logo/reach_wall 단계에 필요, 우리 기존 5클래스 모델로는
그 두 단계가 동작 안 함).

**우리 쪽에 없던 주요 기법** (README에 상세 비교 정리): bbox가 프레임 상/하단에
잘리면 높이 대신 너비로 거리 추정, 프레임 기반 다수결 투표 안정화, 접근
추세(거리 감소 추세) 확인으로 "스쳐 지나감" 오탐 방지, YOLO near 판정 순간에만
OCR로 재확인하는 하이브리드, TTS 음성 안내. 필요시 이 중 일부를 우리
`GridPositionTracker`/`visual_localization.py`에 반영할 수 있음 (아직 안 함).

## 19. seojiwoo_core 로직을 실제 프로젝트에 통합 (2026-09-17)

`external/seojiwoo_core/`는 원본 그대로 보존하고, 그 로직으로 실제 구동이
되도록 `classroom_locator` 패키지 규약(상대 임포트, `src/` sys.path 추가
방식)에 맞게 별도 위치에 옮겨 붙임.

**옮긴 위치**:
- `src/classroom_locator/seojiwoo/__init__.py` (신규) — `Locator`,
  `Navigator`, `DISPLAY`, `REAL_SIZE`, `SIGNS` 재노출
- `src/classroom_locator/seojiwoo/locator.py` — 원본 `locator.py`와 로직
  동일 (거리추정+프레임투표+approach_only+route+OCR훅)
- `src/classroom_locator/seojiwoo/navigator.py` — 원본 `navigator.py`와
  로직 동일, `from locator import ...` → `from .locator import ...`로만
  변경
- `src/classroom_locator/seojiwoo/ocr_verify.py` — 원본과 동일 (모듈간
  임포트가 없어 수정 없이 그대로)
- `models/yolo/final_best_v2_with_logo.pt` (신규 복사) — `--nav`(로고
  단계)에 필요한 6클래스 모델
- `scripts/run_seojiwoo_demo.py` (신규, 원본 `demo_live.py` 포팅) — 기본
  `--weights`는 `models/yolo/final_best.pt`, `--nav` 사용 시 `--logo_weights`
  미지정이면 `models/yolo/final_best_v2_with_logo.pt` 자동 적용. 그 외
  인자/동작은 원본과 동일
- `requirements.txt`에 `pyttsx3`(선택, `--tts` 전용) 안내 주석 추가

`classroom_locator.pipeline`(GridPositionTracker 등 우리 자체 구현)은
그대로 유지 — 이번 통합은 그것을 대체하지 않고, 서지우님 로직을 **독립된
대안 실행 경로**(`scripts/run_seojiwoo_demo.py`)로 우리 프로젝트 안에서
바로 실행 가능하게 만든 것.

**동작 확인**: conda env `classroom-locator`에서
`from classroom_locator.seojiwoo import Locator, Navigator` 임포트 성공,
`Locator('models/yolo/final_best.pt', ...)` 및
`Locator(..., logo_weights='models/yolo/final_best_v2_with_logo.pt')` 둘 다
모델 로딩 성공 확인 (6클래스 모델의 `logo` 클래스 인덱스 5 확인).

## 다음 할 일

## 20. 실시간 파이프라인 격자 지도를 seojiwoo Locator 판정으로 전환 (2026-09-17)

`external/seojiwoo_core/v1_best.pt`를 저장소에서 직접 받아 해시 재검증함 —
`models/yolo/final_best.pt`와 완전히 동일(`d47b406c...`). 새 파일을 쓸 필요
없이 기존 `final_best.pt` 그대로 쓰면 됨.

`scripts/run_realtime.py --grid-map`으로 뜨는 "격자 위치" 창의 판정 방식을,
우리 자체 `distance_calibration.json` 기반 추정에서 **seojiwoo_core의
Locator 판정 로직**으로 교체함:

- `src/classroom_locator/pipeline/grid_tracker.py`에 `SeojiwooGridPositionTracker`
  (`GridPositionTracker`를 상속) 추가. `classroom_locator.seojiwoo.Locator`를
  내부에 두고, `update_from_frame(frame, t)`가 `Locator.process()`를 그대로
  호출해서 (REAL_SIZE 실측 물리크기 기반 핀홀 거리추정 + bbox 잘림 시
  높이<->너비 전환 + 최근 5프레임 다수결 투표 + 접근추세 확인) 결과를 받고,
  "계산된 거리와 가장 가까운 격자점 찾기"(경로 제약 포함)는 부모 클래스 로직을
  그대로 재사용.
  - `reason`이 `"votes ..."`(투표 수 부족) 또는 `"not approaching"`(스쳐
    지나가는 중)이면 이전 위치를 유지(스티키)하도록 처리 — Locator의
    오탐 방지 로직을 격자 갱신 여부에도 반영.
  - 랜드마크별 근접 임계값은 Locator의 전역 `near_m`(3.0 고정)이 아니라
    기존처럼 `locations.yaml`의 랜드마크별 `snap_distance_m`(room 2.5m,
    door 3.0m)을 그대로 씀.
- `src/classroom_locator/pipeline/realtime_pipeline.py`: `distance_calibration.json`
  로딩 코드 제거, `SeojiwooGridPositionTracker(grid_config, pipeline.locations,
  weights=config["detector"]["weights"])`로 교체. 모델은 `detector.weights`
  (`models/yolo/final_best.pt` = v1_best.pt와 동일)를 그대로 재사용 — 프레임당
  우리 자체 탐지기와 별개로 Locator가 자기 모델로 한 번 더 추론함(그리드
  지도 계산용, process_every_n_frames 주기에서만 실행되어 비용은 제한적).
- `pipeline/__init__.py`에 `SeojiwooGridPositionTracker` export 추가.

**검증**: conda env에서 `SeojiwooGridPositionTracker` 생성 + 랜덤 노이즈
프레임으로 `update_from_frame()` 호출 + `render_grid_image()`까지 정상
동작 확인 (탐지 없는 프레임이라 `current_cell=None` 유지, 예상대로).

**참고**: 기존 `GridPositionTracker`(distance_calibration.json 기반)는
클래스 자체는 그대로 남겨뒀음 — `SeojiwooGridPositionTracker`가 그걸
상속해서 경로 제약/`_nearest_point_by_distance`/`render_grid_image`를
재사용하기 때문. 다만 `realtime_pipeline.py`에서 더는 직접 안 씀.

## 21. 격자 지도 판정에 ocr_verify.py 로직 연결 (2026-09-17)

`SeojiwooGridPositionTracker`가 내부에서 쓰는 `Locator`에 OCR 검증기를
아예 안 넘겨줬었는데(=Locator 자체 로직 중 OCR 훅이 죽어있었음), 이번에
`classroom_locator.seojiwoo.ocr_verify.OcrVerifier`(원본 `ocr_verify.py`와
로직 동일)를 연결함:

- `SeojiwooGridPositionTracker.__init__`에 `use_ocr: bool = True` 추가,
  기본으로 `_build_ocr_verifier()`가 `OcrVerifier`를 만들어 `Locator(...,
  verifier=...)`로 넘김. GPU 유무는 우리 `easyocr_reader.py`와 같은 패턴으로
  `torch.cuda.is_available()`로 자동 판단. easyocr 미설치 등으로 생성 실패하면
  경고 로그만 남기고 OCR 없이(YOLO 판정만으로) 계속 동작 — 필수 의존성으로
  만들지 않음.
- `update_from_frame()`의 스티키(이전 위치 유지) 조건에 `reason.startswith("OCR")`
  추가. `Locator.update()`는 near 판정 표지판을 `OcrVerifier.verify()`로
  재확인해서 글자가 다르면(reject) 또는 아직 안 읽혔으면(unknown/대조 중)
  `reason`을 `"OCR reject ..."` / `"OCR unread"` / `"OCR n/m"`으로 채우는데,
  이전엔 이 reason을 안 보고 distance_m만 보고 격자를 갱신해버려서 OCR
  검증이 있으나 마나였음. 이제는 OCR이 아직 확정 안 됐으면 격자 위치를
  갱신하지 않고 이전 위치를 유지함.

**검증**: `SeojiwooGridPositionTracker` 생성 시 `_locator.verifier`가
`OcrVerifier` 인스턴스로 채워지는 것, 랜덤 프레임으로 `update_from_frame()`이
에러 없이 도는 것 확인.

## 22. logo 클래스 + 3강의실(OCR) 격자 위치 추가 (2026-09-17)

`configs/locations.yaml`:
- `logo` 위치 신규 추가: `grid_cell: [0, 0]`, `snap_distance_m: 3.0`(문 클래스와
  같은 값으로 우선 맞춰둠 - 실측치 아님, 필요하면 조정).
- `room3`: 그동안 비워뒀던 `grid_cell`/`snap_distance_m` 채움 —
  `grid_cell: [6, 0]`, `snap_distance_m: 2.5`(room2/room4와 동일). 실제
  표지판 위치는 (7,0) 부근이지만, 사용자 지정에 따라 인식(스냅) 기준점은
  (6,0)으로 설정.

`src/classroom_locator/pipeline/grid_tracker.py` (`SeojiwooGridPositionTracker`):
- `logo_weights` 매개변수 추가. `logo` 클래스는 5클래스 `final_best.pt`엔
  없어서, `final_best_v2_with_logo.pt`(6클래스)를 Locator의 `logo_weights`로
  같이 돌려서 인식함 (`Locator`가 원래 지원하던 "로고 전용 별도 모델" 기능,
  `external/seojiwoo_core/locator.py` 참고).
- **3강의실 OCR 정정 로직 추가** (`_read_sign_digit`): YOLO 모델에는 애초에
  `3_class`가 없어서 3강의실 표지판은 항상 `2_class` 또는 `4_class`로
  오분류됨. `Locator` 내장 OCR confirm/reject 훅(`ocr_verify.OcrVerifier.verify()`)은
  "YOLO가 예상한 숫자와 일치하는지"만 확인하는 구조라 3강의실을 절대 못
  맞추길래(항상 reject), 별도로 `OcrVerifier.read()`를 직접 호출해서 표지판
  글자에서 2/3/4 숫자를 읽어 실제 위치로 정정하도록 함. 못 읽으면 YOLO
  라벨(2_class→room2, 4_class→room4)을 그대로 신뢰(폴백).

`realtime_pipeline.py`: `models/yolo/final_best.pt` 옆의
`final_best_v2_with_logo.pt`가 있으면 자동으로 `logo_weights`로 넘겨주도록
수정.

**검증**: conda env에서 `SeojiwooGridPositionTracker` 생성 시 격자 규칙에
`room3`/`logo`가 포함되는 것, `logo_model`이 6클래스로 정상 로딩되는 것,
`_read_sign_digit()`이 에러 없이 도는 것 확인.

## 23. 격자 지도에 물체(랜드마크) 고정 표시 추가 (2026-09-17)

기존엔 "격자 위치" 창에 현재 위치(빨간 점)만 표시되고, 그 기준이 되는
로고/4강의실/3강의실/2강의실/앞문/뒷문이 지도 어디에 있는지는 안 보였음.
이제 각 물체의 `grid_cell` 위치에 회색 점 + 한글 이름을 같이 그려서, 현재
위치가 어느 물체를 기준으로 잡힌 건지 지도에서 바로 확인 가능.

- `GridPositionTracker`에 `landmarks` 프로퍼티 추가 (`grid_cell`이 있는
  물체 목록, `_rules.values()`).
- `render_grid_image()`에 `landmarks` 매개변수 추가. 라벨이 지도 밖으로
  안 잘리도록, 위쪽 행(y=rows, 앞문/뒷문)은 점 아래에, 오른쪽 끝(x=cols,
  2강의실/뒷문)은 점 왼쪽에 글씨를 그림.
- `realtime_pipeline.py`: `render_grid_image(...) `호출 시
  `landmarks=grid_tracker.landmarks` 넘기도록 수정.

**검증**: 실제 렌더링 이미지를 떠서 눈으로 확인 — 로고/4강의실/3강의실/
2강의실이 아래축, 앞문/뒷문이 위쪽에 라벨과 함께 잘리지 않고 표시됨.

## 24. locations.yaml의 coords.x 최대값을 grid.cols(14)에 맞춤 (2026-09-17)

`room2`/`rear_door`의 `coords.x`가 `16`으로 남아있었음 — grid가 14칸
(x: 0~14)으로 확정되기 전, 예전 좌표 체계(scripts/visualize_map.py 전용,
실시간 격자 지도와는 무관)에서 쓰던 값이 안 지워지고 남아있던 것. 실시간
격자 지도(`SeojiwooGridPositionTracker`, `grid_cell` 기준)엔 영향 없었지만
헷갈릴 수 있어서 `coords.x: 16` → `14`로 맞춤 (room2, rear_door 둘 다).
관련 주석도 정리.

## 25. 3강의실 grid_cell 좌표 수정 (2026-09-17)

`room3`의 `grid_cell`을 `[6, 0]` → `[8, 0]`으로 수정 (coords.x와 동일하게
맞춤).

## 26. 웹 UI(webapp) 추가 — machine-vision1 frontend 연동 (2026-09-17)

`https://github.com/JIWOO1127/machine-vision1/tree/main/backend`(Flask+React
풀스택 웹앱, PC에 폰 브라우저로 접속해서 사진/영상/실시간 카메라 분석)를
조사함. 원본은 Faster R-CNN/YOLOv8 + PaddleOCR + 실측 mm 좌표계 기반
`VisualLocationEstimator`를 씀. 사용자가 "frontend는 그대로 연동"을
선택해서, frontend는 그대로 가져오고 backend는 새로 작성해서 우리
파이프라인에 연결함.

- `webapp/frontend/`: 원본 저장소의 `frontend/` 폴더를 **수정 없이** 복사
  (React+Vite, 사진 촬영/영상 업로드/실시간 카메라 3모드 + TTS 음성 안내
  UI).
- `webapp/backend/app.py`: 새로 작성한 Flask 서버. 원본과 같은 API 경로
  (`/api/health`, `/api/analyze`, `/api/analyze-frame`, `/api/reset-tracking`,
  `/api/results/<file>`)를 그대로 구현 — 라우팅/파일 처리만 담당, 판정
  로직은 `vision_bridge.py`에 위임.
- `webapp/backend/vision_bridge.py`: `SeojiwooGridPositionTracker`를 감싸서
  프론트가 기대하는 JSON 모양(`App.jsx`의 `cue.landmarks`/`stability`/
  `location.candidates` 등)으로 변환하는 어댑터. 새 탐지/거리 로직은
  안 만들고 기존 트래커 결과를 옮겨 담기만 함. 사진 1장 분석용으로는
  `window=1, min_votes=1, approach_only=False`로 설정한 별도 트래커
  (`photo_tracker`)를 만들어 프레임 투표 없이 즉시 판정하도록 함(라이브
  스트림용 `live_tracker`는 기존 기본값 그대로 유지).
- `grid_tracker.py`에 `SeojiwooGridPositionTracker.resolve_location_name()`
  공개 메서드 추가 (기존 `update_from_frame()`의 3강의실 OCR 정정 로직을
  추출해서 재사용 가능하게 함) + `last_result` 속성 추가 (같은 프레임에
  대해 webapp이 재추론 없이 `Locator.process()` 원본 결과를 다시 쓸 수
  있게 캐시).

**아직 안 한 것**: `/api/analyze-video`(영상 업로드 분석)는 501로 안내만
하고 미구현 — 필요하면 추가 예정. PaddleOCR 기반 텍스트 인식은 안 씀
(우리 프로젝트의 "OCR은 3강의실 정정용으로만" 결정을 그대로 따름).

**검증**: conda env에서 Flask 테스트 클라이언트로 `/api/health`,
`/api/analyze`, `/api/analyze-frame`, `/api/reset-tracking`,
`/api/analyze-video`(501 확인), `/api/locations`, `/api/results/<file>`
전부 기대한 상태 코드/JSON 모양으로 응답하는 것 확인 (CUDA 모델 정상 로딩,
`location_count: 7`).

## 다음 할 일

- [ ] `room3` 라벨링
- [ ] 표지판당 30~50장씩 다양한 거리/각도/조명으로 사진 추가 확보 (목표 150~300장 이상)
- [ ] 사진이 충분해지면 Roboflow에서 train/valid 분리하여 재 export
- [ ] bbox 크기 기반 "도착/복도" 판정 로직을 `core.py`/`sign_matcher.py`에 추가

## 27. Git 저장소 초기화 및 machine-vision1에 브랜치 푸시 (2026-09-17)

`git init -b classroom-locator`로 (main과 공유 이력 없는) 독립 브랜치로
초기화하고, `https://github.com/JIWOO1127/machine-vision1`에
`classroom-locator` 브랜치로 첫 커밋을 푸시함.

- `.gitignore` 정리: 기존엔 `models/yolo/*`를 통째로 무시하고 있었는데,
  이번엔 모델 가중치를 포함하기로 해서 그 규칙을 뺌. 대신 이전에 안
  걸러지던 `runs/`(YOLO 추론/학습 산출물, 99MB), `weights/`(베이스
  체크포인트), `yolov8n.pt`(ultralytics 자동 다운로드 베이스 모델),
  `.pytest_cache/`, `node_modules/`, `webapp/frontend/dist/`,
  `webapp/backend/results/*`를 새로 제외함.
- 원인 불명의 쓰레기 파일 `=3.8`(깨진 pip 명령어가 만든 것으로 추정) 삭제.
- 최종 커밋: 1009개 파일, models/yolo 4개 + external/seojiwoo_core 2개
  (.pt, 총 36MB) 포함, `data/dataset*`(Roboflow 학습 데이터) 포함.
- `git push -u origin classroom-locator` 성공, 원격에 새 브랜치 생성 확인
  (`git ls-remote`로 확인).

## 28. 위치별 좌표/안내 문구 수정 + 로고 2단계 지원 (2026-09-17)

사용자가 준 정확한 좌표/문구로 `configs/locations.yaml` 수정:
front_door "정문 앞"(0,2)/"뒤로 돌아주세요.", room4 "4 강의실 앞"(2,0)/
"벽을 따라 앞으로 직진하세요.", room2 "2 강의실 앞"(13,0)/"문까지 직진 후
왼쪽으로 돌아주세요.", rear_door "후문 앞"(14,2)/"직진해주세요.".

**로고는 두 단계**로 나눔 - 멀리서 처음 보일 때는 기존 (0,0) "로고"
(guidance 없음), 아주 가까이(1.5m 이내) 붙으면 (0,1) "중앙 로고 앞"
(guidance "왼쪽으로 돌아주세요.")으로 갈림. 이걸 위해
`GridPositionTracker`(`grid_tracker.py`) 구조를 바꿈:
- `_rules`가 `dict[str, Location]`(이름당 하나)에서
  `dict[str, list[Location]]`(이름당 여러 단계, snap_distance_m 오름차순
  정렬)로 바뀜 - **같은 name으로 locations.yaml에 여러 번 등록하면 여러
  단계가 됨**(멀리서/가까이 등), 다른 물체들은 원래대로 단일 항목 리스트라
  동작 그대로.
- `_best_stage(name, distance_m)`: 등록된 단계 중 distance_m이 처음으로
  맞아떨어지는(=가장 가까운 조건의) 단계를 고름.
- `current_location: Location | None` 속성 추가 - 지금 스냅된 지점이
  정확히 locations.yaml의 어느 항목(guidance 포함)인지 바로 접근 가능
  (기존엔 `current_label`이라는 문자열만 있어서 guidance를 못 가져왔음).
- `update()`(부모, calibration 기반)/`update_from_frame()`(Seojiwoo Locator
  기반) 둘 다 `_best_stage`/`_apply`를 공유하도록 리팩터링.

**UI/자막 문구를 실제로 locations.yaml의 guidance에서 가져오도록 연결**
(이전엔 항상 자동 생성 문장만 썼음):
- `webapp/backend/vision_bridge.py`: `tracker.current_location`이 있으면
  `"현재 위치는 '{display_name}'입니다. {guidance}"` 형식으로 `cue.guidance`
  구성(화면 자막 + TTS). 확정 전(랜드마크만 보이고 아직 스냅 안 됐을 때)엔
  기존처럼 "~m에서 ~이 보입니다" 문장으로 폴백.
- `grid_tracker.py`의 `render_grid_image()`도 `current_location` 인자를
  받아 CLI 격자 지도 창에도 같은 형식(+ guidance 두 번째 줄)으로 표시.
  `realtime_pipeline.py`에서 이 인자를 넘기도록 호출부 수정.

**검증**: front_door/rear_door/room4/room2/logo(멀리·가까이) 5가지
시나리오 전부 `vision_bridge._build_cue()`로 직접 돌려서 사용자가 준
문구와 정확히 일치하는 것 확인.

## 29. configs/ 폴더 정리 (2026-09-17)

파일 정리 요청으로 `configs/final_best.yaml`, `configs/yolo_only.yaml` 삭제.
둘 다 9/16 모델 비교 실험(OCR vs YOLO단독v1 vs YOLO단독v2, 15번 항목 참고)때
만든 것으로, 어떤 스크립트의 기본값도 아니고(`--config`로 수동 지정해야만
쓰임) 결정이 끝난 뒤로는 아무 데서도 참조 안 됨. `configs/default.yaml`
(전 스크립트 기본 설정)과 `configs/locations.yaml`(위치 매핑)만 남김 —
둘 다 grep으로 실제 사용처 확인 후 남긴 것.

## 30. 모델/데이터/학습산출물 정리 + 안 쓰는 파이썬 파일 대량 삭제 (2026-09-17)

파일 정리 2단계. 1단계(디스크만): `models/yolo/best.pt`/`best_yolo_only.pt`,
`data/dataset_yolo_only/`, `runs/`(99MB, 학습 실행 기록), `weights/`,
`yolov8n.pt`, `.pytest_cache/`, `outputs/eval|results|logs`의 내용물 삭제
(전부 재생성 가능, git엔 애초에 안 올라가 있었음 - `.gitignore` 참고).

2단계: **안 쓰는 파이썬 파일 삭제 + 관련 코드 리팩터링**. grep으로 실제
호출부가 있는지 하나씩 확인한 뒤 진행:

- **OCR 관련 완전 삭제** (메인 파이프라인은 이미 OCR 안 쓰기로 결정했었고,
  실제로 아무 데서도 안 부르고 있었음 - 3강의실 정정용 OCR은 완전히 다른
  경로인 `classroom_locator.seojiwoo.ocr_verify.OcrVerifier`라 안 건드림):
  `src/classroom_locator/ocr/`(전체 - easyocr/paddleocr/tesseract 리더 +
  base + 팩토리), `src/classroom_locator/localization/sign_matcher.py`
  (`SignMatcher`). `MatchResult` 데이터클래스만 `location_map.py`로 옮김
  (core.py가 OCR과 무관하게 "탐지 클래스 -> Location" 결과 타입으로
  여전히 씀).
- **평가(evaluation) 관련 완전 삭제** (사용자 요청): `src/classroom_locator/eval_metrics.py`,
  `src/classroom_locator/evaluation.py`, `scripts/evaluate.py`,
  `scripts/evaluate_test_sample.py`. `scripts/run_realtime.py`의 `--eval`
  옵션과 관련 코드도 같이 제거.
- **VisualLocationEstimator 계열 삭제** (아무 데서도 실제로 안 부르고 있었음 -
  격자 방식(SeojiwooGridPositionTracker)으로 완전히 대체됨):
  `src/classroom_locator/localization/visual_localization.py`,
  `scripts/estimate_position.py`, `scripts/calibrate_distance.py`,
  `data/distance_calibration.json`, `data/eval/`. `grid_tracker.py`의
  `GridPositionTracker`도 이제 안 쓰는 calibration 기반 `update()`
  메서드/`calibration` 매개변수를 제거하고 `SeojiwooGridPositionTracker`가
  공유하는 부분(_best_stage/_apply/경로 계산)만 남김.
- **평면도 시각화 삭제**: `scripts/visualize_map.py`,
  `src/classroom_locator/utils/map_viz.py`(`draw_floor_map`) - `coords`
  기반 구식 지도로, 실시간 격자 지도(`render_grid_image`)로 대체됨.
- **테스트**: 삭제된 모듈을 테스트하던 `tests/test_ocr.py`,
  `tests/test_localization.py` 삭제. `tests/test_detection.py`만 남음
  (`pytest` 2개 통과 확인).
- **`external/` 폴더도 삭제** (사용자 요청) - `src/classroom_locator/seojiwoo/`가
  이미 실제로 쓰는 이식 버전이라 원본 코드 파일들은 중복. 단, `.md 파일은
  제외`라는 지시를 처음에 놓쳐서 `external/seojiwoo_core/README.md`까지
  같이 지웠다가 git 기록(`git show HEAD:...`)에서 복원하고, 코드가 삭제된
  사실을 안내하는 문구를 맨 위에 추가함.
- `requirements.txt`에서 `rapidfuzz`(SignMatcher 전용), `matplotlib`
  (eval_metrics/map_viz 전용) 제거. `configs/default.yaml`의 `ocr:` 섹션과
  `localization.match_threshold` 제거.

**검증**: 전체 컴파일 체크 통과, conda env에서 `LocatorPipeline`/`SeojiwooGridPositionTracker`/
webapp Flask 테스트 클라이언트(`/api/health`, `/api/analyze`) 전부 정상
동작 확인, `pytest tests/` 2개 통과. grep으로 삭제된 모듈에 대한 잔여
참조 없음 확인.

이제 `src/classroom_locator/`에 OCR 백엔드/평가/시각화 관련 코드 없이
탐지 → 클래스 직접 매칭 → 격자 위치 추적(YOLO + 3강의실 OCR 정정)
경로만 남음.

## 31. final_best.pt를 logo 포함 6클래스로 교체, 모델 파일 1개로 통합 (2026-09-17)

사용자가 팀원 쪽에서 로고까지 포함해서 새로 학습한 모델을
`models/yolo/final_best.pt`에 직접 덮어썼고(기존 5클래스 내용은
`v1_best.pt`라는 새 이름으로 남겨둠), "이제 `final_best.pt` 하나만 쓰고
나머지는 다 지워달라"고 요청함.

파일로 직접 클래스 목록을 확인해서 반영:
- `final_best.pt` (교체됨): `2_class, 4_class, front_door, rear_door,
  water_dispenser, logo` — **6클래스, logo 포함**
- `v1_best.pt`(옛 final_best.pt 내용, 로고 없음), `final_best_v2_with_logo.pt`
  (교체 전 별도 로고 모델, 이제 새 final_best.pt와 내용 중복) 둘 다 삭제.
  `models/yolo/`엔 이제 `final_best.pt` 하나만 있음.

`final_best.pt` 자체가 로고를 포함하게 되면서, 로고 인식을 위해 별도
`logo_weights` 모델을 같이 돌릴 필요가 없어짐 - 관련 코드 단순화:
- `realtime_pipeline.py`: `logo_weights_path`(=`final_best_v2_with_logo.pt`
  자동 탐색) 로직 제거, `weights`만 넘김.
- `webapp/backend/vision_bridge.py`: `DEFAULT_LOGO_WEIGHTS` 제거.
- `scripts/run_seojiwoo_demo.py`: `--nav` 시 자동으로 로고 모델 채워주던
  로직 제거 (`--logo_weights`는 나중에 다시 모델을 분리하고 싶을 경우를
  대비해 옵션 자체는 남겨둠, 기본값 None).
- `configs/locations.yaml`의 logo 관련 주석 갱신.

**검증**: `SeojiwooGridPositionTracker`가 `final_best.pt` 하나로만 생성돼도
`logo_model`이 `None`(불필요)이고 주 모델의 `names`에 `logo`가 포함된
것 확인, webapp `/api/health` 정상 응답 확인. grep으로 삭제된 파일명에
대한 잔여 참조 없음 확인.

## 32. external/ 폴더 완전 삭제 (2026-09-17)

지난번(30번 항목)엔 "md 파일 제외" 지시 때문에 `external/seojiwoo_core/README.md`만
남기고 코드/모델 파일을 지웠는데, 이번엔 "external 파일이랑 폴더 다
지워줘, 필요하면 올바른 위치로 옮겨줘"라는 요청으로 그 README까지 포함해
폴더 전체를 삭제함.

README.md 중 아직 쓸모 있는 내용(원본 `Locator`/`Navigator`/`ocr_verify`와
우리 구현의 차이점 비교 - 실측 물리크기 거리추정, 접근추세 확인, 동선 제약,
근접시 OCR, TTS 등)은 그 코드가 실제로 있는 위치인
`src/classroom_locator/seojiwoo/README.md`로 옮겨서 새로 정리함 (오래된
파일 목록/실행법 등 지금은 안 맞는 부분은 빼고, 지금 구조에 맞게 다시 씀).

`external/seojiwoo_core/`를 로컬 경로로 언급하던 나머지 파일들의 주석도
정리 - `locator.py`/`navigator.py`/`ocr_verify.py`/`run_seojiwoo_demo.py`의
"원본:" 표기를 로컬 경로 대신 실제 GitHub URL로, `vision_bridge.py`/
`webapp/README.md`의 언급은 `classroom_locator/seojiwoo/`(코드가 실제로
있는 위치)로 갱신. `docs/setup_log.md`(이 파일)의 과거 기록은 그 시점의
사실이라 그대로 둠.

**검증**: grep으로 `external/seojiwoo_core` 잔여 참조가 `docs/setup_log.md`
(과거 기록, 의도적으로 유지) 외에는 없는 것 확인. `classroom_locator.seojiwoo`
import 정상 동작 확인.

## 33. CLI 격자 지도에 방향 표시 추가 + "위치 불확실" UI 문구 제외 (2026-09-17)

- `grid_tracker.py`에 `_direction_from_box()` 추가 (webapp `vision_bridge.py`의
  `_direction_from_box`/`Navigator._steer`와 동일 임계값 0.35/0.65). `GridPositionTracker`에
  `current_direction` 속성 추가, `_apply()`가 같이 저장하도록 수정.
  `SeojiwooGridPositionTracker.update_from_frame()`이 확정된 탐지의 bbox로
  방향을 계산해서 넘김. `render_grid_image()`가 `current_direction`을 받으면
  캡션에 "(왼쪽에서 보임)" 식으로 붙여서 보여줌. `realtime_pipeline.py`
  호출부도 갱신. 실제 렌더 이미지로 확인 완료.
- 웹 UI에서 "위치 불확실"/"재생 대기" 같은 미확정 상태 문구를 아예 안 보이게
  변경 (요청에 따라 완전 제외, 대체 문구 없음): `webapp/frontend/src/App.jsx`의
  `LivePanel`(`live-location-row`)과 결과 카드(`location-box`)를
  `location?.status === 'localized'`일 때만 렌더링하도록 조건부 처리.
  `vision_bridge.py`의 `_build_cue()`도 위치 미확정 + 탐지물도 없을 때
  "이 화면에서는 위치를 판단할 기준 객체가 보이지 않습니다" 대신 빈 문자열을
  반환하도록 수정(프론트가 자체 안내 문구로 대체). `npm run build`로 재빌드함.

## 34. `seojiwoo` -> `landmark_locator` 이름 변경, Navigator 삭제, 위치별 안내 재정비 (2026-09-17)

**이름 변경**: 사람 이름(`서지우`)을 패키지/클래스 이름으로 쓰는 게 부적절하다는
지적에 따라 전면 변경.
- `src/classroom_locator/seojiwoo/` → `src/classroom_locator/landmark_locator/`
- `SeojiwooGridPositionTracker` → `LandmarkGridPositionTracker`
- `scripts/run_seojiwoo_demo.py` → `scripts/run_landmark_demo.py`
- 관련 import/문서(`webapp/README.md`, `landmark_locator/README.md` 등) 전부 갱신.
  원본 GitHub URL(`.../seojiwoo-detection/seojiwoo/core/...`)과 "서지우님"
  이름 attribution은 실제 출처 표기라 그대로 둠.

**`Navigator` 완전 삭제**: "SEARCH_HINT 같은 중복 코드가 많아 보인다"는 지적을
조사해보니, 진짜 문제는 중복이 아니라 `Navigator`(그리고 `STEPS`/`SEARCH_HINT`/
`ON_REACH`)가 `scripts/run_landmark_demo.py --nav` 말고는 **실제 제품(웹 UI,
CLI 격자 지도) 어디에도 안 쓰이는** 죽은 코드였다는 것. `navigator.py` 삭제,
`landmark_locator/__init__.py`에서 export 제거, `run_landmark_demo.py`에서
`--nav`/`--wall_m`/시나리오 관련 코드 제거.

**위치별 안내 문구/좌표 최종본**으로 `configs/locations.yaml` 갱신:
- front_door (0,2) "정문 앞" / "뒤로 돌아주세요."
- logo 2단계: (0,0) "로고"(멀리서, guidance 없음), (0,1) "중앙로고 앞"
  (1.5m 이내) / "왼쪽으로 돌아주세요."
- room4 (2,0) "4 강의실 앞" / "벽을 따라 앞으로 직진하세요."
- room2 (13,0) "2 강의실 앞" / "문까지 직진 후 왼쪽으로 돌아주세요."
- rear_door 2단계(신규): (14,2) "후문 앞"(3.0m 이내) / "직진해주세요.",
  (14,3) "목적지"(1.0m 이내, 더 가까움) / "도착하였습니다. 안내 모드를
  종료합니다." — logo와 같은 다단계(`_best_stage`) 패턴 재사용.

**로고 중앙 정렬 동적 안내 추가**: Navigator의 `_steer()`(bbox 중심 x로
좌/우/직진 판정, 임계값 0.35/0.65) 아이디어 중 이 부분만 뽑아서 재구현.
`grid_tracker.py`에 `_describe_position(location, direction)` 모듈 함수 신설 -
로고를 추적 중인데 화면 중앙(앞쪽)이 아니면, locations.yaml에 적힌 고정
guidance 대신 "왼쪽/오른쪽으로 회전해서 중심을 유지해주세요"라는 동적 문구로
바꿔치기함 (중앙이면 원래 guidance 그대로). `GridPositionTracker.describe()`
메서드로 감싸서 웹/CLI 양쪽이 같은 로직을 공유하도록 함(전에 있던 중복 로직
문제를 이번엔 제대로 해소):
- `render_grid_image()`: `_describe_position()` 결과를 헤드라인/보조문구
  두 줄로 표시.
- `webapp/backend/vision_bridge.py`의 `_build_cue()`: `tracker.describe()`
  결과를 그대로 `cue.guidance`로 씀 (미확정 상태일 때는 기존 폴백 로직 유지).

**검증**: 9가지 시나리오(정문/4강/2강/후문/목적지/로고 멀리·중앙·좌·우)를
`tracker._apply()` + `tracker.describe()`로 직접 돌려서 전부 사용자가 준
문구와 정확히 일치하는 것 확인. `render_grid_image()` 실제 렌더 이미지로도
로고-왼쪽 시나리오 확인. webapp `/api/health` 정상 응답(`location_count: 9`,
새 rear_door 2단계 항목 포함) 확인.

## 35. room3 문구 정리 + 로고 각도 안내를 "3초 이상 놓쳤을 때만"으로 재설계 (2026-09-17)

**room3**: `(8, 0)`에 이미 등록돼 있었는데(사용자가 "새로 추가"라고 착각했던
부분), 예전 스타일 문구("3강의실"/"앞으로 쭉 직진하세요.")가 안 바뀌어 있던
것만 최신 패턴("3 강의실 앞"/"계속 벽을 따라 직진하세요.")으로 갱신.

**로고 거리 2단계 폐지**: "거리에 따라 다른 메시지가 나오는 게 싫다"는
피드백에 따라, logo를 (0,0)"로고"(멀리서)+(0,1)"중앙로고 앞"(가까이) 2단계에서
**(0,1) "중앙로고 앞" 단일 지점**으로 되돌림 (`snap_distance_m: 3.0`, 다른
지점들과 동일 패턴). rear_door의 2단계((14,2) "후문 앞" / (14,3) "목적지")는
사용자가 이번에도 그대로 유지해서 안 건드림 - "거리별 다른 메시지" 불만은
로고 한정이었음.

**로고 각도 안내를 "실시간 off-center 감지"에서 "3초 이상 놓쳤을 때만"으로
전면 재설계**: 이전엔 로고가 화면 중앙이 아니면 매 프레임 "중심을
유지해주세요"가 계속 나왔는데, 이제는 **로고를 3초 이상 새로 못 찾았을 때만**
"왼쪽/오른쪽으로 더 회전해주세요"가 나오고, 평소(잘 보일 때)엔 화면 중앙이든
아니든 그냥 locations.yaml의 고정 guidance("왼쪽으로 돌아주세요.")만 나옴.

구현 (`grid_tracker.py`):
- `_LOGO_LOST_TIMEOUT_S = 3.0` 상수 추가.
- `GridPositionTracker`에 `current_matched_at`(마지막 실제 갱신 시각)과
  `logo_lost`(bool) 상태 추가.
- `_apply()`가 `t` 매개변수를 받아 `current_matched_at`을 갱신하고
  `logo_lost`를 즉시 False로 리셋.
- `LandmarkGridPositionTracker.update_from_frame()`을 재구성 - 이번 프레임에
  못 잡았어도(조기 return 대신) 매번 "로고 추적 중이면 몇 초째 못 찾고
  있는지" 계산해서 `logo_lost`를 갱신하도록 흐름을 바꿈
  (`out["t"] - current_matched_at >= 3.0`).
- `_describe_position(location, direction, lost)`: `lost=True`일 때만
  "{마지막 방향}으로 더 회전해주세요." (마지막 방향이 없거나 "앞쪽"이었으면
  일반적인 "왼쪽이나 오른쪽으로 천천히 회전해보세요.")로 바뀜.
- `render_grid_image()`/`describe()`/`realtime_pipeline.py` 호출부에
  `logo_lost` 전달.

**검증**: t=0(감지, 왼쪽) → t=2(2초 미탐지, 아직 고정 guidance 유지) →
t=3.5(3.5초 미탐지, "왼쪽으로 더 회전해주세요"로 전환) → t=4(재감지, 오른쪽,
즉시 고정 guidance로 복귀) 시나리오 전부 의도대로 동작 확인.
webapp `/api/health` 정상(`location_count: 8`, 로고 단일화 반영) 확인.

## 36. 로고 각도 안내 조건 정정: "3초 미탐지"가 아니라 "3초 연속 중앙 이탈" (2026-09-17)

35번 항목에서 구현한 조건이 실제 요청과 달랐음이 확인됨 - "3초 이상 못
찾으면"(탐지 자체가 끊김)이 아니라 **"로고가 보이긴 하는데 화면 중앙이
아닌 상태가 3초 이상 계속되면"**이 맞는 조건이었음. 재구현:

- `_LOGO_LOST_TIMEOUT_S` → `_LOGO_OFF_CENTER_TIMEOUT_S`(3.0)로 이름/의미 변경.
- `current_matched_at`(마지막 매칭 시각)에 기반한 "미탐지 경과 시간" 계산
  대신, `_logo_off_center_since`(중앙 이탈이 시작된 시각)를 추적. 매 프레임
  로고가 왼쪽/오른쪽으로 잡히면: 이번이 처음 벗어난 거면 그 시각을 기록,
  이미 벗어나 있던 중이면 시작 시각은 그대로 두고 경과 시간만 재계산.
  중앙(앞쪽)으로 돌아오거나 다른 랜드마크가 되면 즉시 해제.
- `logo_lost` bool 속성 → `logo_off_center`로 이름 변경 (`grid_tracker.py`,
  `realtime_pipeline.py`, `render_grid_image()` 전부 갱신). `update_from_frame()`은
  "매칭 안 된 프레임에서도 매번 재계산"하던 로직이 필요 없어져서 원래의
  단순한 early-return 스타일로 되돌림 - 이제 이탈 타이머는 오직 실제로
  로고가 다시 잡힐 때(`_apply()` 호출 시점)만 갱신됨.
- 메시지도 "왼쪽/오른쪽으로 더 회전해주세요"(미탐지를 전제로 한 "더") →
  "왼쪽/오른쪽으로 회전해주세요"(이미 알고 있는 현재 방향으로) 로 단순화.

**검증**: 왼쪽 치우침이 t=0→3.2초까지 연속되는 시나리오에서 3.2초째부터
`logo_off_center=True`로 바뀌는 것, 중간에 중앙으로 한 번 돌아오면 즉시
해제되는 것, 다시 치우치기 시작하면 타이머가 그 시점부터 새로 재는 것
(이전 이력에 영향 안 받음) 전부 확인. webapp 정상 동작 확인.

## 37. 격자 지도 점 라벨에서 "~앞" 제거 (2026-09-17)

`render_grid_image()`가 점 옆에 찍는 이름표(예: "정문 앞", "목적지")를
음성/자막 안내 문구(`display_name`, 그대로 유지)와 분리해서, 지도용으로만
짧은 이름을 쓰도록 `_MAP_LABELS` 매핑 추가: front_door→앞문, rear_door→뒷문
(두 단계 다 동일하게 "뒷문"), room2/3/4→2/3/4강의실, logo→로고. 아래쪽 안내
문구("현재 위치는 '2 강의실 앞'입니다...")는 안 건드림. 렌더 이미지로 확인.

## 38. 죽은 `coords` 필드 완전 삭제 (2026-09-17)

"모든 장소 위치가 다르다"는 지적을 조사해보니, `configs/locations.yaml`에
예전 좌표 체계(`coords: {x, y}`)가 `grid_cell`과 다른 값으로 그대로 남아있던
게 원인이었음 (예: room2는 `coords.x=14`인데 `grid_cell=[13,0]`, front_door는
`coords={x:-2,y:5}`인데 `grid_cell=[0,2]`로 완전히 다른 스케일). `coords`는
그걸 그리던 `scripts/visualize_map.py`/`utils/map_viz.py`가 30번 항목에서
이미 삭제돼서 지금은 어떤 코드에서도 안 읽는 완전한 죽은 필드였음
(`grep '\.coords\b'` 결과 0건 확인 후 제거).

- `configs/locations.yaml`: 모든 위치에서 `coords` 필드 제거. 관련 주석도 정리.
- `src/classroom_locator/localization/location_map.py`: `Location` 데이터클래스의
  `coords` 필드와 로더의 `coords=item.get("coords")` 라인 제거.

이제 위치 좌표는 `grid_cell` 하나만 있음 (단일 소스). 로딩 테스트 +
webapp `/api/health` 정상 확인.

## 39. 격자 좌표 최종 확정: 그리드 모서리/끝점 배치로 정리 (2026-09-17)

사용자가 6개 지점 좌표를 다시 정리해서 줌 - 세션 초반의 원래 구상(앞문/뒷문을
격자 끝점에, room4/room2를 x축 위에)으로 되돌아가는 형태:

- 앞문(front_door): (0,2) → **(0,4)** (y축 맨 끝)
- 로고(logo): (0,1) → **(0,0)** (원점, x축·y축 교차점)
- 4강의실(room4): (2,0) → **(3,0)**
- 3강의실(room3): (8,0) 그대로
- 2강의실(room2): (13,0) → **(14,0)** (x축·x=cols축 교차점)
- 뒷문(rear_door): (14,2)/(14,3) → **두 단계 모두 (14,4)**로 통일 (x=cols축
  맨 끝). 거리가 가까워질수록 어차피 두 단계 다 자기 anchor 쪽으로 수렴하는
  방식이라, 같은 anchor를 써도 "후문 앞"(3.0m)/"목적지"(1.0m) 두 단계
  거리 임계값 차이는 그대로 유지됨 - "목적지 도착" 안내 기능은 안 없앰
  (사용자가 언급 안 했지만 최근에 명시적으로 원했던 기능이라 임의로 빼지
  않고 새 anchor에 맞춰 유지, 답변에서 이 판단을 명시적으로 알림).

**검증**: 6개 지점 전부 새 좌표로 로딩되는 것, describe() 7가지 시나리오
전부 문구 그대로 유지되는 것, 실제 렌더 이미지로 앞문/로고/4강/3강/2강/뒷문이
격자 모서리·끝점에 정확히 찍히는 것 확인. webapp 정상 동작 확인.

## 40. "~앞 없는 짧은 이름표"를 webapp에도 연동 (2026-09-17)

37번 항목에서 CLI 격자 지도 점 라벨에만 적용했던 `_MAP_LABELS`를
`grid_tracker.py`에서 `MAP_LABELS`(공개)로 이름 바꾸고, `webapp/backend/vision_bridge.py`의
`_landmarks_and_labels()`(실시간 탐지 목록 - `cue.landmarks[]`/`cue.detections[]`,
CLI 격자 지도의 점들과 같은 역할)에도 적용. "현재 위치" 확정 문구
(`_location_summary`/`describe()`, 예: "현재 위치는 '4 강의실 앞'입니다...")는
CLI와 마찬가지로 안 건드리고 그대로 둠 - 확정 위치는 풀네임, 화면에 보이는
물체 목록은 짧은 이름, 이 구분을 웹/CLI 양쪽에서 동일하게 유지.

**검증**: `_build_cue()`로 직접 확인 - `location.label`/`guidance`는 여전히
"4 강의실 앞" 풀네임, `landmarks[0].name`/`detections[0].label`은 "4강의실"로
짧게 나오는 것 확인. webapp `/api/health` 정상 확인.

**참고**: 이 세션 마지막에 사용자가 "4·3·2강의실이 전혀 인식이 안 된다"고
보고함 - 원인 조사를 시작하려다 사용자가 먼저 이 UI 연동 작업을 요청해서
그쪽으로 전환함. 인식 문제 원인은 아직 미확인 (다음 세션에서 이어서 조사
필요 - 표지판 전용 conf 임계값(`sign_conf=0.65`)이 실제 영상 대비 너무
엄격할 가능성, room3는 OcrVerifier 초기화 실패 시 항상 인식 불가능한
구조적 한계 등을 우선 의심해볼 것).

## 41. "인식은 되는데 지도에 안 뜸" 진단용 표시 추가 (2026-09-17)

사용자가 "4·3강의실은 인식은 되는데 지도엔 안 뜬다"고 구체화 - 탐지 자체는
성공하지만(YOLO conf 통과 + 프레임 투표 통과) 격자 갱신의 마지막 문턱인
`distance_m <= snap_distance_m`(room3/room4는 2.5m, 문/로고는 3.0m)을 못
넘는 상황으로 추정됨. 근데 그동안 실제 계산된 거리를 확인할 방법이 CLI
화면에 전혀 없어서(그리드 트래커의 판정 과정이 안 보임), 원인 진단용으로
표시를 추가함:

- `GridPositionTracker`에 `pending_candidate: tuple[str, float, float] | None`
  추가 (랜드마크 이름, 계산된 거리, 문턱값). 투표까지는 통과했는데
  `_best_stage()`가 거리 초과로 None을 반환하면 여기에 기록되고, 확정
  갱신되면 다시 None으로 지워짐.
- `render_grid_image()`가 이 값을 받으면 화면 맨 위에 회색 글씨로
  "4강의실 감지됨 · 4.2m (기준 2.5m 이내)" 식으로 보여줌.
- `realtime_pipeline.py`가 매 프레임 이 값을 넘겨주도록 연결.

이걸로 실제 테스트해보면 표시되는 거리 값을 보고 `snap_distance_m`을
얼마로 조정해야 할지(또는 REAL_SIZE 물리 크기 보정이 필요한지) 바로 판단
가능해짐 - 다음 실제 테스트 결과 기다리는 중.

**검증**: `pending_candidate`를 수동으로 채운 뒤 `render_grid_image()` 렌더
결과로 화면에 정상 표시되는 것 확인. webapp 정상 동작 확인(이 기능은
CLI 격자 지도 전용, webapp엔 아직 안 넣음).

## 42. room2/3/4 인식 안 됨 - 근본 원인 확인 및 수정: snap_distance_m 상향 (2026-09-17)

사용자가 "room2/3/4는 인식은 되는데 지도엔 안 뜬다"고 재확인 - 41번 항목의
진단 표시를 기다리는 대신, `data/dataset/`의 실제 학습 이미지(182장, 문/표지판/
정수기 등이 섞여 찍힌 원본 사진들)로 `Locator`를 직접 돌려서 실측함.

**측정 결과** (2_class/4_class 감지 59건): 계산된 거리 범위 1.25~5.48m,
중앙값 3.18m. 기존 문턱 `snap_distance_m: 2.5`로는 이 중 **32%만 통과**
(2.5m 이내인 것만) - 즉 버그가 아니라 **표지판 클래스의 거리 계산값 자체가
2.5m보다 훨씬 크게 나오는 경우가 다수**였음. `logo_off_center` 조사(41번)
때 추측했던 "표지판 REAL_SIZE(팀원 저장소 값 그대로 사용 중, 우리 표지판 실측
아님)가 안 맞을 수 있다"는 게 실제 원인으로 확인됨 - 다만 정확한 물리
크기를 재실측하지 않는 한 계산식 자체를 정확히 고칠 수는 없어서, 우선
임계값을 실측 분포에 맞게 넓히는 방식으로 대응.

- `configs/locations.yaml`: room4/room3/room2의 `snap_distance_m`을
  `2.5` → `5.0`으로 상향 (그 표본의 97% 커버). 관련 근거를 주석으로 남김.

**같은 182장으로 실제 파이프라인(`update_from_frame`) 전후 비교**: 기존
2.5m 기준으로는 18장만 room2/3/4로 격자 갱신됐는데, 5.0m로 올리니 55장으로
3배 증가 확인 (같은 이미지 세트, 같은 모델).

**남은 과제**: 근본적으로는 REAL_SIZE(현재 2_class/4_class 둘 다
0.20m×0.13m)를 우리 실제 표지판 크기로 재실측하면 더 정확해질 수 있음 -
지금은 "충분히 넓혀서 놓치지 않게" 하는 임시 대응. `pending_candidate`
진단 표시(41번)는 앞으로도 유용하니 그대로 둠.

## 43. "4강의실 앞에서 지도 위치가 안 갱신됨" - OCR 오독으로 인한 room2 오판정 버그 수정 (2026-09-17)

사용자가 `--source testvideo.mp4 --grid-map`으로 재생 시 "메인 화면의
텍스트/위치 안내(핵심 파이프라인 `core.py` 경로)는 정확한데, 격자 지도
(`grid_tracker.py` 경로)만 특히 4강의실 앞에서 위치가 갱신되지 않는다"고
보고. 두 경로는 완전히 별개 구현(메인 화면은 YOLO 클래스명으로 바로
`locations.yaml` 조회, 격자 지도는 seojiwoo `Locator`의 투표+접근추세+OCR
판정을 거침 - 20번 항목 참고)이라 한쪽만 고장 날 수 있는 구조.

**진단**: cv2 GUI 없이 `LandmarkGridPositionTracker.update_from_frame()`을
프레임마다 그대로 호출하며 `landmark`/`distance_m`/`reason`/`current_cell`을
찍는 헤드리스 스크립트로 실제 영상을 재생해 재현. 4강의실 표지판이 0.85m
거리로 뚜렷하게 잡혀 투표까지 통과한 순간, 로그에 `reason=OCR reject
"강의실2"`와 함께 `cell_before=(0,0) cell_after=(14,0)`이 찍힘 - **4강의실
바로 앞인데 격자 위치가 2강의실 지점((14,0))으로 튀어버림**. 화면상으로는
"엉뚱한 곳으로 튐" 또는 (그 뒤로 4_class 투표가 다시 min_votes를 못 채우고
`rear_door`로 넘어가버려) 사실상 "4강의실 근처에서는 갱신 안 됨"으로
보였던 것.

**근본 원인**: `resolve_location_name()`(`grid_tracker.py`)의 의도는
docstring에 명시된 대로 "YOLO가 구조적으로 낼 수 없는 3강의실만 OCR로
잡아내고, 2/4는 YOLO 라벨을 그대로 신뢰"하는 것이었는데, 실제 코드는
`_read_sign_digit()`이 뭐가 됐든 숫자 하나(2/3/4)를 읽기만 하면 그 숫자로
덮어썼음 (`if digit is not None: return _DIGIT_TO_LOCATION_NAME[digit]`).
이 영상에서 4강의실 표지판을 OCR이 "강의실2"로 오독(글자 순서/폰트 문제로
"4"를 "2"처럼 읽은 것으로 추정)하면서, YOLO가 이미 올바르게 `4_class`로
분류했는데도 `room2`로 잘못 정정되어 격자 지도가 저 멀리 있는 2강의실
지점으로 순간이동한 것.

**해결**: `resolve_location_name()`에서 `digit == "3"`일 때만 OCR 결과를
신뢰하도록 조건 추가 (`grid_tracker.py`). 2/4를 오독해도 더 이상 YOLO
라벨을 덮어쓰지 않음.

**검증**: 같은 헤드리스 재생 스크립트로 재확인 - 수정 전엔 4강의실 근처에서
`cell_after=(14, 0)`(2강의실 지점)으로 튀었던 것이, 수정 후엔
`cell_after=(2, 0)`(4강의실 grid_cell `(3,0)` 바로 옆, 거리가 가까워질수록
자연스럽게 근접)으로 정상 갱신되고 이후 여러 프레임 동안 유지되는 것 확인.

## 44. 격자 지도의 room3 문턱 우회 추가 + 메인 화면에 로고/room3 인식 도입 + 안내 문구 단순화 (2026-09-17)

**44-1. 격자 지도에서 room3가 사실상 인식 안 되던 문제**: 사용자가 "3강의실
인식이 안 된다"고 보고. 43번에서 고친 OCR 오독 문제와 별개로, room3 판정
자체가 `LandmarkGridPositionTracker.update_from_frame()`의 표결 문턱(최근
5프레임 중 4번 이상 같은 랜드마크여야 함, `reason.startswith("votes")`면
그 프레임은 `resolve_location_name()`/OCR 호출까지 가지도 못하고 보류됨)에
막히고 있었음. room3 표지판은 YOLO 전용 클래스가 없어 2_class/4_class로만
애매하게 잡히는 데다 화면에도 짧게만 스쳐서, 헤드리스 재생 로그로 확인해보니
`votes 1/5`에서만 계속 맴돌고 4/5를 넘긴 적이 없었음 - OCR이 호출될 기회
자체가 거의 없었던 것.

**해결**: `update_from_frame()`에서 랜드마크가 2_class/4_class일 때는 표결
문턱과 무관하게 즉시 OCR로 숫자를 읽어보고, "3"이 읽히면 그 즉시 room3로
반영하도록 예외 경로 추가 (room2/room4/문/로고의 기존 표결 안정화 로직은
그대로 유지 - room3만 예외). 같은 영상 재검증 결과, 이전엔 4강의실 위치에
멈춰있던 구간이 정상적으로 room3 위치((7,0), room3 grid_cell (8,0) 인근)로
전환·유지되는 것을 확인.

**44-2. 메인 화면(지도 제외)에 로고/room3 인식 추가**: 사용자가 "3강의실엔
바운딩박스가 쳐지는데 room3로는 안 뜨고, 로고는 박스 자체가 안 쳐진다"고
확인 요청 - 조사해보니 **로고는 이 프로젝트 역사상 단 한 번도 메인 화면
경로(`pipeline/core.py`의 `LocatorPipeline`)의 인식 대상에 포함된 적이
없었음** (`configs/default.yaml`을 비롯해 지금까지 존재했던 모든 설정 파일의
`target_classes`를 확인함 - 로고 관련 작업은 전부 격자 지도 쪽
(`grid_tracker.py`/`Locator`)에만 있었음, `docs/setup_log.md` 17~31번 참고).
3강의실도 마찬가지로 `core.py`는 원래(2026-09-17 오전, 16번 항목) "메인
파이프라인은 OCR 완전 배제"로 정했던 결정 때문에 OCR 기반 정정 로직 자체가
없어서 구조적으로 room3를 낼 수 없었음.

사용자가 이 결정을 되돌리고 메인 화면에도 로고/room3 인식을 넣어달라고
요청함에 따라 반영:

- **로고**: `configs/default.yaml`의 `detector.target_classes`에 `"logo"`
  추가 (YOLO가 이미 6클래스 중 하나로 직접 분류하므로 그 외 코드 변경 불필요).
- **room3**: `pipeline/core.py`의 `LocatorPipeline`에 OCR 기반 정정 로직 추가
  - `_ocr_verifier`를 `__init__`에서 준비(단, `locations.yaml`에 room3가 없으면
    스킵 - 불필요한 easyocr 로딩 방지)
  - `_resolve_location()`: 탐지가 room2/room4로 분류됐을 때만 표지판 bbox를
    OCR로 읽어 "3"이면 room3로 정정, 아니면 YOLO 라벨 그대로 신뢰 (43번에서
    지도 쪽에 적용한 것과 동일한 "3만 신뢰" 원칙)
  - **중복 방지**: OCR 헬퍼(`build_verifier()`/`read_digit()`)를
    `landmark_locator/ocr_verify.py`로 옮겨서 `grid_tracker.py`와 `core.py`가
    공용으로 쓰도록 정리 (기존엔 grid_tracker.py에만 있던 정적 메서드/사설
    로직을 재사용 가능한 모듈 함수로 승격)
  - 이 변경은 "OCR로 전체 텍스트를 읽어 위치를 찾는" 예전 방식(9번/14번
    항목에서 성능이 나빠서 폐기됨)과는 다름 - 표지판이 이미 근접
    판정됐을 때 숫자 하나만 좁게 재확인하는 용도로 범위가 훨씬 좁음.

**44-3. 안내 문구를 "인식된 객체: (이름)"으로 단순화**: 사용자가 로고/room3를
포함한 모든 인식 객체에 대해, 기존 "현재 위치는 X 앞입니다. <guidance>"
안내 대신 그냥 "인식된 객체: (이름)"만 표시해달라고 요청. `realtime_pipeline.py`의
`_build_guidance_lines()`를 이렇게 교체 (한 줄만 반환, `guidance` 필드는 더
이상 이 화면에서 안 씀). 이름 표시는 `grid_tracker.py`의 공개 상수
`MAP_LABELS`(예: room4→"4강의실", logo→"로고")를 그대로 재사용해서 지도
쪽과 표기를 통일함.

**검증**: `testvideo.mp4`로 `core.LocatorPipeline`을 직접 헤드리스로 돌려
`object_matches`에 잡힌 위치 이름 전체를 확인 -
`{logo, room2, room3, room4, rear_door}` 전부 정상적으로 잡히는 것 확인
(예: "인식된 객체: 로고", "인식된 객체: 3강의실"). 패키지 임포트 순환 문제
없음 확인 (`landmark_locator` -> `pipeline.core`/`pipeline.grid_tracker` 양쪽
다 `ocr_verify`의 `build_verifier`/`read_digit`을 문제없이 가져다 씀).

**성능 트레이드오프**: `core.py`는 격자 지도처럼 표결/근접 게이팅이 없어서,
room2/room4로 분류된 프레임마다 매번 OCR을 호출함 (기존엔 지도 쪽에서
표결 통과한 프레임에서만 호출됐음). 실사용 중 프레임이 눈에 띄게 느려지면
이 부분을 근접(bbox 크기) 조건으로 한 번 더 제한하는 걸 고려할 것.

## 45. "2강의실을 4강의실로 인식" - YOLO의 2_class/4_class 혼동을 OCR 재확인으로 정정 (2026-09-17)

44번에서 메인 화면(`core.py`)에 room3 OCR 정정을 넣은 직후, 사용자가
"2강의실이 4강의실로 인식된다"고 보고.

**원인 조사**: `testvideo.mp4`를 헤드리스로 재생하며 YOLO 원본 분류 결과와
OCR이 같은 bbox에서 읽은 숫자를 나란히 찍어봄. 프레임 940에서 실제 사례를
확인: 같은 프레임에 `room4`로 분류된 박스가 2개 있었는데, 그중 하나(더
작고 신뢰도도 낮은 박스, conf 0.534)는 OCR로 읽으면 "2"가 나옴 - 즉
**실제로는 2강의실 표지판인데 YOLO 분류기 자체가 4_class로 잘못
분류**하고 있었음. `2_class`/`4_class`는 표지판 생김새가 거의 동일해서
(REAL_SIZE 실측값도 두 클래스가 완전히 같음, 16번 항목 참고) YOLO
분류기가 종종 혼동하는 게 확인된 셈 - 이번에 새로 만든 버그가 아니라
모델 자체의 기존 한계였고, 지도 쪽(`grid_tracker.py`)은 다수결 투표+접근
추세 확인으로 이런 일회성 오분류가 어느 정도 걸러졌지만, `core.py`는
프레임 단위로 바로 반영하는 구조라 그대로 노출된 것.

**주의할 점**: 43번에서 "OCR이 읽은 숫자를 무조건 신뢰하면 안 된다"는 걸
이미 한 번 겪었음 (실제 4강의실 표지판을 OCR이 "강의실2"로 잘못 읽어서
room2로 잘못 정정된 사례). 그래서 이번엔 OCR을 무조건 신뢰하는 대신:
- OCR이 "3"을 읽으면 - YOLO는 애초에 3을 낼 수 없는 클래스라 오판 리스크가
  없으므로 즉시 신뢰 (기존과 동일)
- OCR이 "2"/"4"를 읽었는데 YOLO 라벨과 다르면 - 한 프레임의 오독만으로
  뒤집지 않고, **같은 불일치(같은 YOLO 라벨 + 같은 OCR 숫자)가 연속으로
  한 번 더 나와야만** 정정하도록 2프레임 확인 로직 추가 (`core.py`의
  `_ocr_disagreement`, `Locator.ocr_confirm_frames=2`와 동일한 관례를
  재사용).

**변경 파일**: `pipeline/core.py`의 `_resolve_location()`에
`_DIGIT_TO_SIGN_LOCATION={"2":"room2","4":"room4"}` 매핑과 2프레임 확인
스트릭(`self._ocr_disagreement`) 추가.

**한계**: 이 스트릭은 객체별이 아니라 파이프라인 전체에 1개만 있어서,
극히 드물게 서로 다른 두 물체가 우연히 같은 (YOLO 라벨, OCR 숫자) 조합을
연속으로 내면 잘못 정정될 수 있음 - 이론적 엣지 케이스이고 실사용
빈도로는 무시 가능하다고 판단해 더 복잡한 객체별 추적은 넣지 않음.

**검증**: `testvideo.mp4` 재검증 - `object_matches`에 잡히는 위치 이름
`{logo, room2, room3, room4, rear_door}` 그대로 유지(회귀 없음), 프레임
940의 그 일회성 불일치는 2프레임 확인을 통과하지 못해 정정되지 않고
넘어감(정확히 의도한 동작 - 바로 다음 관측 구간인 프레임 980부터는 YOLO
자체가 이미 정확하게 room2로 분류해서 문제 없음).

## 46. 45번 수정 후에도 "여전히 2강의실을 4강의실로 읽는다" - 진짜 원인은 박스 라벨 미반영 (2026-09-17)

45번 수정을 적용했는데도 사용자가 "아직도 2강의실을 4강의실로 읽는다"고
재보고. 재조사 결과 **두 가지가 겹쳐 있었음**:

1. **박스 라벨이 정정을 반영 안 함 (진짜 원인)**: `_resolve_location()`이
   정정된 `Location`을 반환해도, 화면에 박스와 함께 찍히는 텍스트는
   `image_utils.draw_detections()`가 원본 `Detection.class_name`(YOLO가
   내놓은 그대로, 즉 오분류된 "room4")을 그대로 씀 - `process_image()`가
   `candidates`/`match`에는 정정된 `Location`을 넣어줬지만, 정작
   `FrameResult.detections`(박스 그리는 데 쓰는 리스트)의 `Detection`
   객체 자체는 안 건드렸던 것. 그래서 "인식된 객체: X" 캡션 문구는
   맞게 나와도, 박스 옆 라벨은 여전히 YOLO의 원래(틀린) 이름으로 남아있어
   사용자 눈엔 "여전히 4강의실로 읽는다"로 보였음.
   **해결**: `process_image()`에서 위치가 정정되면 `det.class_name`
   자체를 `location.name`으로 갱신 (`Detection`은 `@dataclass`라 그대로
   변경 가능) - 박스 라벨과 캡션 문구가 항상 같은 이름을 쓰게 통일.
2. **2프레임 확인 스트릭이 OCR 실패로 거의 안 쌓임**: 45번의
   `_ocr_disagreement` 로직이 OCR이 `None`(글자를 못 읽음)을 반환하면
   스트릭을 초기화해버렸는데, EasyOCR이 이 표지판에서 글자를 못 읽는
   비율이 꽤 높아서(헤드리스 재생 로그 기준 상당수 프레임이 `ocr_digit=None`),
   실제로는 지속되는 오분류여도 "연속 두 번 같은 숫자"에 거의 도달하지
   못했을 가능성이 큼. **해결**: OCR이 못 읽은 프레임(`None`)에서는
   스트릭을 유지하도록 변경 (YOLO와 일치하거나 다른 숫자로 바뀔 때만
   초기화) - 정보가 없을 뿐이지 "불일치가 없어졌다"는 뜻은 아니므로.

**검증**: 같은 헤드리스 스크립트로 `Detection.class_name`과
`MatchResult.location.name`이 모든 프레임에서 항상 일치하는지 확인 -
불일치 0건. 기존 인식 위치 집합(`{logo, room2, room3, room4, rear_door}`)도
그대로 유지되어 회귀 없음.

## 47. "회의실" 표지판 오인식 필터 추가 + 진짜 근본 원인 발견: LocationLocker의 3초 잠금 (2026-09-17)

사용자가 준 실제 스냅샷(`20260917_212704_room4_score78.jpg`)을 분석하다가
새 문제를 발견: 같은 프레임에 "4 회의실(Meeting Room)" 표지판과 실제
"2 강의실(Class Room)" 표지판이 동시에 잡혔는데, **둘 다 YOLO가
`room4`(2_class/4_class)로 분류**하고 있었음. 회의실 표지판은 강의실
표지판과 생김새(검은 명판+큰 숫자+두 줄 텍스트)가 거의 똑같아서 YOLO가
구분을 못 하는 것.

**47-1 해결(회의실 필터)**: `ocr_verify.py`에 `read_sign_info()` 신설 -
기존 `read_digit()`(숫자만 추출)과 달리 한 번의 OCR 호출로 (숫자, 회의실
여부)를 같이 반환. `core.py`의 `_resolve_location()`에서 OCR 텍스트에
"회의"가 포함되면(`OcrVerifier.verify()`의 기존 판정 기준 재사용) 그
탐지를 위치 후보에서 아예 제외하도록 변경. 실제 스냅샷으로 재검증 -
회의실 표지판이 더 이상 room4로 매칭 안 됨(`BEST: None`) 확인.

**47-2. 그런데도 "아직도 4강의실로 인식한다"는 재보고 → 진짜 근본 원인
발견**: `testvideo.mp4`의 해당 구간(프레임 940 부근, 39초 지점)을
`realtime_pipeline.py`와 완전히 동일하게 `LocationLocker`까지 포함해서
재생해봄. 회의실 필터를 적용해도 여전히 문제가 있었는데, 원인은
전혀 다른 곳(`pipeline/locking.py`)에 있었음:

- 프레임 940에서 회의실 표지판은 걸러지지만, 남은 "2강의실"(YOLO
  오분류, OCR은 "2" 정확히 읽음)이 **아직 확정 전(2프레임 중 1번째
  불일치)**이라 45번 항목 로직에 따라 YOLO의 원래(틀린) 라벨 `room4`를
  그대로 돌려주고 있었음.
- 문제는 `LocationLocker`(`min_lock_seconds=3.0`) - 이 "아직 미확정"
  상태인 `room4`를 그냥 하나의 정상적인 새 위치 전환으로 받아들여서
  **3초간 화면을 잠가버림**. 그 사이(39.17s~42.17s)에 실제 정답인
  `room2`가 40.83초에 들어와도 **잠금 때문에 완전히 무시됨** - 3초
  잠금이 풀린 42.29초가 돼서야 겨우 `room2`로 전환됨(약 1.5초 지연).
  실사용 환경에서 카메라가 더 빨리 지나가면 잠금이 안 풀릴 때까지 표지판
  자체가 화면에서 사라져서 **room2가 영영 안 뜨는 경우**도 충분히 가능함.

**해결**: `_resolve_location()`에서 "아직 확정 전인 첫 번째 불일치"일 때
YOLO의 원래(불확실한) 라벨을 반환하던 것을 **`None`(모르겠음)으로
반환**하도록 변경 - 확정 전인 탐지는 애초에 위치 후보로도, `LocationLocker`의
잠금 대상으로도 들어가지 않게 함. (2번째로 같은 불일치가 재확인되면
여전히 정상적으로 정정된 위치를 반환함 - 그 경로는 안 건드림.)

**검증**: `LocationLocker`를 포함한 전체 재생(`testvideo.mp4`, 실제 fps
24 기준)으로 화면에 표시될 문구 전환 시점을 비교:
```
수정 전: ... 39.17s SHOWN=room4(오답, 확정 전인데 잠금 걸림) → 42.29s SHOWN=room2 (1.5초 지연)
수정 후: ... 40.83s SHOWN=room2 (오답 없이 곧바로 정답)
```
전체 영상 재생 결과도 `logo(1.88s) → room4(10.00s, 진짜) → room3(25.83s,
진짜) → room4(28.96s, OCR 실패로 여전히 남아있는 별개 이슈) → room2(40.83s,
이번에 고침) → rear_door(52.92s)`로, 이번에 고친 부분 외엔 회귀 없음.

**남은 별개 이슈(미해결)**: 28.96초의 `room4`는 사실 3강의실 구간에서
OCR이 "3"을 못 읽은 프레임에 YOLO의 원래 라벨이 그대로 노출된 것(45번
항목의 "정보 없음 → 스트릭 유지, 라벨은 그대로 신뢰" 경로) - 이번 47번
수정과는 다른 케이스라 손대지 않음. 필요하면 다음에 확인할 것.

## 48. distance_data로 거리 계산 공식 재검증 (2026-09-17)

18~19번 항목에서 이식한 `seojiwoo/locator.py`(핀홀 공식 기반 `_distance()`)가
실제로 `src/classroom_locator/landmark_locator/locator.py`에 그대로 반영돼
있고, `pipeline/grid_tracker.py` → `realtime_pipeline.py`/
`webapp/backend/vision_bridge.py`에 실제로 연결돼 있는지, 그리고 공식
자체가 정확한지 바탕화면 `distance_data`(파일명이 실제 거리(m))로
재검증함. `models/yolo/final_best.pt`로 15장 전체 추론 후 컨피던스
게이트 없이(원시 박스 기준) 예측 거리/실제 거리 비율을 계산:

```
front_door (n=3): 예측/실제 = 0.901 ± 0.022  (이상적 f_norm ~0.94, 현재 0.85)
rear_door  (n=5): 예측/실제 = 1.047 ± 0.037  (이상적 f_norm ~0.81)
sign       (n=6): 예측/실제 = 1.011 ± 0.078  (이상적 f_norm ~0.84, 현재값과 거의 일치)
```

표준편차가 다 작아서 **공식(1/bbox크기에 선형) 자체는 정확히 반영되어
잘 동작함**을 확인. 다만 `f_norm`을 전역 값 하나로 공유하다 보니
`front_door`만 약 10% 과소추정 오차가 있음 (사용자 판단: 오차가 크지
않고 `near_m=3.0` 판정에 실질 영향이 제한적이라 지금은 보정 보류).

**별개로 발견한, 공식보다 더 중요한 문제**: 실제 운영값인
`sign_conf=0.65`(표지판 전용 컨피던스 임계값, `grid_tracker.py`가 별도
kwargs를 안 넘겨서 `Locator` 기본값 그대로 씀) 기준으로는, 표지판이
0.9m보다 멀면 컨피던스가 0.03~0.19까지 떨어져서 **거의 탐지 자체가
안 됨** (`sign_0.9.jpg`만 conf=0.72로 통과, 나머지 6장은 0.65 미달).
즉 거리 공식은 맞지만 표지판은 근접 판정(`near_m=3.0`)까지 갈 일이
컨피던스 게이트에 막혀 거의 없음 - 14번 항목에서 이미 확인된 "표지판
소객체 탐지 recall이 낮다"는 문제와 같은 근본 원인. 사용자 판단: 이번엔
원인 파악까지만 하고 `sign_conf` 조정이나 재학습 등 조치는 보류
(장기적으로는 표지판 소객체 학습 데이터를 더 확보해 재학습하는 게
근본 해결책으로 보임).

## 49. "2강의실은 인식되는데 지도 위치가 안 바뀐다" 원인 규명 + min_votes 3으로 하향 (2026-09-17)

사용자가 실사용 중 room2(2강의실)만 화면 박스로는 잡히는데 격자 지도
위치가 안 바뀌는 문제를 보고함. `testvideo.mp4`(바탕화면)로
`LandmarkGridPositionTracker.update_from_frame()`을 프레임 단위로 그대로
재생하며 `Locator.buf`(최근 5프레임 투표 버퍼)를 직접 찍어 원인을 확인함.

**48번 항목의 `sign_conf` 문제와는 다른, 별개의 원인**: 41.5~43.25초
구간에서 2_class가 컨피던스 0.71~0.76으로 **충분히 잘 잡혔지만**, 표지판이
화면에 머무른 시간이 약 1.7초뿐이었고 그 짧은 구간 중간에 한 프레임을
놓쳐서(42.00초) `min_votes=4`(window=5, 최근 5프레임 중 4프레임 이상 같은
랜드마크여야 확정)를 한 번도 못 채우고 매번 votes 2~3에서 리셋됨 -
`reason='votes X/5'`가 계속 4 미만에 머무는 로그로 확인. 카메라가 표지판
앞을 빠르게 지나가는 경우, 컨피던스가 충분해도 투표 안정화 문턱 자체가
너무 엄격해서 위치 갱신을 놓칠 수 있음을 실제 영상으로 확인한 것.

**해결**: 사용자 선택에 따라 [locator.py](../src/classroom_locator/landmark_locator/locator.py)의
`Locator.__init__` 기본값 `min_votes=4` → `min_votes=3`으로 하향. 이 기본값을
그대로 쓰는 `realtime_pipeline.py`/`webapp/backend/vision_bridge.py`의
`live_tracker`에 자동 반영됨. `min_votes`를 명시적으로 넘기는 곳
(`photo_tracker`의 `min_votes=1`, `scripts/run_landmark_demo.py`의 CLI
`--min_votes` 기본값 4)은 영향 없음.

**검증**: 같은 영상으로 같은 구간 재생 → 42.25초에 votes 3/5을 채워 OCR
2연속 확인까지 통과하고, 42.50초에 `current_location`이 실제로 `room2`,
`cell=(13, 0)`로 갱신되는 것을 확인함 (수정 전엔 이 구간 전체에서 room2가
한 번도 지도에 반영되지 않았음). 그 외 구간(logo/room4/room3/rear_door
전환)은 회귀 없이 그대로 동작.

**트레이드오프**: 오탐 억제력이 min_votes=4 대비 약간 줄어듦(5프레임 중
3프레임만 같아도 후보로 인정). 표지판 쪽은 `ocr_confirm_frames=2`(OCR
2연속 확인)가 별도로 남아 있어 완전히 무방비해지지는 않음. 문/로고처럼
OCR이 없는 클래스는 이 완화의 영향을 더 직접적으로 받으므로, 향후 오탐이
늘어나는지 관찰 필요.

### 49-1. min_votes만으로는 부족 → sign_conf도 0.65→0.5로 하향 (2026-09-17)

49번 수정 후에도 사용자가 "여전히 room2 위치가 지도에 안 바뀐다"고 재보고.
재현해보니 직전 검증이 실제 조건과 안 맞았던 게 원인이었음: 처음엔
디버그 스크립트를 6프레임 간격(stride=6)으로 샘플링해서 우연히 감지되는
프레임을 잘 맞혔던 것뿐이고, **실제 앱의 `process_every_n_frames: 5`
(24fps 기준 0.21초 간격)로 다시 재현하면 room2가 여전히 안 바뀜**을 확인함.

**진짜 원인**: `Locator.detect()`를 스트라이드 없이 원본 프레임 전부에
대해 돌려보니(`sign_conf=0.65` 기준), 표지판이 화면에 보이는 약 2.6초
구간(라 130프레임) 중 실제로 conf 0.65를 넘겨 잡히는 프레임이 10개
안팎(~8%)뿐이었음. 5프레임마다 1번만 보는 실제 앱의 샘플링이 하필 이
"미검출 공백" 구간에 거의 다 걸려서, `min_votes`를 아무리 낮춰도 애초에
5프레임 투표 윈도우 안에 감지 자체가 거의 안 들어옴 - 48번 항목에서
distance_data로 이미 확인했던 "표지판이 sign_conf=0.65 근처에서 불안정하게
검출된다"는 문제가 실제 영상에서도 그대로 재현된 것.

**해결**: `Locator.__init__`의 `sign_conf` 기본값을 `0.65` → `0.5`로 하향
(사용자 선택 - process_every_n_frames를 낮추는 대안은 이중 추론 중인
`realtime_pipeline.py` 구조상 연산량이 2.5배 늘어 보류). 원본 프레임
전부를 다시 찍어보니 같은 2.6초 구간에서 conf 0.5 이상으로 잡히는
프레임이 훨씬 촘촘해졌고(거의 연속), `process_every_n_frames=5` 샘플링도
5프레임 윈도우 안에서 3표 이상을 안정적으로 채움. 같은 구간 재검증 →
`t=49.17s`부터 `cur_loc=room2, cell=(13,0)`로 정상 갱신 확인, 나머지 구간도
회귀 없음.

**교훈**: 이번처럼 실제 프로덕션 스트라이드(`process_every_n_frames`)와
다른 샘플링 간격으로 재현 테스트를 하면, 우연히 감지 프레임을 더 잘
맞혀서 "고쳐진 것처럼" 보이는 거짓 양성 검증이 나올 수 있음 - 항상
`configs/default.yaml`의 실제 값(현재 5)으로 재현해서 검증할 것.

### 49-2. sign_conf 하향의 부작용: room3/room4 오탐(깜빡임) 증가 → room3 판정에 디바운스 추가 (2026-09-17)

49-1에서 `sign_conf`를 낮춘 뒤 사용자가 "room2는 이제 되는데 4강의실/
3강의실 쪽에서 오탐이 늘었다"고 재보고. 재현해보니
[grid_tracker.py](../src/classroom_locator/pipeline/grid_tracker.py)의
room3 즉시반영 경로(표결 문턱을 우회해서 OCR이 "3"을 읽은 **그 프레임
단 한 번만으로** room3로 바로 튀는 로직, 원래 있던 설계)가 원인이었음.
`sign_conf`를 낮추기 전에는 표지판 자체가 드물게 잡혀서 OCR 호출 빈도가
낮았지만, 지금은 표지판이 거의 매 분석 프레임마다 잡히면서 OCR 호출도
그만큼 잦아졌고, 그중 한 프레임이라도 "3"으로 오독하면 즉시 room3로
튀었다가 다음 프레임에 다시 room4로 돌아오는 왕복이 훨씬 잦아짐. 같은
`testvideo.mp4` 25.83~30.21초 구간을 재생해보니, 이번 수정 전엔 이
구간에서만 room3↔room4가 7번 왕복함을 확인.

**해결**: `LandmarkGridPositionTracker`에 `_room3_digit_streak`(연속으로
"3"이 읽힌 횟수) 카운터를 추가하고, room3 즉시반영 경로와 표결 통과 후
최종 위치 이름 결정 두 곳 모두 `_room3_required_streak`(연속 프레임
확인 필요 횟수) 이상 연속으로 "3"이 읽혀야만 room3로 인정하도록 변경.
이 문턱은 트래커의 `window`에 따라 다르게 둠:
- 실시간 트래커(`window=5`, 여러 프레임이 있음): `_room3_required_streak=2`
  (연속 2프레임 확인)
- `webapp/backend/vision_bridge.py`의 `photo_tracker`(`window=1`, 사진
  한 장짜리 단발 판정이라 "다음 프레임"이 없음): `_room3_required_streak=1`
  (기존과 동일하게 즉시 신뢰 - 안 그러면 사진 업로드에서 room3를 영영
  못 잡게 됨)

부가로, 기존엔 같은 프레임에 대해 OCR 숫자 판독(`_read_sign_digit`)을
즉시반영 경로와 `resolve_location_name()` 두 곳에서 각각 따로 호출해서
프레임당 OCR을 2번 부르고 있었는데, 이번에 한 번만 불러서 공용으로 쓰도록
정리함(호출 횟수 절감 + 두 경로가 서로 다른 판독 결과를 쓸 가능성 제거).

**검증**: 같은 25.83~30.21초 구간 재생 → room3↔room4 왕복이 7번에서
27.92초 이후 room4로 안정되기까지 2번으로 줄어듦(완전히 0으로 없어지진
않음 - OCR 판독 자체의 노이즈가 남아있는 한 프레임 단위 미세한 흔들림은
있을 수 있음). 27.92초 이후 room4로 유지되는 건 47번 항목에서 이미 확인된
"OCR이 3을 연속으로 못 읽는 구간엔 YOLO 원래 라벨(room4)로 남는다"는
별개의 기존 이슈와 같은 현상이라 이번 수정 범위 밖(그 구간의 실제 정답은
room3). room2/rear_door 전환 등 나머지 구간은 회귀 없음.

### 49-3. room3 진입만 디바운스했더니 이탈 쪽에서 여전히 오탐 → 양방향 히스테리시스로 확장 (2026-09-17)

49-2 적용 후에도 사용자가 "아직도 오탐이 존재한다"고 재보고. 49-2의
디바운스는 **room3로 들어갈 때만**(즉시반영 경로) 연속 확인을 요구했고,
표결 통과 후 최종 이름을 정하는 쪽(`name = "room3" if room3_confirmed
else _YOLO_CLASS_TO_LOCATION_NAME.get(...)`, 옛 코드)은 매 프레임 새로
읽은 값을 바로 썼음 - 그래서 room3가 이미 확정된 상태에서 단 한 프레임만
"3"이 아니게 읽혀도(OCR 노이즈) 즉시 room4로 되돌아갔다가 다음 프레임에
다시 room3로 돌아오는 왕복이 여전히 남아있었음.

**해결**: room3 전용 스트릭(`_room3_digit_streak`)을 없애고, room2/room3/
room4 어느 쪽이든 공통으로 쓰는 `_resolved_sign_name`(확정된 이름) +
`_sign_name_streak_value`/`_sign_name_streak_count`(직전 raw_name과 그
연속 횟수)로 일반화함. 매 프레임 raw_name(YOLO 라벨, 단 OCR이 "3"을 읽으면
"room3")을 구하고, **같은 raw_name이 연속 `_sign_name_required_streak`
(실시간 트래커 2, `photo_tracker`는 여전히 1)번 나와야만** `_resolved_sign_name`을
그 값으로 갱신 - room3로 들어갈 때뿐 아니라 나갈 때도 동일하게 적용됨.
이 `resolved_sign_name`을 표결 우회 경로와 표결 통과 후 최종 이름 결정
양쪽에 공용으로 씀.

**검증**: 같은 25.83~39.79초 구간 재생 → 이전엔 room3 구간 안에서도
여러 번 흔들렸는데, 이번엔 전 구간이 room3로 안정적으로 유지되고
27.92~28.12초 사이에 room4로 딱 한 번(2틱)만 흔들렸다가 28.33초에
다시 room3로 복귀함 - 왕복이 사실상 1회로 줄어듦. room2(49.17s)/
rear_door(60.42s) 전환도 회귀 없이 그대로 동작.

**남은 한계**: 완전히 0은 아님 - OCR이 우연히 연속 2프레임 동일하게
오독하면 여전히 짧게 흔들릴 수 있음. 실사용에서 계속 거슬리면
`_sign_name_required_streak`를 2→3으로 올리는 걸 다음 단계로 고려
(단, 그만큼 진짜 전환 반응 속도도 느려짐 - 트레이드오프).

### 49-4. "확정되면 최소 1초는 유지"하는 위치 잠금 추가 (2026-09-17)

49-3 적용 후에도 사용자가 "4강의실→3강의실 전환 중 3강의실로 인식된 뒤
잠깐 4강의실로 바뀌었다가 다시 3강의실로 돌아온다"고 재보고하며, "한 번
갱신되면 최소 1초는 그 장소에 머무르게" 명시적으로 요청함. 기존
히스테리시스(49-3)는 "raw_name이 연속 2프레임 나와야 확정"까지만 다뤄서,
2연속 오독이 우연히 겹치면 여전히 통과할 수 있었음.

**구현**: `GridPositionTracker`(부모 클래스, `LandmarkGridPositionTracker`가
상속)의 `_apply()`에 위치 잠금을 추가. `pipeline/locking.py`의
`LocationLocker`(OCR 텍스트 매칭 경로, `min_lock_seconds=3.0`)와 같은
목적이지만, 격자 위치 경로는 이미 votes/히스테리시스로 어느 정도
안정화돼 있어 `MIN_LOCATION_LOCK_SECONDS = 1.0`(모듈 상수)로 짧게 둠.

처음엔 "전환된 순간부터 고정 1초"로 구현했는데, 재검증해보니 room3가
전환 후 2초 넘게 계속 재확인되고 있어도 "전환 시점 + 1초"가 지나면
잠금이 이미 풀려서 그 뒤에 들어오는 단발 오탐(room4)을 못 막는 문제가
있었음(27.92초에 여전히 통과). **그래서 같은 위치가 재확인될 때마다
`_locked_until`을 `now + min_lock_seconds`로 매번 연장하도록 수정** -
"전환 후 고정 1초"가 아니라 "그 위치가 계속 확인되고 있는 한, 마지막
확인 시점으로부터 최소 1초는 다른 곳으로 못 바뀐다"는 규칙이 됨.

`photo_tracker`(웹앱 사진 한 장 판정, `window<=1`)는 서로 무관한 사진이
매번 새로 들어오므로, 이전 사진의 잠금이 다음 사진 판정을 막지 않도록
`min_lock_seconds=0.0`으로 꺼둠 (`LandmarkGridPositionTracker.__init__`에서
`_sign_name_required_streak`를 1로 낮춘 것과 같은 이유).

**검증**: 같은 25.83~39.79초 구간 재생 → 이전엔 27.92~28.33초에 room4로
흔들리던 게, 이번엔 전 구간이 room3로 완전히 안정적으로 유지됨(흔들림
0회). room2(49.17s)/rear_door(60.42s) 전환도 잠금에 막히지 않고 정상
동작 확인 - 이는 room2/rear_door로의 전환 시점에는 이미 이전 위치(room3/
room2)가 재확인 안 된 지 오래(수 초)라 잠금이 자연히 풀려있었기 때문.
