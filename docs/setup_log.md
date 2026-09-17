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
- [ ] Git 저장소 초기화 및 원격 저장소 연결
