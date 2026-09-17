# Faster R-CNN 객체 탐지 작업 HANDOFF

작성 시점: 2026-09-16 16:35 KST  
작업 경로: `C:\Users\한국전파진흥협회\workspace\vision`

## 1. 프로젝트 목표와 현재 결정사항

- 모델: torchvision Faster R-CNN ResNet-50 FPN
- 탐지 대상은 아래 4개 클래스만 사용한다.
  1. `2_class`
  2. `4_class`
  3. `front_door`
  4. `rear_door`
- 원본 학습 데이터의 다섯 번째 클래스 `water_dispenser`는 학습 및 추론에서 제외한다.
- OCR은 객체 탐지 모델의 재학습 대상이 아니다. 객체 탐지가 안정된 다음 별도 후처리 단계로 붙일 수 있다.
- 사용자는 추가 라벨링을 하지 않기로 결정했다. 따라서 벽 오탐은 우선 추론 임계값 조정으로 완화한다.

## 2. 최초 문제와 핵심 원인

### 최초 증상

- 기존 Faster R-CNN 체크포인트에서 `2_class`, `4_class`가 거의 탐지되지 않았다.
- 같은 데이터로 만든 YOLO 결과와 Faster R-CNN 결과 차이가 지나치게 컸다.
- 처음에는 모델 구조나 하이퍼파라미터 문제로 의심했다.

### 확인된 핵심 원인: EXIF 회전

- 주 학습 데이터의 train/valid 이미지에 EXIF orientation `6`이 들어 있었다.
- Roboflow YOLO 라벨은 사람이 보는 회전된 이미지 좌표를 기준으로 만들어졌지만, 기존 PIL 로딩 코드는 EXIF 회전을 적용하지 않았다.
- 결과적으로 이미지 픽셀과 bounding box가 서로 다른 방향을 가리킨 상태로 학습되고 있었다.
- 이 문제는 YOLO와 Faster R-CNN 성능 차이가 비정상적으로 컸던 주된 코드 원인으로 판단된다.

### 적용한 수정

- 학습 이미지 로딩 시 `ImageOps.exif_transpose()` 적용
- YOLO → COCO 변환 시 EXIF가 반영된 실제 표시 크기로 bbox 계산
- COCO annotation 생성 후 다음 항목을 자동 검증
  - 클래스 순서
  - EXIF 반영 이미지 크기
  - bbox 범위 및 양수 크기
  - category ID 범위
- 수정 전 체크포인트가 새 추론/평가에 섞이지 않도록 `exif_corrected=True` 메타데이터 저장 및 검사

검증용 이미지:

- `backend/tests/output/fixed_labels_4classes.jpg`
- `backend/tests/output/test_set_labels_4classes.jpg`

## 3. 데이터셋 상태

### 학습/검증 데이터

경로: `backend/916machine_vision 2.yolov8`

| 구분 | 이미지 | 4클래스 bbox | 빈 이미지 | 클래스별 bbox 순서: 2 / 4 / front / rear |
|---|---:|---:|---:|---:|
| train | 179 | 141 | 38 | 28 / 36 / 34 / 43 |
| valid | 45 | 40 | 5 | 5 / 11 / 10 / 14 |

- 빈 이미지는 대부분 제외 대상인 `water_dispenser`만 있던 이미지이며, 4클래스 모델에는 배경 negative로 사용된다.
- 생성 annotation:
  - `backend/annotations/916machine_4class/train_coco.json`
  - `backend/annotations/916machine_4class/valid_coco.json`

### 독립 테스트 데이터

경로: `backend/test sample.v1i.yolov8`

- 이미지: 100장
- 이미지 크기: 512×512
- 클래스별 객체 수: `2_class=27`, `4_class=25`, `front_door=27`, `rear_door=32`
- 학습/검증/테스트 이미지의 파일 해시 중복: 0
- Roboflow export 구조상 실제 평가 split 이름은 `train`이므로 평가 명령에도 `--split train`을 사용해야 한다.

## 4. 변경 및 추가한 파일

### `backend/faster_rcnn_pipeline.py`

주 학습 파이프라인이다.

- 클래스 5개 → 4개 필터링 및 연속 category ID `1..4`로 재매핑
- `water_dispenser` 제외
- EXIF 회전 보정
- COCO annotation 생성 및 시작 전 무결성 검사
- 입력 크기 기본값 `min_size=1200`, `max_size=2000`
- CUDA AMP 지원
- batch size 1 기준 learning rate `0.0025`
- `StepLR(step_size=8, gamma=0.1)`
- seed `42`
- validation loss가 아닌 COCO `mAP50-95` 기준으로 best checkpoint 선택
- 학습 기록을 `backend/results/faster_rcnn_4class_history.json`에 저장
- checkpoint에 클래스, 입력 크기, EXIF 수정 여부, best mAP 저장

### `backend/faster_rcnn_pipeline_tpu.py`

Colab TPU v5e-1용 별도 실험 파일이다.

- 기존 파이프라인의 데이터 변환/EXIF/검증/모델 코드를 재사용
- XLA device 확인
- `xm.optimizer_step()` 및 `xm.save()` 사용
- 첫 batch XLA compile 안내
- BF16은 `--bf16` 옵션으로만 제공하며 기본 비활성화
- torchvision Faster R-CNN의 NMS/ROI Align XLA 호환 오류가 나면 원인을 명확히 출력

주의: Faster R-CNN은 `torchvision::nms`, ROI Align 등의 연산 때문에 TPU에서 CPU fallback 또는 연산 오류가 발생할 수 있다. TPU라고 해서 RTX GPU보다 반드시 빠른 것은 아니며 실제 TPU 실행 검증은 아직 완료되지 않았다.

### `backend/test_inference.py`

이미지/영상 추론 파일이다.

- `--image`, `--video`, `--input-dir`, `--weights`, `--output-dir` 지원
- 4개 클래스만 표시
- 클래스별 confidence 분리
  - 숫자: `--number-confidence`, 기본 `0.1`
  - 문: `--door-confidence`, 기본 `0.65`
  - 기타 fallback: `--confidence`, 기본 `0.3`
- 벽/유리벽을 문으로 탐지하는 저신뢰도 오탐을 줄이기 위해 문 threshold를 높임
- 검은 라벨 배경, 노란색 큰 글씨, 해상도 비례 글자/박스 두께 적용
- 영상 결과를 MP4로 저장

### `backend/evaluate_faster_rcnn.py`

독립 사진 테스트셋 평가 파일이다.

- 전체 COCO 지표
  - mAP50-95
  - mAP50
  - mAP75
  - AR100
- 클래스별 COCO AP/AR
- 클래스별 precision, recall, F1, TP, FP, FN
- IoU 기반 greedy matching 혼동행렬
  - 행: 실제 클래스
  - 열: 예측 클래스
  - 표시 순서: `front_door`, `rear_door`, `room2`, `room4`
  - 평가 표시에서 `2_class→room2`, `4_class→room4`로 이름 변환
  - 마지막 `(무응답)` 열: 미탐 FN
  - 예시 자료와 형식을 맞추기 위해 background/오탐 행은 표시하지 않음
  - 행렬에 표시하지 않은 오탐도 클래스별 precision 계산에는 반영
- 생성 결과
  - `metrics.json`
  - `predictions.json`
  - `class_metrics.csv`
  - `class_metrics.png`
  - `confusion_matrix.csv`
  - `confusion_matrix.png`
- CLI 옵션 `--confidence`, `--iou-threshold` 지원
- matplotlib cache를 Windows 한글 사용자 경로가 아닌 임시 폴더에 저장하도록 수정
- 합성 예제로 오분류/오탐/미탐 계산 및 이미지 저장 검증 완료

### `backend/tests/run_inference.py`

- 기존 영상 추론에서 OCR 기반 객체 억제 로직 제거
- 객체 탐지 결과 자체를 확인하도록 변경
- 숫자 클래스에 더 낮은 confidence를 적용하도록 수정

### `.vscode/tasks.json`

추가한 VS Code task:

- `Train Faster R-CNN (4 classes)`
- `Evaluate Faster R-CNN (held-out test set)`

### 로컬 Python 환경

- 손상되어 있던 `backend/venv`를 로컬 Python 3.14.6 runtime과 연결해 복구했다.
- runtime 위치: `backend/.python-runtime/cpython-3.14.6-windows-x86_64-none`
- 설치된 torch는 CUDA를 인식하며 로컬 RTX 4050 6GB에서 smoke test를 수행했다.

## 5. 현재 학습 상태

문서 작성 시점 기준:

- 학습 기록 완료: epoch 15 / 목표 25
- 현재 best checkpoint: epoch 13
- best 경로: `backend/models/faster_rcnn_916machine_4class_best.pt`
- best epoch 13:
  - `mAP50-95 = 0.721266`
  - `mAP50 = 0.953195`
  - `exif_corrected = True`
- epoch 15는 `mAP50-95 = 0.706801`이므로 best 파일은 epoch 13 상태를 유지하고 있다.

주요 epoch 변화:

| Epoch | Train loss | mAP50-95 | mAP50 | LR |
|---:|---:|---:|---:|---:|
| 1 | 0.2707 | 0.4257 | 0.6983 | 0.0025 |
| 4 | 0.0957 | 0.6463 | 0.9657 | 0.0025 |
| 7 | 0.0574 | 0.6684 | 1.0000 | 0.0025 |
| 9 | 0.0412 | 0.7118 | 0.9752 | 0.00025 |
| 10 | 0.0333 | 0.7129 | 0.9661 | 0.00025 |
| 13 | 0.0276 | **0.7213** | 0.9532 | 0.00025 |
| 15 | 0.0252 | 0.7068 | 0.9307 | 0.00025 |

전체 로그: `backend/results/faster_rcnn_4class_history.json`

참고: mAP50은 단일 IoU 0.5 기준이고 mAP50-95는 더 엄격한 여러 IoU 기준의 평균이다. 따라서 mAP50이 조금 내려가도 mAP50-95가 올라가면 bbox 위치 정밀도가 종합적으로 개선된 것으로 해석할 수 있다.

## 6. 생성한 모델 스냅샷

| 파일 | Epoch | mAP50-95 | mAP50 | 용도 |
|---|---:|---:|---:|---|
| `faster_rcnn_916machine_4class_epoch04_video_test.pt` | 4 | 0.6463 | 0.9657 | 초기 영상 비교 |
| `faster_rcnn_916machine_4class_epoch07_video_test.pt` | 7 | 0.6684 | 1.0000 | 중간 영상 비교 |
| `faster_rcnn_916machine_4class_epoch10_video_test.pt` | 10 | 0.7129 | 0.9661 | 최근 영상/사진 평가 |
| `faster_rcnn_916machine_4class_best.pt` | 현재 13 | 0.7213 | 0.9532 | 학습 중 갱신되는 현재 best |

- 각 snapshot은 복사 시점에 원본 best와 SHA-256이 동일한지 확인했다.
- `faster_rcnn_916machine_best.pt`, `faster_rcnn_colab_best.pt` 등 과거 이름의 체크포인트는 EXIF 수정 전일 수 있으므로 최종 평가에 사용하지 않는다.
- 학습 중 best 파일은 epoch 종료 시 갱신될 수 있으므로 콘솔에 `Saved best model ...`이 완전히 출력된 후 복사해야 한다.

## 7. 영상 테스트 결과

입력 영상:

- `backend/tests/input/videos/nayeon.mp4`
- 1920×1080, 30 FPS, 859 frames, 약 28.6초

생성 결과:

- epoch 4: `backend/tests/output/epoch04/nayeon_annotated.mp4`
- epoch 7 + door threshold 0.65: `backend/tests/output/epoch07_threshold65/nayeon_annotated.mp4`
- epoch 10 + 큰 라벨: `backend/tests/output/epoch10_readable/nayeon_annotated.mp4`

영상에서 확인한 사항:

- EXIF 수정 전 모델보다 숫자 및 문 탐지가 개선됐다.
- `front_door`, `rear_door`가 벽/유리벽/사각형 경계를 문으로 탐지하는 문제가 남아 있다.
- 확인한 영상에서는 일부 벽 오탐 confidence가 약 0.36~0.43, 실제 문 검출이 약 0.78 수준으로 분리되어 있었다.
- 그래서 문 클래스만 기본 threshold를 0.65로 높였다.
- 추가 라벨링은 하지 않기로 했으므로 현재 계획은 threshold 조정과 최종 best 선택이다.

## 8. 주요 실행 명령

### 로컬 학습

VS Code에서 `Tasks: Run Task` → `Train Faster R-CNN (4 classes)`를 실행하거나:

```powershell
backend\venv\Scripts\python.exe backend\faster_rcnn_pipeline.py train `
  --dataset-root "backend\916machine_vision 2.yolov8" `
  --annotation-dir "backend\annotations\916machine_4class" `
  --model-path "backend\models\faster_rcnn_916machine_4class_best.pt" `
  --history-path "backend\results\faster_rcnn_4class_history.json" `
  --epochs 25 `
  --batch-size 1 `
  --num-workers 2 `
  --learning-rate 0.0025 `
  --min-size 1200 `
  --max-size 2000
```

### 현재 snapshot 영상 테스트 예시

```powershell
backend\venv\Scripts\python.exe backend\test_inference.py `
  --video "backend\tests\input\videos\nayeon.mp4" `
  --weights "backend\models\faster_rcnn_916machine_4class_epoch10_video_test.pt" `
  --output-dir "backend\tests\output\epoch10_readable" `
  --frame-step 3 `
  --number-confidence 0.1 `
  --door-confidence 0.65
```

`--frame-step 3`은 매 3번째 프레임만 추론하고 나머지는 원본 프레임을 기록하므로 박스가 깜빡이는 것처럼 보일 수 있다. 최종 제출 영상은 시간이 허용되면 `--frame-step 1`을 사용한다.

### 독립 사진 테스트 및 지표 생성

```powershell
backend\venv\Scripts\python.exe backend\evaluate_faster_rcnn.py `
  --dataset-root "backend\test sample.v1i.yolov8" `
  --split train `
  --weights "backend\models\faster_rcnn_916machine_4class_epoch10_video_test.pt" `
  --results-dir "backend\results\faster_rcnn_test_epoch10" `
  --confidence 0.3 `
  --iou-threshold 0.5
```

평가에서는 클래스별 공정한 비교를 위해 영상 추론의 클래스별 threshold가 아니라 모든 클래스에 동일한 confidence `0.3`을 사용한다. 따라서 영상에서 보이는 결과와 평가 수치가 정확히 동일하지 않을 수 있다.

### 테스트 이미지에 박스를 그려 저장

```powershell
backend\venv\Scripts\python.exe backend\test_inference.py `
  --input-dir "backend\test sample.v1i.yolov8\train\images" `
  --weights "backend\models\faster_rcnn_916machine_4class_epoch10_video_test.pt" `
  --output-dir "backend\tests\output\test_photos_epoch10" `
  --number-confidence 0.1 `
  --door-confidence 0.65
```

### 최종 학습 후 best 고정 복사

학습 종료 후 epoch를 확인하고 파일명에 실제 epoch를 넣는다.

```powershell
Copy-Item `
  -LiteralPath "backend\models\faster_rcnn_916machine_4class_best.pt" `
  -Destination "backend\models\faster_rcnn_916machine_4class_final_best.pt"
```

### Colab TPU v5e-1 실험

두 파일을 같은 Google Drive 폴더에 둬야 한다.

- `faster_rcnn_pipeline.py`
- `faster_rcnn_pipeline_tpu.py`

```python
%cd "/content/drive/MyDrive/현대오토에버/vision collap"
!pip install -q pycocotools

!python faster_rcnn_pipeline_tpu.py \
  --dataset-root "916machine_vision 2.yolov8" \
  --annotation-dir "annotations/916machine_4class" \
  --model-path "faster_rcnn_916machine_4class_best.pt" \
  --history-path "faster_rcnn_4class_tpu_history.json" \
  --epochs 25 \
  --batch-size 2 \
  --learning-rate 0.005
```

정상 연결 시 `Using device: xla:0; XLA device type=TPU`가 출력되어야 한다.

## 9. 현재 오류 및 주의사항

### 9.1 사진 평가가 아직 완료되지 않음

- `backend/results/faster_rcnn_test_epoch10/train_coco.json`은 생성됐다.
- 문서 작성 시점에는 `metrics.json`, `class_metrics.png`, `confusion_matrix.png`가 아직 없었다.
- GPU에서 학습 프로세스와 별도의 프로세스가 동시에 실행 중이므로 epoch 10 사진 평가가 진행 중인 것으로 보인다.
- 평가 완료 메시지와 아래 파일 생성을 확인해야 한다.
  - `metrics.json`
  - `class_metrics.csv/png`
  - `confusion_matrix.csv/png`

### 9.2 학습과 평가/영상 추론 동시 실행

- 동시에 실행하면 RTX 4050 GPU 메모리와 연산을 공유하여 둘 다 느려진다.
- 가능하면 학습 종료 후 최종 best로 영상 및 사진 평가를 한 번씩 다시 실행한다.
- 현재 실행 중인 프로세스를 자동으로 종료하지 않았다.

### 9.3 MP4가 실행 중에는 열리지 않는 현상

- OpenCV `VideoWriter`가 종료되기 전에는 MP4의 `moov atom`이 기록되지 않아 플레이어에서 열리지 않는다.
- 터미널에 `Saved ... detections to ...mp4`가 나온 다음 열어야 한다.
- 중간에 `Ctrl+C`로 종료한 파일은 재생 불가능할 수 있으므로 새 출력 폴더로 다시 실행한다.

### 9.4 문 클래스의 벽 오탐

- 모델이 문 자체뿐 아니라 세로선, 사각형, 벽/유리 경계도 문 특징으로 학습한 것으로 보인다.
- 현재 완화책: `--door-confidence 0.65`
- 오탐이 계속되면 `0.75`, 실제 문이 너무 많이 사라지면 `0.55`로 조정한다.
- 추가 라벨링은 하지 않기로 했으므로 데이터 보강 작업은 다음 계획에서 제외한다.

### 9.5 TPU 호환성

- 기존 파이프라인은 `torch.cuda.is_available()`만 확인하므로 TPU 런타임에서 CPU로 표시됐다.
- TPU는 CUDA가 아니라 XLA device이므로 별도 파일이 필요하다.
- torchvision Faster R-CNN custom op 호환 문제 때문에 TPU가 실패하거나 CPU fallback으로 더 느려질 수 있다.

### 9.6 matplotlib 경고

- Windows 한글 사용자 경로에서 font cache lock 권한 경고가 한 번 발생했다.
- `evaluate_faster_rcnn.py`가 `MPLCONFIGDIR`을 임시 폴더로 지정하도록 수정했다.

## 10. 다음 작업 순서

1. 현재 25 epoch 학습 완료를 기다린다.
2. `faster_rcnn_4class_history.json`과 best checkpoint의 epoch/mAP를 확인한다.
3. 최종 best를 `faster_rcnn_916machine_4class_final_best.pt`로 복사해 고정한다.
4. 최종 best로 영상 테스트를 다시 실행한다.
   - 빠른 확인: `--frame-step 3`
   - 최종 제출 영상: `--frame-step 1`
5. 최종 best로 100장 독립 테스트셋 평가를 실행한다.
6. 다음 파일이 정상 생성됐는지 확인한다.
   - `class_metrics.png`
   - `confusion_matrix.png`
   - `metrics.json`
7. 테스트 사진 전체에 bbox를 그린 결과를 생성하고 성공/실패 예시를 선별한다.
8. YOLO 결과와 Faster R-CNN 결과를 동일 테스트셋/동일 기준으로 비교한다.
9. 아래 PPT 구성안에 결과 이미지를 삽입한다.

## 11. PPT 작성용 구성안

### 슬라이드 1. 프로젝트 개요

- 목적: 실내 시설물 4종 객체 탐지
- 모델 비교: YOLO vs Faster R-CNN
- 최종 클래스: `2_class`, `4_class`, `front_door`, `rear_door`
- `water_dispenser` 제외 이유 명시

### 슬라이드 2. 데이터셋 구성

- train 179 / valid 45 / held-out test 100
- 클래스별 객체 수 표
- train/valid/test 해시 중복 0

### 슬라이드 3. 초기 문제

- Faster R-CNN에서 숫자 클래스가 거의 검출되지 않음
- YOLO 대비 성능이 비정상적으로 낮음
- 초기 추론 실패 이미지 삽입

### 슬라이드 4. 원인 분석

- EXIF orientation 6 설명
- 원본 픽셀 방향과 라벨 좌표가 어긋난 과정
- 수정 전/후 라벨 시각화 비교
- 추천 자료: `fixed_labels_4classes.jpg`

### 슬라이드 5. 파이프라인 개선

- EXIF transpose
- 5클래스 → 4클래스 remap
- annotation validation
- 고해상도 입력 1200/2000
- mAP50-95 기준 best 저장
- 클래스별 threshold

### 슬라이드 6. 학습 결과

- epoch별 train loss와 mAP50-95 그래프
- 현재 최고 epoch 13, mAP50-95 0.7213
- 학습 로그 원본: `faster_rcnn_4class_history.json`

### 슬라이드 7. 클래스별 성능

- 최종 평가 후 `class_metrics.png` 삽입
- 클래스별 precision/recall/F1 해석
- 숫자 클래스와 문 클래스 성능 차이 설명

### 슬라이드 8. 혼동행렬

- 최종 평가 후 `confusion_matrix.png` 삽입
- 대각선: 정분류
- 클래스 간 셀: 오분류
- `(무응답)` 열: 미탐
- 오탐은 행렬에 별도 행으로 표시하지 않지만 precision 계산에는 포함

### 슬라이드 9. 영상 추론 결과와 오류 분석

- epoch 4/7/10 영상 프레임 비교
- 숫자 탐지 개선
- 벽을 문으로 탐지하는 사례
- `door-confidence=0.65` 적용 전/후 비교
- 추천 자료: `backend/tests/output/epoch04/review_contact_sheet.jpg`

### 슬라이드 10. YOLO와 Faster R-CNN 비교

- 동일 테스트셋에서 mAP, precision, recall, F1 비교
- 속도/FPS 비교
- 모델 크기와 추론 환경 비교
- 현재 Faster R-CNN checkpoint 크기 약 330MB

### 슬라이드 11. 결론 및 한계

- 코드 문제였던 EXIF 라벨 불일치를 해결해 성능 개선
- 독립 테스트셋 평가로 일반화 성능 확인
- 문/벽 오탐은 threshold로 완화
- 추가 라벨링을 하지 않는 조건에서의 한계 명시
- TPU는 Faster R-CNN custom op 호환성 때문에 GPU보다 유리하다고 단정할 수 없음

## 12. 최종 산출물 체크리스트

- [ ] 25 epoch 학습 완료
- [ ] 최종 best checkpoint 고정 복사
- [ ] 최종 영상 MP4 생성 및 재생 확인
- [ ] held-out 100장 평가 완료
- [ ] 클래스별 precision/recall/F1 PNG 및 CSV
- [ ] confusion matrix PNG 및 CSV
- [ ] 전체 mAP 지표 JSON
- [ ] 테스트 이미지 bbox 결과
- [ ] YOLO 동일 테스트셋 지표 확보
- [ ] 학습 곡선 이미지 생성
- [ ] PPT 작성

## 13. 100 Epoch 학습 완료 업데이트 (2026-09-16 19:46 KST)

- 학습 로그: epoch 100까지 완료
- 이번 100-epoch 실행의 validation best: epoch 16
  - validation mAP50-95: `0.710237`
  - validation mAP50: `0.930693`
- epoch 100 지표:
  - train loss: `0.022752`
  - validation mAP50-95: `0.697877`
  - validation mAP50: `0.930693`
- 최종 고정 복사본:
  - `backend/models/faster_rcnn_916machine_4class_100epochs_best_epoch16.pt`
  - 원본 best checkpoint와 SHA-256 동일 확인

### 영상 테스트 완료

- 결과: `backend/tests/output/100epochs_best_epoch16/nayeon_annotated.mp4`
- 1920×1080, 30 FPS, 859 frames
- 파일 크기: 42,226,901 bytes
- OpenCV 재생/첫 프레임 읽기 검증 완료
- 총 표시 detection: 359개

### 외부 사진 100장 평가 완료

- 결과 폴더: `backend/results/faster_rcnn_test_100epochs_best_epoch16`
- 전체 mAP50-95: `0.359500`
- 전체 mAP50: `0.571095`
- 전체 mAP75: `0.405427`
- AR100: `0.417334`

| 클래스 | Precision | Recall | F1 |
|---|---:|---:|---:|
| front_door | 0.250 | 0.074 | 0.114 |
| rear_door | 0.765 | 0.812 | 0.788 |
| room2 | 1.000 | 0.370 | 0.541 |
| room4 | 0.588 | 0.800 | 0.678 |

- 혼동행렬 표시 형식은 `front_door`, `rear_door`, `room2`, `room4`, `(무응답)` 순서의 4×5 행렬이다.
- 행(Y)은 정답, 열(X)은 예측이다.
- 오탐 background 행은 표시하지 않지만 클래스별 precision 계산에는 반영한다.

### Epoch 10 스냅샷과 비교

| 모델 | 외부 test mAP50-95 | mAP50 |
|---|---:|---:|
| epoch 10 snapshot | **0.379132** | **0.596545** |
| 100-epoch run best (epoch 16) | 0.359500 | 0.571095 |

- 100 epoch까지 학습했다고 외부 일반화 성능이 더 좋아지지는 않았다.
- 전체 mAP 기준으로는 기존 epoch 10 snapshot이 더 우수하다.
- 숫자 클래스 F1은 100-epoch run best가 소폭 개선됐다.
  - room2: `0.513 → 0.541`
  - room4: `0.667 → 0.678`
- 문 클래스와 전체 mAP를 포함한 최종 선택은 epoch 10 snapshot이 더 안전하고, 숫자 클래스 우선이면 epoch 16도 후보로 비교할 수 있다.

## 14. OCR 영상 테스트 추가 (2026-09-16)

- 새 파일: `backend/services/ocr.py`
  - EasyOCR(`ko`, `en`) 래퍼
  - CUDA/CPU 자동 선택
  - 한글 표시를 위한 맑은 고딕 기반 OCR 요약 오버레이
- 수정 파일: `backend/test_inference.py`
  - `--ocr` 옵션을 줄 때만 OCR 활성화
  - `--ocr-device {auto,cpu,cuda}`
  - `--ocr-confidence` (기본값 0.3)
  - `--ocr-frame-step` (기본값 15)
  - OCR은 탐지 라벨이 그려지기 전 원본 프레임에서 수행
  - OCR을 수행하지 않는 중간 프레임에도 직전 OCR 요약을 유지
- 실행 명령:
  - `backend/venv/Scripts/python.exe backend/test_inference.py --video backend/tests/input/videos/nayeon.mp4 --weights backend/models/faster_rcnn_916machine_4class_100epochs_best_epoch16.pt --output-dir backend/tests/output/100epochs_best_epoch16_ocr --frame-step 3 --ocr --ocr-frame-step 15 --ocr-device auto`
- 결과 영상: `backend/tests/output/100epochs_best_epoch16_ocr/nayeon_annotated.mp4`
  - 1920x1080, 30 FPS, 859 frames
  - 객체 표시 359개, OCR 실행 58회
  - OpenCV로 재개방 및 프레임 수 확인 완료
  - 영상 위 한글이 깨지지 않고 표시되는 것을 프리뷰 프레임으로 확인
- OCR 관찰 결과:
  - `회의실`, `Meeting Room`, 숫자 `2`, `3`, `4` 등이 인식됨
  - 먼 거리/기울어진 표지판에서는 오인식이 있으므로 OCR 단독 위치 확정은 금지
  - 다음 단계에서 객체 탐지 결과와 OCR 숫자를 bbox 위치 기준으로 결합하고, 시간축 누적 및 지도 제약으로 현재 위치 후보를 계산해야 함

## 15. 숫자 클래스 대상 OCR 검증 방식으로 변경 (2026-09-16)

- 사용자 요구사항에 맞춰 전체 화면 OCR을 최종 판정용으로 사용하지 않고, Faster R-CNN이 `2_class` 또는 `4_class` 후보를 검출했을 때만 해당 표지판 crop에 OCR을 수행하도록 변경했다.
- 변경 파일:
  - `backend/test_inference.py`
  - `backend/services/ocr.py`
- 최종 판정 규칙:
  - 줄이 나뉘어 읽힌 `2` + `강의실`을 결합하여 `2_class`로 확정
  - `4` + `강의실`을 결합하여 `4_class`로 확정
  - 모델 클래스와 OCR 숫자가 다르면 OCR 숫자로 `2_class`/`4_class`를 교정
  - `회의실`, `회의심`, `희의실`, `최의실`, `Meeting Room` 계열이면 최종 탐지에서 제거
  - 숫자 `3`과 강의실이 읽히면 지원하지 않는 클래스이므로 2/4 후보에서 제거
  - 강의실 여부를 확인하지 못한 후보도 안전하게 제거
  - `강의신`, `강의심`, `강외실`, `강의스`, `강의설`, `경의신` 등 실제 OCR 오독을 강의실 계열로 보정
  - 같은 표지판에 2/4가 중복 검출되면 class-agnostic IoU 억제로 하나만 남김
- OCR 전처리:
  - 탐지 bbox 주변을 확장 crop
  - 2~4배 확대
  - detail enhancement 및 CLAHE 대비 강화
  - 한국어/영어 일반 OCR과 숫자 전용 OCR을 함께 수행
  - 일반 OCR에서 읽힌 숫자를 우선하고 숫자 전용 결과는 fallback으로만 사용
- 단위 검증 완료:
  - `2 + 강의실` → `2_class`
  - `2 + 회의심` → 제거
  - 모델 `4_class`, OCR `2 + 강의신` → `2_class`로 교정
  - `3 + 강의심` → 제거
  - OCR 미확인 → 제거
- 최종 영상:
  - `backend/tests/output/100epochs_best_epoch16_targeted_ocr_final/nayeon_annotated.mp4`
  - 1920x1080, 30 FPS, 859 frames, 42,431,317 bytes
  - 숫자 후보 OCR 검사 209회
  - OCR로 제거한 2/4 오탐 후보 188개
  - 문 클래스 포함 최종 표시 83개
  - `2 강의실`은 표시되고 `4 회의실`은 박스가 제거되는 프레임을 시각 검증함

## 16. Python 3.11 + PaddleOCR 전환 (2026-09-16)

- 신규 가상환경: `backend/venv311`
- 설치 및 검증 버전:
  - Python 3.11.8
  - PyTorch 2.13.0+cu132 / torchvision 0.28.0+cu132
  - Faster R-CNN CUDA: RTX 4050 정상 인식
  - PaddlePaddle 3.2.2 (Windows CPU)
  - PaddleOCR 3.7.0
  - PP-OCRv5 mobile detection + Korean mobile recognition
- 재현용 파일: `backend/requirements-faster-paddle-py311.txt`
- Paddle Windows C++ 로더가 한글 사용자 경로 아래의 모델을 열지 못하는 문제가 있어 Paddle 모델 캐시를 ASCII 경로 `C:\paddle_cache`로 고정했다.
- `backend/test_inference.py`에 `--ocr-engine {paddle,easyocr}` 옵션을 추가했고 기본값을 `paddle`로 변경했다.
- 최종 숫자 클래스 판정:
  - Faster R-CNN의 2/4 후보 bbox를 좁게 확장 crop
  - PaddleOCR로 숫자와 `강의실/회의실` 읽기
  - OCR에서 숫자 2 또는 4와 강의실이 함께 확인될 때만 유지
  - OCR 숫자가 모델과 다르면 OCR 숫자로 교정
  - 회의실, 3번 강의실, 숫자 미확인 후보는 제거
  - 표지판 위쪽의 독립 숫자를 영문 속 숫자(`C4` 등)보다 우선
- 검증 프레임 결과:
  - `3 + 강의실` → 제거
  - `2 + 회의실` → 제거
  - 모델 `4_class`, OCR `2 + 강의실` → `2_class` 교정 및 유지
- 최종 영상:
  - `backend/tests/output/100epochs_best_epoch16_paddleocr_final_v3/nayeon_annotated.mp4`
  - 숫자 후보 OCR 검사 209회
  - 제거 181개
  - 문 클래스 포함 최종 표시 90개
- 실행 명령:
  - `backend/venv311/Scripts/python.exe backend/test_inference.py --video backend/tests/input/videos/nayeon.mp4 --weights backend/models/faster_rcnn_916machine_4class_100epochs_best_epoch16.pt --output-dir backend/tests/output/100epochs_best_epoch16_paddleocr_final_v3 --frame-step 3 --ocr --ocr-engine paddle --ocr-device auto --ocr-confidence 0.25`

## 17. 미니맵 기반 현재 위치 후보 추정 (2026-09-17)

- 목표를 `탐지된 객체의 위치 = 현재 위치`가 아니라, 지도에 등록된 랜드마크 위치와 화면 속 bbox 크기/좌우 각도/시간축을 결합해 카메라의 현재 위치 후보를 추정하는 것으로 정리했다.
- 변경 파일:
  - `backend/data/minimap_coordinates.json`
  - `backend/services/visual_localization.py`
  - `backend/services/ocr.py`
  - `backend/test_inference.py`
  - `backend/test_visual_localization.py`
- 지도 설정:
  - 강의실 4, 3, 2와 앞문/쪽문 좌표를 시각 랜드마크로 사용
  - 강의실 3은 Faster R-CNN 학습 클래스가 아니지만, 2/4 후보 crop에서 OCR이 `3 + 강의실`로 읽으면 탐지 박스는 제거하고 위치 추정용 `3_ocr` 관측으로만 사용
  - 강의실 4/3/2·쪽문 구역을 지도 X 좌표 중간점 기준으로 구분
  - 벽/회의실/테이블 등 사각형 내부는 카메라 위치 후보에서 제외
  - 표지판과 문의 지도상 앞면 방향을 이용해 뒤쪽에서 보는 불가능한 후보를 제외
- 위치 계산:
  - bbox 높이 70%, 너비 30%를 사용한 핀홀 카메라 근사로 랜드마크까지 거리를 계산
  - bbox 중심 X와 카메라 수평 FOV로 좌우 bearing을 계산
  - 지도 위 250mm 격자 후보들을 거리·bearing·랜드마크 앞면·이전 프레임 위치로 점수화
  - 랜드마크 1개만 보이면 정확한 좌표라고 거짓 확정하지 않고 여러 구역 후보를 표시
  - 서로 다른 랜드마크가 2개 이상 보이거나 시간축에서 후보가 충분히 안정되면 근사 x/y/yaw를 표시
  - 영상과 단일 이미지 결과 하단에 한글 위치 패널을 표시
- 검증:
  - `backend/venv311/Scripts/python.exe -m unittest backend/test_visual_localization.py -v`
  - 3개 테스트 통과
  - 합성 입력에서 강의실 3 표지판이 약 9.68m 떨어져 보이면 현재 위치 후보 1순위 `강의실 4 앞 구역`, 2순위 `강의실 2·쪽문 구역`으로 계산됨
  - 한글 위치 패널 렌더링 확인 파일: `backend/tests/output/localization_overlay_smoke.jpg`
  - 실제 `raw_frame_385.jpg` 통합 테스트에서 강의실 3까지 약 8.5m로 추정하고, 1순위 `강의실 4 앞 구역` 57.4%, 2순위 `강의실 2·쪽문 구역` 28.9%로 계산됨
  - 실제 결과: `backend/tests/output/localization_smoke/raw_frame_385_annotated.jpg`
- 통합 테스트 중 `faster_rcnn_pipeline.py`의 `build_model()`이 추론/평가/TPU 코드의 `pretrained`, `min_size`, `max_size` 인자를 받지 못하는 호환 오류를 확인해, 기존 학습 기본값은 유지하면서 해당 인자를 받도록 수정했다.
- 중요 제한:
  - JSON의 표지판 실제 크기 600x400mm와 문 높이는 아직 실측값이 아니므로 거리에는 `(미보정)`을 표시한다.
  - 정확한 거리/좌표를 얻으려면 실제 표지판 가로·세로 크기, 영상 촬영 카메라의 실제 수평 FOV 또는 기준 거리 1개를 측정해 `minimap_coordinates.json` 값을 보정해야 한다.

## 18. 휴대폰 웹 → PC 서버 추론 연결 (2026-09-17)

- 휴대폰에는 모델을 설치하지 않고 촬영/업로드/결과 표시만 수행한다.
- PC Flask 서버가 100-epoch Faster R-CNN, PaddleOCR, 미니맵 위치 추정을 모두 실행한다.
- 변경 파일:
  - `backend/services/vision.py`: 기존 YOLO 엔진을 현재 Faster R-CNN + targeted PaddleOCR + VisualLocationEstimator 서버 엔진으로 교체
  - `backend/app.py`: 모델 상태 API, 위치 결과 반환, `frontend/dist` 정적 제공, 25MB 업로드, 단일 5000번 포트 구성
  - `backend/test_inference.py`: 웹 응답용 구조화 OCR reading 추가
  - `frontend/src/App.jsx`: 휴대폰 후면 카메라 촬영, 서버 상태, 위치 후보, OCR 텍스트, 객체, 결과 이미지 표시
  - `frontend/src/styles.css`: 모바일 위치/OCR 결과 UI
  - `README.md`: 서버 하나로 실행하는 방법으로 갱신
- 휴대폰 사진의 EXIF 회전을 서버에서 정규화한다.
- 모델 호출은 lock으로 직렬화해 동시 요청에서 PyTorch/Paddle 상태가 충돌하지 않게 했다.
- Windows 한글 사용자 경로에서 OpenCV 저장 실패를 피하기 위해 `cv2.imencode()` 후 바이트 저장 방식을 사용한다.
- 프론트엔드 `npm run build` 성공.
- Python `pip check`: 문제 없음.
- Flask test client 실제 통합 검증:
  - `/api/health`: model_ready=true, ocr_ready=true, device=cuda:0
  - `/`: 200
  - `/api/analyze`: 200
  - 실제 프레임 처리 약 1.375초
  - OCR 텍스트 `3 + 강의실 + O`, `2 + 회의실` 반환
  - 위치 후보 강의실 4 앞 57.4%, 강의실 2·쪽문 28.9%, 강의실 3 앞 13.8%
  - 결과 이미지 URL: 200
- 실행 명령:
  - `.\backend\venv311\Scripts\python.exe backend\app.py`
  - 휴대폰에서 `http://PC의-IPv4:5000` 접속

## 19. 휴대폰 영상 업로드 및 서버 분석 (2026-09-17)

- 사진 촬영 외에 휴대폰의 기존 영상 파일을 선택하는 `영상 업로드` UI를 추가했다.
- 지원 형식: MP4, MOV, AVI, MKV, WEBM, M4V.
- 제한: 최대 300MB, 최대 5분.
- 신규 API: `POST /api/analyze-video`
  - multipart 필드 `video`
  - 기본 `frame_step=5`
  - 전체 영상 길이/FPS는 유지하면서 5프레임마다 Faster R-CNN + targeted PaddleOCR + 위치 추정 실행
  - 영상 전체에서 중복 OCR을 `3번 강의실`, `2번 회의실`처럼 정규화해 횟수와 함께 요약
  - 탐지 클래스도 최고 confidence와 등장 횟수로 요약
- 결과 영상은 `imageio-ffmpeg` 번들 FFmpeg로 H.264 High Profile, yuv420p, fast-start MP4로 변환해 Android/iPhone 브라우저 재생 호환성을 확보했다.
- 원본 영상에 오디오가 있으면 AAC로 결과에 유지한다.
- 결과 URL은 HTTP Range 요청을 지원한다. 검증에서 `Range: bytes=0-1023` 요청이 206과 올바른 Content-Range를 반환했다.
- 1920x1080, 6FPS, 2초/12프레임 테스트 영상 검증:
  - 분석 API 200
  - 3개 프레임 추론, 약 4.3초 처리
  - OCR 요약: `3번 강의실` 3회, `회의실` 2회, `2번 회의실` 1회
  - 결과 MP4 GET 200, MIME `video/mp4`
  - 코덱 H.264(avc1), 픽셀 포맷 yuv420p 확인
- 변경 파일:
  - `backend/services/vision.py`
  - `backend/app.py`
  - `backend/requirements.txt`
  - `backend/requirements-faster-paddle-py311.txt`
  - `frontend/src/App.jsx`
  - `frontend/src/styles.css`
  - `README.md`

## 20. 최종 YOLO 모델·시연 로직·모바일 재생 자막 통합 (2026-09-17)

- 프로젝트 경로가 `C:\Users\한국전파진흥협회\workspace\machine-vision1`로 변경됐다.
- 사용자가 `backend/demo/demo_live.py`, `backend/demo/locator.py`를 제공했다.
- `backend/models/final_best.pt`는 확인 결과 YOLOv8 detect 모델이며 클래스는 `2_class`, `4_class`, `front_door`, `rear_door`, `water_dispenser`다.
- 서버 모델 우선순위:
  1. `backend/models/final.pt`
  2. `backend/models/final_best.pt`
  3. Faster R-CNN 체크포인트
- 현재 실제 로드 모델: `final_best.pt`, model=`YOLOv8`, device=`cuda:0`.
- 정수기 클래스는 모델에 포함돼 있어도 최종 탐지/위치 판단에서는 제외한다.
- `locator.py`의 실제 크기를 미니맵 거리 계산에 반영했다.
  - 2/3/4 강의실 표지판: 높이 200mm, 너비 130mm
  - 앞문: 높이 2500mm, 너비 4400mm
  - 뒷문: 높이 2100mm, 너비 1800mm
- 영상 API에 추론 시점별 `timeline`을 추가했다. 각 항목은 `time_seconds`, `texts`, `detections`, `location`을 반환한다.
- 모바일 UI 결과 순서를 `분석 영상 → 현재 재생 구간 OCR 자막 → 위치 → 전체 OCR → 객체`로 변경했다.
- 영상 `timeupdate/seeked` 이벤트로 재생 위치에 가장 가까운 OCR 타임라인 문구를 영상 바로 아래에 실시간 표시한다.
- 640px 이하에서는 화면 좌우 여백 제거, 카드 평면화, 영상 가로 전체 사용, 버튼 54px, safe-area 대응으로 수정했다.
- 실제 이미지 API 검증:
  - YOLOv8/final_best.pt 정상 로드
  - `3번 강의실` OCR 및 위치용 관측 반환
  - 처리 약 2.2초
- 2초 영상 API 검증:
  - 타임라인 0.000초, 0.833초, 1.667초 반환
  - 각 시점 `3번 강의실` 자막 반환
  - 결과 H.264 MP4 Range 요청 206
