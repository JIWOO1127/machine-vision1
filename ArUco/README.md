# ArUco Indoor Localizer


<details open>
<summary><strong>📌 요약 보기</strong></summary>

<br>

ArUco 마커와 OpenCV `solvePnP`를 이용해 **실내에서 카메라의 위치 `(x, y, z)`와 방향을 추정하는 비학습식 위치추정 시스템**입니다.

### 핵심 기능

- 실시간 웹캠 ArUco 검출
- 마커별 카메라 위치 계산
- 다중 마커 이상치 제거 + 가중 평균
- 카메라 Yaw 계산
- 실시간 XY 미니맵
- 문/랜드마크 주변 판정
- 체스보드 카메라 캘리브레이션
- 캘리브레이션이 없는 영상의 HFOV 근사 모드
- 녹화 동영상 분석
- `Best Single / Weighted Fusion / Temporal EMA` 비교
- CSV / JSON / 결과 영상 저장
- Ground Truth 기반 MAE / RMSE / P95 평가

### 가장 빠른 실행

**실시간 위치추정**

```powershell
python webcam_localizer.py --camera 1
```

**카메라 목록 확인**

```powershell
python webcam_localizer.py --list-cameras
```

**카메라 캘리브레이션**

```powershell
python calibrate_webcam.py --camera 1
```

**녹화 영상 분석**

```powershell
python video_localizer.py test.mp4 --preview
```

**촬영 장비를 알 수 없는 영상**

```powershell
python video_localizer.py test.mp4 --no-calibration --preview
```

**결과 비교**

```powershell
python compare_localization_results.py test_results\positions.csv
```

### 전체 처리 흐름

```mermaid
flowchart LR
    A[Camera / Video] --> B[ArUco Detection]
    B --> C[Marker ID Lookup]
    C --> D[solvePnP]
    D --> E[World Coordinate Transform]
    E --> F[Per-Marker Pose]
    F --> G[Outlier Rejection]
    G --> H[Weighted Fusion]
    H --> I[Final XYZ + Yaw]
    I --> J[Live Map / Landmark Decision]
```

### 사용 기술

```text
ArUco Fiducial Marker
+ OpenCV
+ Camera Calibration
+ solvePnP
+ Coordinate Transform
+ Weighted Multi-Marker Fusion
```

사용하지 않는 기술:

```text
Deep Learning
Machine Learning
Neural Network
Model Training
```

> 더 자세한 설치 방법, 좌표계, 가중 평균 방식, 동영상 테스트, Ground Truth 평가, 파일 구조와 제한사항은 아래 **자세히 보기**를 펼쳐 확인할 수 있습니다.

</details>



<details>
<summary><strong>📖 자세히 보기</strong></summary>

<br>

ArUco 마커와 PnP 기반으로 실내에서 카메라의 위치를 추정하는 **비학습식 실내 위치추정 시스템**입니다.

딥러닝, 머신러닝, 신경망 모델을 사용하지 않으며, OpenCV의 ArUco 검출과 카메라 기하학을 이용합니다.

주요 기능은 다음과 같습니다.

- 실시간 웹캠 기반 ArUco 검출
- 마커별 카메라 월드 좌표 계산
- 여러 마커 검출 시 이상치 제거 + 가중 평균
- 카메라 방향(Yaw) 계산
- 실시간 XY 미니맵 표시
- 문/랜드마크 주변 판정
- 체스보드 기반 카메라 캘리브레이션
- 캘리브레이션이 없는 카메라의 HFOV 근사 모드
- 녹화 동영상 기반 위치 판정
- 단일 마커 / 다중 마커 융합 / 시간축 EMA 결과 비교
- 결과 영상, CSV, JSON 저장
- Ground Truth 기반 MAE / RMSE / P95 비교

---

## 1. 시스템 개요

```mermaid
flowchart LR
    A[Camera / Video] --> B[ArUco Detection]
    B --> C[Marker ID Lookup]
    C --> D[solvePnP]
    D --> E[Marker Coordinate to World Coordinate]
    E --> F[Camera Position per Marker]

    F --> G[Outlier Rejection]
    G --> H[Weighted Fusion]
    H --> I[Final XYZ Position]

    H --> J[Best Rotation Selection]
    J --> K[Camera Yaw]

    I --> L[Live XY Map]
    K --> L

    I --> M[Landmark Distance Check]
    M --> N[주변 위치 판정]
```

---

## 2. 위치 추정 방식

각 ArUco 마커는 다음 정보를 가집니다.

- Marker ID
- 실제 마커 크기(mm)
- 월드 좌표 `(x, y, z)`
- 마커 정면 방향 `FACE`
- 마커 이미지의 위쪽 방향 `TOP`

마커 로컬 좌표계는 다음과 같습니다.

```text
local +X = RIGHT
local +Y = TOP
local +Z = FACE
```

`RIGHT` 방향은 다음 식으로 계산합니다.

```text
RIGHT = TOP × FACE
```

카메라 위치 계산은 각 마커에 대해 `solvePnP()`를 수행한 뒤 월드 좌표계로 변환합니다.

---

## 3. 여러 마커가 동시에 보일 때

여러 마커에서 계산된 위치를 단순 평균하지 않습니다.

먼저 위치가 크게 벗어난 결과를 제거하고, 남은 마커에 대해 신뢰도 기반 가중 평균을 수행합니다.

```mermaid
flowchart TD
    A[여러 ArUco 마커 검출] --> B[마커별 카메라 위치 계산]
    B --> C[위치 중앙값 계산]
    C --> D{중앙값과의 거리}
    D -->|허용 범위 이내| E[사용]
    D -->|기본 1500 mm 이상| F[이상치 제거]

    E --> G[가중치 계산]
    G --> H[가중 평균]
    H --> I[최종 카메라 XYZ]
```

가중치는 기본적으로 다음 요소를 반영합니다.

```text
가중치 ∝ 마커가 화면에서 차지하는 면적 / 재투영 오차²
```

따라서:

- 화면에서 크게 보이는 마커 → 높은 가중치
- 재투영 오차가 작은 마커 → 높은 가중치
- 작게 보이거나 자세 추정이 불안정한 마커 → 낮은 가중치

위치는 여러 마커를 융합하지만, 회전행렬은 단순 평균하지 않고 신뢰도가 높은 마커의 회전을 사용합니다.

---

## 4. 월드 좌표계

본 프로젝트는 **mm 단위**를 사용합니다.

```text
+X : 평면도 기준 오른쪽
-X : 평면도 기준 왼쪽

+Y : 평면도 기준 위쪽
-Y : 평면도 기준 아래쪽

+Z : 바닥 → 천장
-Z : 천장 → 바닥
```

```mermaid
flowchart TB
    Z["+Z : Ceiling"]
    O["Origin / XY Plane"]
    NZ["-Z"]
    X["+X : Right"]
    NX["-X : Left"]
    Y["+Y : Up on Map"]
    NY["-Y : Down on Map"]

    Z --- O
    O --- NZ
    NX --- O --- X
    Y --- O --- NY
```

---

# 설치

## 5. 요구 환경

권장 환경:

```text
Python 3.10+
Windows
OpenCV Contrib
NumPy
```

필수 패키지:

```powershell
python -m pip install numpy opencv-contrib-python
```

또는:

```powershell
python -m pip install -r requirements.txt
```

설치 확인:

```powershell
python -c "import cv2, numpy; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```

마지막 값이 `True`여야 합니다.

---

## 6. 가상환경 사용

Windows PowerShell 기준:

```powershell
python -m venv .venv
```

활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

패키지 설치:

```powershell
python -m pip install -r requirements.txt
```

현재 사용 중인 Python 확인:

```powershell
python -c "import sys; print(sys.executable)"
```

---

# 카메라 캘리브레이션

## 7. 캘리브레이션 실행

카메라 고유의 초점거리와 렌즈 왜곡을 얻기 위해 체스보드 캘리브레이션을 지원합니다.

기본 설정:

```text
내부 코너 : 9 × 6
정사각형 크기 : 25 mm
해상도 : 1280 × 720
```

실행:

```powershell
python calibrate_webcam.py --camera 1
```

카메라 번호를 모를 경우:

```powershell
python calibrate_webcam.py
```

### 조작키

```text
SPACE : 현재 체스보드 프레임을 샘플로 추가
C     : 캘리브레이션 계산 및 저장
Q     : 종료
```

권장 샘플 수:

```text
15 ~ 30장
```

샘플은 화면 중앙만 찍지 말고 다음과 같이 다양하게 확보하는 것이 좋습니다.

- 좌측 / 우측
- 상단 / 하단
- 가까운 거리 / 먼 거리
- 기울어진 각도

결과 파일:

```text
camera_calibration.npz
```

채택된 체스보드 이미지는 기본적으로:

```text
calibration_samples/
```

에 저장됩니다.

> 캘리브레이션 파일은 **촬영한 카메라와 해상도에 종속적**입니다. 다른 장비로 촬영한 영상에 기존 캘리브레이션 값을 그대로 적용하면 위치 오차가 증가할 수 있습니다.

---

# 실시간 위치 추정

## 8. 연결된 카메라 확인

```powershell
python webcam_localizer.py --list-cameras
```

예:

```text
[FOUND] camera 0 ...
[FOUND] camera 1 ...
```

---

## 9. 라이브 실행

예를 들어 외장 웹캠이 `camera 1`이라면:

```powershell
python webcam_localizer.py --camera 1
```

기본 동작:

- 카메라 화면 표시
- ArUco ID 표시
- 마커별 위치 계산
- 다중 마커 위치 융합
- 현재 XYZ 표시
- 현재 방향 표시
- 실시간 XY 미니맵 표시
- 가까운 문/랜드마크 판정

카메라 창과 미니맵은 **서로 독립된 창**으로 표시됩니다.

### 노트북 화면에 맞게 축소

```powershell
python webcam_localizer.py --camera 1 --preview-width 720 --map-preview-width 650
```

이 옵션은 **표시 화면만 축소**합니다.

ArUco 검출 및 PnP 계산은 원본 카메라 프레임 해상도로 계속 수행됩니다.

---

## 10. 라이브 조작키

```text
S     : 현재 카메라 프레임 저장
I     : 현재 검출된 마커별 위치 정보 콘솔 출력
Q     : 종료
ESC   : 종료
```

---

## 11. 캘리브레이션 없이 라이브 테스트

캘리브레이션 파일을 강제로 사용하지 않고 근사 카메라 모델로 실행:

```powershell
python webcam_localizer.py --camera 1 --force-approx
```

기본 수평 화각:

```text
HFOV = 60°
```

다른 값을 사용할 경우:

```powershell
python webcam_localizer.py --camera 1 --force-approx --hfov 70
```

이 방식은 현장 기능 테스트에는 사용할 수 있지만, 실제 mm 단위 절대 위치 정확도는 직접 캘리브레이션한 경우보다 낮을 수 있습니다.

---

# 실시간 미니맵

## 12. 미니맵 표시

`live_map_view.py`는 실측 및 CAD 기반 실내 구조를 XY 평면도로 표현합니다.

표시되는 정보:

```text
구조물 / 벽 / 문
등록된 ArUco 위치
현재 카메라 위치
카메라의 XY 진행 방향
사용된 마커
위치 산포(spread)
```

```mermaid
flowchart LR
    A[World XYZ] --> B[XY Projection]
    B --> C[World-to-Pixel Transform]
    C --> D[Indoor XY Map]
    E[Camera Rotation] --> F[Yaw]
    F --> D
```

미니맵을 끄려면:

```powershell
python webcam_localizer.py --camera 1 --no-map
```

---

# 녹화 동영상 테스트

## 13. 동영상 위치 판정

테스트 동영상이 프로젝트 폴더의 `test.mp4`라면:

```powershell
python video_localizer.py test.mp4
```

실시간으로 처리 과정을 확인하려면:

```powershell
python video_localizer.py test.mp4 --preview
```

---

## 14. 촬영 장비를 알 수 없는 테스트 영상

촬영 카메라와 기존 `camera_calibration.npz`가 일치하지 않는 경우 기존 캘리브레이션을 사용하지 않는 것이 좋습니다.

```powershell
python video_localizer.py test.mp4 --no-calibration --preview
```

이 경우:

- 영상 해상도는 실제 파일에서 읽음
- 수평 화각은 기본 `60°`로 가정
- 렌즈 왜곡은 0으로 가정
- 근사 카메라 행렬을 생성하여 사용

HFOV를 알고 있다면:

```powershell
python video_localizer.py test.mp4 --no-calibration --hfov 70 --preview
```

> 촬영 장비와 카메라 내부 파라미터가 불명확한 영상은 절대 위치 정확도 평가보다 **동일 영상에 대한 알고리즘 비교**에 사용하는 것이 적절합니다.

---

## 15. 동영상 미리보기 조작

```text
SPACE       : 일시정지 / 재생
N 또는 .    : 다음 프레임 1장
[ 또는 -    : 재생속도 감소
] 또는 +    : 재생속도 증가
R           : 1.0배속
S           : 현재 판정 화면 저장
Q / ESC     : 종료
```

지원 속도:

```text
0.125x
0.25x
0.5x
1.0x
1.5x
2.0x
4.0x
8.0x
```

미리보기 화면 크기 변경:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-width 800
```

---

# 동영상 결과

## 16. 자동 생성 결과

기본적으로 다음 구조로 결과가 생성됩니다.

```text
test_results/
├─ annotated.mp4
├─ positions.csv
├─ summary.json
└─ snapshots/
```

### `annotated.mp4`

다음 정보를 영상에 표시합니다.

- 검출된 ArUco ID
- 마커별 위치
- 최종 위치
- 융합 결과
- 시간축 보정 결과
- 미니맵

### `positions.csv`

프레임별 위치 추정 결과입니다.

주요 컬럼:

```text
frame
time_sec

detected_ids
registered_ids
used_ids
rejected_ids

best_single_x_mm
best_single_y_mm
best_single_z_mm

fused_x_mm
fused_y_mm
fused_z_mm

ema_x_mm
ema_y_mm
ema_z_mm

yaw_deg
spread_mm

nearest_landmark
nearest_landmark_distance_mm
```

### `summary.json`

전체 영상에 대한 다음 정보를 저장합니다.

- 처리 프레임 수
- ArUco 검출률
- 등록 마커 검출률
- Pose 계산 성공률
- 평균 재투영 오차
- 중앙값 재투영 오차
- 위치 spread
- 사용된 마커 빈도
- 카메라 모델 방식
- HFOV 근사값

---

# 테스트 방법 비교

## 17. 세 가지 위치 결과

동영상 테스트에서는 동일 프레임에 대해 다음 세 결과를 저장합니다.

```mermaid
flowchart TD
    A[Frame] --> B[ArUco Pose Estimates]

    B --> C[Best Single]
    B --> D[Weighted Fusion]

    D --> E[Temporal EMA]

    C --> F[positions.csv]
    D --> F
    E --> F
```

### Best Single

현재 프레임에서 신뢰도가 가장 높은 단일 마커의 위치만 사용합니다.

```text
best_single_x_mm
best_single_y_mm
best_single_z_mm
```

### Weighted Fusion

여러 마커의 이상치를 제거한 뒤 가중 평균한 결과입니다.

```text
fused_x_mm
fused_y_mm
fused_z_mm
```

### Temporal EMA

Weighted Fusion 결과에 시간축 EMA를 적용합니다.

```text
ema_x_mm
ema_y_mm
ema_z_mm
```

기본 EMA 계수:

```text
alpha = 0.25
```

변경:

```powershell
python video_localizer.py test.mp4 --ema-alpha 0.4
```

---

# 모델 / 방법 비교

## 18. Ground Truth 없이 비교

```powershell
python compare_localization_results.py test_results\positions.csv
```

Ground Truth가 없을 경우 다음과 같은 값을 이용해 결과를 비교할 수 있습니다.

- Pose Coverage
- 프레임 간 위치 변화량
- Mean Step
- Median Step
- P95 Step

> 이동 중인 영상의 프레임 간 변화량은 실제 이동도 포함하므로 순수한 위치 오차와 동일한 값은 아닙니다. 정지 구간의 흔들림 비교에 특히 유용합니다.

---

## 19. Ground Truth 기반 비교

Ground Truth CSV 형식:

```csv
frame,x_mm,y_mm,z_mm
0,19195,2500,1300
1,19195,2500,1300
2,19195,2500,1300
```

실행:

```powershell
python compare_localization_results.py test_results\positions.csv --ground-truth gt.csv
```

계산 항목:

```text
MAE 3D
RMSE 3D
Median 3D Error
P95 3D Error
MAE XY
RMSE XY
```

---

# Pose Frames 해석

## 20. `Pose frames : 127 (14.8%)`

예:

```text
Pose frames : 127 (14.8%)
```

의 의미는 전체 영상 프레임 중 **유효한 위치/자세를 계산할 수 있었던 프레임이 127개이며, 전체의 14.8%라는 뜻**입니다.

Pose 계산 과정:

```mermaid
flowchart TD
    A[Video Frame] --> B{ArUco 검출?}
    B -->|No| X[Pose 없음]
    B -->|Yes| C{등록된 Marker ID?}
    C -->|No| X
    C -->|Yes| D{solvePnP 성공?}
    D -->|No| X
    D -->|Yes| E[Camera Pose]
    E --> F[Pose Frame]
```

Pose Coverage는 전체 시스템의 한 지표이며, 영상에 마커가 애초에 보이지 않은 구간이 많다면 낮게 나올 수 있습니다.

---

# 프로젝트 구조

## 21. 파일 구성

```text
aruco_indoor_localizer_v2/
├─ aruco_world_map.py
├─ camera_calibration.py
├─ calibrate_webcam.py
├─ check_marker_database.py
├─ coordinate_system.py
├─ live_map_view.py
├─ pose_localizer.py
├─ webcam_localizer.py
├─ video_localizer.py
├─ compare_localization_results.py
├─ requirements.txt
└─ README.md
```

각 파일의 역할:

| 파일 | 역할 |
|---|---|
| `aruco_world_map.py` | ArUco ID, 실제 크기, 월드 좌표, FACE/TOP 방향 관리 |
| `camera_calibration.py` | 카메라 내부 파라미터 로딩 및 HFOV 근사 모델 생성 |
| `calibrate_webcam.py` | 체스보드 기반 카메라 캘리브레이션 |
| `check_marker_database.py` | 등록된 마커 데이터 검증 |
| `coordinate_system.py` | 문/랜드마크 좌표 및 주변 판정 |
| `pose_localizer.py` | solvePnP, 월드 좌표 변환, 다중 마커 융합 |
| `live_map_view.py` | 실시간 XY 미니맵 렌더링 |
| `webcam_localizer.py` | 실시간 웹캠 위치 추정 실행 |
| `video_localizer.py` | 녹화 영상 위치 추정 및 결과 저장 |
| `compare_localization_results.py` | 위치 추정 결과 비교 |

---

## 22. 모듈 구조

```mermaid
graph TD
    WM[aruco_world_map.py]
    CC[camera_calibration.py]
    CS[coordinate_system.py]
    PL[pose_localizer.py]
    LM[live_map_view.py]

    WEB[webcam_localizer.py]
    VID[video_localizer.py]
    CMP[compare_localization_results.py]
    CAL[calibrate_webcam.py]

    WM --> PL
    CC --> PL

    WM --> WEB
    CC --> WEB
    CS --> WEB
    PL --> WEB
    LM --> WEB

    WM --> VID
    CC --> VID
    CS --> VID
    PL --> VID
    LM --> VID

    VID --> CMP
    CAL --> CC
```

---

# ArUco 데이터 관리

## 23. Marker Database

마커 정보는 `aruco_world_map.py`에서 관리합니다.

각 마커에는 다음 정보가 정의됩니다.

```text
point_name
name
description
marker_id
size_mm
x
y
z
face_world
top_world
```

마커 설치 위치나 방향을 변경한 경우 반드시 데이터베이스 좌표도 함께 수정해야 합니다.

검증:

```powershell
python check_marker_database.py
```

---

# 정확도에 영향을 주는 요소

## 24. 주요 오차 원인

본 시스템의 위치 정확도는 다음 조건에 영향을 받습니다.

### 카메라 캘리브레이션

다른 카메라의 캘리브레이션을 사용할 경우 PnP 결과에 오차가 발생할 수 있습니다.

### 마커 실제 크기

설정한 `size_mm`와 실제 출력된 마커 크기가 다르면 거리 추정 오차가 발생합니다.

### 마커 설치 방향

FACE / TOP 정보가 실제 설치 방향과 다르면 월드 좌표 변환 결과가 잘못됩니다.

### 마커 평면 상태

마커가 구겨지거나 벽에서 들떠 평평하지 않으면 4개 코너의 기하학적 관계가 변형되어 Pose 오차가 증가할 수 있습니다.

### Occlusion

중간 장애물로 마커 일부가 가려지면 검출률과 Pose 안정성이 떨어집니다.

### 작은 마커

34 mm와 같이 작은 마커는 먼 거리에서 픽셀 수가 부족해 검출과 자세 계산이 불안정해질 수 있습니다.

### 카메라와 마커의 각도

마커를 매우 비스듬하게 바라보면 코너 오차와 단일 평면 Pose의 불안정성이 커질 수 있습니다.

### 조명 / 반사

빛 반사, 어두운 환경, 모션 블러는 마커 검출률에 영향을 줍니다.

---

# 시스템 특성 및 제한

## 25. 현재 시스템의 성격

이 프로젝트는 다음과 같은 **전통적 컴퓨터 비전 기반 시스템**입니다.

```text
ArUco Fiducial Marker
        +
Camera Calibration
        +
solvePnP
        +
Coordinate Transform
        +
Multi-Marker Weighted Fusion
```

사용하지 않는 것:

```text
Deep Learning
Machine Learning
Neural Network
Model Training
```

따라서 별도의 학습 데이터나 GPU 학습 과정이 필요하지 않습니다.

---

## 26. 현재 제한사항

- 등록된 ArUco가 보이지 않으면 절대 위치를 계산할 수 없음
- 단일 평면 마커 Pose는 거리와 각도에 따라 노이즈가 발생할 수 있음
- 실측/CAD 기반 월드 좌표 자체에 오차가 있으면 결과에도 반영됨
- 카메라 위치는 사용자의 몸 중심이 아니라 **카메라 광학 중심** 기준
- 랜드마크 주변 판정은 현재 XY 평면 거리 기준
- 경로 거리나 장애물 회피 거리는 계산하지 않음
- HFOV 근사 모드는 직접 캘리브레이션보다 정확도가 낮음

---

# 빠른 실행 요약

## 27. 실시간 위치 추정

```powershell
python webcam_localizer.py --camera 1
```

카메라 번호 확인:

```powershell
python webcam_localizer.py --list-cameras
```

작은 화면:

```powershell
python webcam_localizer.py --camera 1 --preview-width 720 --map-preview-width 650
```

---

## 28. 카메라 캘리브레이션

```powershell
python calibrate_webcam.py --camera 1
```

---

## 29. 테스트 동영상

동일 카메라 + 정상 캘리브레이션:

```powershell
python video_localizer.py test.mp4 --preview
```

촬영 장비 불명:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview
```

---

## 30. 결과 비교

```powershell
python compare_localization_results.py test_results\positions.csv
```

Ground Truth 포함:

```powershell
python compare_localization_results.py test_results\positions.csv --ground-truth gt.csv
```

---

# 권장 테스트 시나리오

모델/방법 비교를 위해 하나의 테스트 영상에 다음 상황을 포함하는 것을 권장합니다.

```mermaid
flowchart LR
    A[정지 구간] --> B[단일 마커]
    B --> C[다중 마커]
    C --> D[이동 구간]
    D --> E[작은 마커]
    E --> F[비스듬한 각도]
    F --> G[일부 가려짐]
```

이렇게 촬영하면 다음 항목을 비교하기 좋습니다.

- 위치 정확도
- 검출률
- Pose Coverage
- 프레임 간 흔들림
- 다중 마커 융합 효과
- 시간축 EMA 효과
- Occlusion 영향
- 작은 마커의 인식 한계

---

## License

팀 프로젝트 정책에 맞는 라이선스를 별도로 지정할 수 있습니다.


</details>



## 동영상 미리보기 처리 방식 선택

### 1. 모든 프레임 분석 — `all`

모든 영상 프레임을 빠짐없이 ArUco/PnP 분석합니다.

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-mode all
```

특징:

```text
모든 프레임 분석
CSV에 모든 처리 프레임 기록
정확한 프레임별 비교에 적합

단점:
한 프레임 처리 시간이 영상 FPS보다 길면
미리보기 재생이 원본보다 느려질 수 있음
```

대기 없이 가능한 최대 속도로 모든 프레임을 분석하려면:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-mode all --preview-fast
```

### 2. 실제 재생시간 우선 — `realtime`

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-mode realtime
```

특징:

```text
원본 영상 재생시간을 최대한 유지
분석이 느릴 경우 중간 source frame을 자동 skip
현장에서 영상 흐름을 확인하기 좋음

주의:
skip된 프레임은 ArUco/PnP 분석 및 CSV 기록을 하지 않음
따라서 전체 프레임 정밀 비교에는 all 모드를 사용
```

`summary.json`에는 다음 값이 기록됩니다.

```text
settings.preview_mode
video.processed_frames
video.skipped_frames_realtime
video.source_frames_advanced
video.analysis_fraction_of_advanced_frames
```

모델 비교나 정량 평가에는 `all`,
실시간에 가까운 영상 확인에는 `realtime` 사용을 권장합니다.


## 정지 이미지 판정

이미지 한 장에서 ArUco 기반 위치를 판정하려면:

```powershell
python image_localizer.py test.jpg --preview
```

촬영 장비가 불명확하여 기존 캘리브레이션을 사용할 수 없다면:

```powershell
python image_localizer.py test.jpg --no-calibration --preview
```

HFOV를 알고 있다면:

```powershell
python image_localizer.py test.jpg --no-calibration --hfov 70 --preview
```

생성 결과:

```text
test_image_result/
├─ annotated.png
├─ map.png
└─ result.json
```

`annotated.png`에는 검출된 마커, 마커별 위치, Best Single, 다중 마커 융합 위치가 표시됩니다.

`map.png`에는 계산된 현재 위치와 카메라 방향이 XY 미니맵으로 표시됩니다.

`result.json`에는 검출 ID, 등록 ID, 마커별 위치, reprojection error, 가중치, 최종 위치, Yaw, landmark 판정 결과가 저장됩니다.
