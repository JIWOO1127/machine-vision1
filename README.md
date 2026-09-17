
# ArUco Indoor Localizer & Navigation

> **ArUco fiducial marker + PnP 기반의 비학습식 실내 위치추정 및 규칙식 실내 네비게이션 시스템**

이 프로젝트는 단순한 마커 검출 예제가 아니라, **실측 좌표계 · Marker Database · PnP · 월드 좌표 변환 · 다중 마커 융합 · 위치 안정화 · 영상/이미지 평가 · 실시간 네비게이션 UI**까지 하나의 흐름으로 연결한 실내 위치추정 프로젝트입니다.

---

## README 보는 방법

처음 보는 사람은 아래 순서만 보면 전체 구조를 빠르게 이해할 수 있습니다.

```text
1. 한눈에 보기
2. 가장 빠른 실행
3. 전체 시스템 Mermaid
4. UI 구성
5. 최신 네비게이션 설계 결정
6. 좌표 / 마커 / Fusion 요약
7. 필요한 상세 내용만 <details> 토글 열기
```

상세 구현과 기존 README의 긴 설명은 삭제하지 않고 **하단 토글 영역에 보존**했습니다.

---

## 한눈에 보기

| 항목 | 현재 구현 |
|---|---|
| 프로젝트 성격 | 비학습식 실내 위치추정 + 길안내 |
| 마커 | OpenCV ArUco |
| Pose | `solvePnP` |
| 좌표 변환 | Marker → Camera → World |
| 거리 단위 | mm |
| 위치 출력 | 카메라 광학 중심 `X / Y / Z` |
| 방향 출력 | Camera Yaw |
| 다중 마커 | Median 기반 위치 이상치 제거 + 가중 Fusion |
| 위치 안정화 | Jump Gate + Median + EMA |
| 최근접 문 | 항상 이름 + 정확한 거리 표시 |
| 실시간 지도 | 별도 XY Minimap |
| 네비게이션 | 앞문 → 4번 강의실 → 2번 강의실 → 뒷문 |
| 경유/도착 | 각 지점 중심 XY 거리 2500 mm 이내 |
| 방향 안내 | 카메라 시선 대비 목표 방향 |
| 방향 아이콘 | 직선 화살표만 사용 |
| 네비게이션 UI | Camera / Map / Control 3창 분리 |
| 마커 유실 | 마지막 정상 위치/안내 상태 유지 |
| 영상 분석 | Best Single / Weighted Fusion / EMA |
| 이미지 분석 | 단일 이미지 + 폴더 일괄 판정 |
| 평가 | No-GT 안정성 비교 + GT MAE/RMSE |
| 학습 모델 | 사용하지 않음 |

---

## 가장 빠른 실행

### 실시간 네비게이션

```powershell
python navigation_live.py --camera 1
```

### 실시간 위치추정만 실행

```powershell
python webcam_localizer.py --camera 1
```

### 연결된 카메라 확인

```powershell
python navigation_live.py --list-cameras
```

### 카메라 캘리브레이션

```powershell
python calibrate_webcam.py --camera 1
```

### 테스트 영상

```powershell
python video_localizer.py test.mp4 --preview
```

카메라 정보를 모르는 영상:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-mode all
```

### 단일 이미지

```powershell
python image_localizer.py test.jpg --no-calibration --preview
```

### 이미지 폴더 일괄 분석

```powershell
python batch_image_localizer.py "C:\이미지폴더" --no-calibration
```

### 배치 HTML 열기

```powershell
start .\image_results\batch_summary.html
```

### 결과 폴더 ZIP

```powershell
Compress-Archive -Path ".\image_results\*" -DestinationPath ".\image_results.zip" -Force
```

---

## 실행 명령어 모음

| 목적 | 명령 |
|---|---|
| 네비게이션 | `python navigation_live.py --camera 1` |
| 카메라 목록 | `python navigation_live.py --list-cameras` |
| 실시간 위치추정 | `python webcam_localizer.py --camera 1` |
| Calibration | `python calibrate_webcam.py --camera 1` |
| 영상 분석 | `python video_localizer.py test.mp4 --preview` |
| 모든 영상 프레임 분석 | `python video_localizer.py test.mp4 --preview --preview-mode all` |
| 빠른 전 프레임 분석 | `python video_localizer.py test.mp4 --preview --preview-mode all --preview-fast` |
| 실시간 재생 우선 | `python video_localizer.py test.mp4 --preview --preview-mode realtime` |
| 미지 카메라 영상 | `python video_localizer.py test.mp4 --no-calibration --preview` |
| 단일 이미지 | `python image_localizer.py test.jpg --no-calibration --preview` |
| 이미지 일괄 분석 | `python batch_image_localizer.py "C:\이미지폴더" --no-calibration` |
| 결과 비교 | `python compare_localization_results.py test_results\positions.csv` |

---

## 전체 시스템 구조

```mermaid
flowchart LR
    INPUT["Camera / Video / Image"]
    DET["ArUco Detection"]
    DB["Marker Database<br/>ID · Size · XYZ · FACE · TOP"]
    PNP["solvePnP"]
    TF["Marker → Camera → World Transform"]
    SINGLE["Marker별 Camera Pose"]
    OUTLIER["Position Outlier Rejection<br/>1500 mm"]
    FUSE["Weighted Multi-Marker Fusion"]
    XYZ["Final X · Y · Z"]
    YAW["Camera Yaw"]
    STAB["Position Stabilizer<br/>Jump Gate + Median + EMA"]
    MAP["XY Minimap"]
    DOOR["Nearest Door / Exact Distance"]
    NAV["Navigation Engine"]
    CAMUI["Camera Direction Card"]
    MAPUI["Route Progress"]
    CTRL["Navigation Control"]

    INPUT --> DET
    DET --> PNP
    DB --> PNP
    PNP --> TF
    DB --> TF
    TF --> SINGLE
    SINGLE --> OUTLIER
    OUTLIER --> FUSE
    FUSE --> XYZ
    FUSE --> YAW
    XYZ --> STAB
    XYZ --> MAP
    XYZ --> DOOR
    STAB --> NAV
    YAW --> NAV
    NAV --> CAMUI
    NAV --> MAPUI
    NAV --> CTRL
```

---

## 위치추정 핵심 흐름

```mermaid
flowchart TD
    A["ArUco 4 Corner"] --> B["solvePnP"]
    B --> C["Marker→Camera Rotation / Translation"]
    C --> D["Pose Inversion"]
    E["Known Marker World Pose"] --> F["Rigid Transform Composition"]
    D --> F
    F --> G["Camera World Position"]
    G --> H["다중 마커 후보"]
    H --> I["Median 기반 Outlier 제거"]
    I --> J["Area / Reprojection Error² Weight"]
    J --> K["Weighted XYZ Fusion"]
```

### PnP 정사각형 코너 규칙

```text
TL = (-h, +h, 0)
TR = (+h, +h, 0)
BR = (+h, -h, 0)
BL = (-h, -h, 0)

h = marker_size / 2
```

---

## 월드 좌표계

| 축 | 의미 |
|---|---|
| `+X` | 평면도 오른쪽 |
| `-X` | 평면도 왼쪽 |
| `+Y` | 평면도 위쪽 |
| `-Y` | 평면도 아래쪽 |
| `+Z` | 바닥 → 천장 |
| `-Z` | 천장 → 바닥 |

```mermaid
flowchart TB
    PY["+Y"]
    O["World Origin"]
    NY["-Y"]
    NX["-X"]
    PX["+X"]
    PZ["+Z"]

    PY --- O --- NY
    NX --- O --- PX
    PZ --- O
```

> `X/Y`는 바닥 평면, `Z`는 높이입니다.

---

## Marker Database 요약

마커 로컬축 정의:

```text
local +X = RIGHT
local +Y = TOP
local +Z = FACE

RIGHT = TOP × FACE
```

| Point | ID | Size | XYZ mm | FACE | TOP |
|---|---:|---:|---|---|---|
| P01 | 25 | 134 | `(-3760, 1240, 0)` | `+Z` | `-Y` |
| P02 | 9 | 34 | `(450, 1962, 800)` | `-Y` | `-Z` |
| P03 | 13 | 134 | `(1280, 0, 1800)` | `+Y` | `+Z` |
| P04 | 14 | 134 | `(8930, 740, 0)` | `+Z` | `-Y` |
| P05 | 15 | 134 | `(10980, 0, 1800)` | `+Y` | `-X` |
| P06 | 17 | 34 | `(19195, 1962, 820)` | `-Y` | `-Z` |
| P07 | 4 | 134 | `(20440, 0, 180)` | `+Y` | `+X` |
| P08 | 2 | 134 | `(21080, 750, 0)` | `+Z` | `+Y` |
| P09 | 5 | 134 | `(20610, 3570, 0)` | `+Z` | `-X` |
| P10 | 10 | 134 | `(27660, 5862, 163)` | `-X` | `+Y` |
| P11 이동 전 | 12 | 134 | `(19475, 5752, 1250)` | `+X` | `-Y` |
| P11 이동 후 | 12 | 134 | `(19475, 6012, 1250)` | `+X` | `-Y` |
| P12 | 8 | 34 | `(20235, 6792, 1330)` | `-Y` | `+Z` |

```mermaid
flowchart LR
    B["P11 이동 전<br/>Y=5752"] -->|"+260 mm"| A["P11 이동 후<br/>Y=6012"]
```

---

## 지도 / 문 / 구조물 핵심 좌표

### 문

| 이름 | 중심 XY mm | 비고 |
|---|---:|---|
| 앞문 | `(-3760, 6792)` | 상단 벽 |
| 4번 강의실 | `(500, 0)` | 하단 벽 |
| 3번 강의실 | `(10180, 0)` | 하단 벽 |
| 2번 강의실 | `(19640, 0)` | 하단 벽 |
| 뒷문 | `(20306, 6792)` | 현재 기존 쪽문 좌표 사용 |

### 주요 구조물

| 구조물 | X 범위 | Y 범위 |
|---|---:|---:|
| TABLE1 | `-940 ~ 3620` | `1962 ~ 2862` |
| water_1 | `4832 ~ 5610` | `2282 ~ 3710` |
| meeting_1 | `5610 ~ 11250` | `1962 ~ 7370` |
| meeting_2 | `12725 ~ 18875` | `1962 ~ 7380` |
| table_2 | `18875 ~ 19475` | `1962 ~ 4262` |
| water_2 | `18875 ~ 19475` | `4262 ~ 4902` |
| wall_block | `18875 ~ 19475` | `4902 ~ 6792` |
| table_3 | `21607 ~ 23707` | `5252 ~ 6125` |

> `MAP_BOUNDS` 같은 화면 렌더링 범위는 실제 벽 끝점 실측값과 같은 의미가 아닙니다.

---

## 최신 Navigation V7 UI

![Navigation V7 3-window UI](docs/navigation_v7_ui_overview.png)

### 화면을 3개로 분리한 이유

| 창 | 담당 | 이유 |
|---|---|---|
| Camera | 실제 영상 + 방향 카드 | 실제 환경 확인에 집중 |
| Minimap | 지도 + 경로 + 현재 위치 | 버튼이 지도를 가리지 않게 분리 |
| Navigation Control | 시작/종료 + 성능 + 상태 | 조작 요소를 별도 관리 |

### 직선 방향 화살표

![Navigation V7 Straight Arrows](docs/navigation_v7_straight_arrow_preview.png)

```text
↑  직진
↖  왼쪽 앞
←  왼쪽
↙  왼쪽 뒤
↓  뒤쪽
↘  오른쪽 뒤
→  오른쪽
↗  오른쪽 앞
```

둥근 화살표와 U턴 화살표는 사용하지 않습니다.

---

## 네비게이션 경로

```mermaid
flowchart LR
    A["앞문"] --> B["서쪽 통로"]
    B --> C["TABLE1 서쪽"]
    C --> D["4번 강의실"]
    D --> E["y≈1000 복도"]
    E --> F["2번 강의실"]
    F --> G["x≈20500 동쪽 통로"]
    G --> H["뒷문"]
```

세부 route centerline:

```text
(-3760,6792)
(-3760,6100)
(-1800,6100)
(-1800,1200)
(-300,1200)
(500,0)
(1300,1000)
(19640,1000)
(19640,0)
(20500,1000)
(20500,5900)
(20306,6792)
```

### 경유 판정

| 지점 | 판정 반경 |
|---|---:|
| 앞문 | 2500 mm |
| 4번 강의실 | 2500 mm |
| 2번 강의실 | 2500 mm |
| 뒷문 | 2500 mm |

> 현재 판정은 공간 polygon이 아니라 **지점 중심 좌표와 현재 XY 사이의 Euclidean distance**입니다.

---

## 네비게이션 상태

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> GoFront: 시작 / 앞문 밖
    Idle --> Go4: 시작 / 앞문 2.5m 이내

    GoFront --> Go4: 앞문 도달
    Go4 --> Go2: 4번 도달
    Go2 --> GoBack: 2번 도달
    GoBack --> Arrived: 뒷문 도달

    GoFront --> Waiting: 위치 인식 실패
    Go4 --> Waiting: 위치 인식 실패
    Go2 --> Waiting: 위치 인식 실패
    GoBack --> Waiting: 위치 인식 실패

    Waiting --> GoFront: 위치 재입력
    Waiting --> Go4: 위치 재입력
    Waiting --> Go2: 위치 재입력
    Waiting --> GoBack: 위치 재입력

    GoFront --> Idle: 종료
    Go4 --> Idle: 종료
    Go2 --> Idle: 종료
    GoBack --> Idle: 종료
```

---

## 시선 기준 방향 안내

현재 위치의 경로 진행 방향과 카메라 Yaw를 비교합니다.

```mermaid
flowchart LR
    POS["현재 안정 XY"] --> LOOK["경로 Look-ahead Point"]
    LOOK --> BEAR["Desired Bearing"]
    YAW["Camera Yaw"] --> REL["Relative Turn"]
    BEAR --> REL
    REL --> CARD["Direction Card"]
```

화면 해석:

| 표시 | 의미 |
|---|---|
| 가운데 흰 선 | 현재 카메라 정면 |
| 노란 표시 | 가야 할 방향 |
| 초록 표시 | 실제 좌표 이동 방향 |
| 큰 직선 화살표 | 현재 시선 기준 목표 방향 |
| 텍스트 | `왼쪽으로 35° 가세요.` 등 |

예:

```text
왼쪽 35°

현재 시선 기준 왼쪽 방향으로 가세요.

왼쪽으로 35° 가세요.
```

직진:

```text
직진

현재 보고 있는 방향이 이동 방향과 거의 같습니다.

직진하세요.
```

---

## 최신 설계 변경 이력

이 부분은 단순 기능 목록이 아니라 **왜 현재 구조가 되었는지**를 기록합니다.

| 변경 | 현장 문제 | 현재 결정 |
|---|---|---|
| 역방향 경고 제거 | 위치 노이즈와 작은 이동 때문에 오작동이 잦았음 | 자동 `반대방향입니다` 판정 삭제 |
| AR 도로 제거 | 실제 바닥 원근과 맞지 않아 방향이 더 혼란스러움 | 영상 중앙을 비우고 방향 카드 사용 |
| 미니맵 버튼 제거 | 버튼이 지도를 가림 | Control 창 분리 |
| 3창 UI | 작은 화면에 정보가 겹침 | Camera / Map / Control 분리 |
| 이동 방향 문구 개선 | `회전하세요`가 실제 행동을 명확히 전달하지 못함 | `왼쪽으로 35° 가세요` 사용 |
| 직선 화살표 | U턴/둥근 화살표가 직관적이지 않음 | 모든 방향을 직선 화살표로 통일 |
| 인식 실패 Hold | 마커가 잠깐 사라질 때 화면이 초기화됨 | 마지막 위치/경로/안내 유지 |
| 진행선 역행 반영 | 지나온 색이 최고 진행률만 유지되면 실제 위치와 다름 | 현재 route projection 기준으로 다시 계산 |
| P11 이동 | 구조물 가림 | +Y 260 mm 이동 브랜치 |
| P10 주의 | 마커 휨 | 평탄 재부착 + reprojection error 중요 |

---

## 위치 안정화

```mermaid
flowchart TD
    A["New XYZ"] --> B{{"Jump threshold 이내?"}}
    B -->|Yes| C["History"]
    C --> D["Median"]
    D --> E["EMA"]
    E --> F["Stable XYZ"]

    B -->|No| G["Reject / Reacquire Candidate"]
    G --> H{{"새 위치가 연속적으로 일치?"}}
    H -->|No| I["Last Stable Position 유지"]
    H -->|Yes| J["Reacquire"]
    J --> F
```

현재 주요 기준:

| 항목 | 값/개념 |
|---|---|
| History | 5 |
| EMA alpha | 0.45 |
| 기본 XY jump | 850 mm + 속도 허용량 |
| 기본 Z jump | 600 mm + 속도 허용량 |
| Reacquire 유사 거리 | 약 900 mm |
| 인식 실패 | 마지막 출력 무기한 Hold |

---

## 다중 마커 Fusion

1. 마커별 Camera World Position 계산
2. 후보 위치 Median 계산
3. Median에서 1500 mm 이상 벗어난 후보 제거
4. 화면 마커 면적과 Reprojection Error로 가중치 계산
5. XYZ 가중 평균
6. Rotation은 가장 높은 weight 마커 사용

가중치 개념:

```text
weight ∝ image_area / reprojection_error²
```

즉:

```text
크고 선명한 마커
→ 더 높은 신뢰도

작고 기울고 재투영 오차가 큰 마커
→ 더 낮은 신뢰도
```

---

## 최근접 문 표시

네비게이션과 별개로 현재 위치에서 가장 가까운 문 이름과 거리를 항상 표시합니다.

```text
XY Euclidean Distance
```

현재 관련 문/랜드마크:

```text
앞문
4번 강의실
3번 강의실
2번 강의실
쪽문/현재 뒷문 종점
```

과거의 `3m 이내이면 near` 판단과 달리, UI에서는 **3m를 넘더라도 최근접 문 이름과 정확한 거리를 계속 표시**합니다.

---

## 영상 분석 모드

```mermaid
flowchart LR
    V["Video Frame"] --> P["Per-frame Marker Pose"]
    P --> S["Best Single"]
    P --> F["Weighted Fusion"]
    F --> E["Temporal EMA"]
    S --> CSV["positions.csv"]
    F --> CSV
    E --> CSV
```

| 방식 | 목적 |
|---|---|
| Best Single | 가장 좋은 단일 마커 기준 |
| Weighted Fusion | 여러 마커를 통합 |
| Temporal EMA | 시간축 흔들림 완화 |

### Preview Mode

| 모드 | 분석 | 재생 |
|---|---|---|
| `all` | 모든 source frame | 처리속도에 따라 느려질 수 있음 |
| `all --preview-fast` | 모든 source frame | 의도적인 대기 없음 |
| `realtime` | 처리속도가 느리면 source frame skip 가능 | 실제 시간 흐름 우선 |

> `realtime`에서 skip된 source frame은 분석/CSV 기록 대상이 아닙니다.

---

## 이미지 분석

### 단일 이미지

```powershell
python image_localizer.py test.jpg --no-calibration --preview
```

결과:

```text
annotated.png
map.png
result.json
```

### 여러 이미지

```powershell
python batch_image_localizer.py "C:\이미지폴더" --no-calibration
```

결과 구조:

```text
image_results/
├─ 0001_name/
│  ├─ annotated.png
│  ├─ map.png
│  └─ result.json
├─ batch_summary.csv
├─ batch_summary.json
├─ batch_summary.html
└─ failed_images.txt
```

---

## Ground Truth 평가

Ground Truth CSV 형식:

```text
frame,x_mm,y_mm,z_mm
```

비교:

```powershell
python compare_localization_results.py test_results\positions.csv --ground-truth gt.csv
```

주요 지표:

```text
MAE XYZ
MAE XY
RMSE XYZ
RMSE XY
Median 3D Error
P95 3D Error
```

Ground Truth가 없는 경우에는 절대 정확도 대신:

```text
coverage
frame-to-frame jitter
방법별 안정성
```

을 비교합니다.

---

## 실측 / CAD 관련 주의

실내 좌표는 다음 방식으로 구축되었습니다.

```text
3m 줄자
+ 바닥 타일
+ 상대 거리 측정
+ CAD 좌표 전사
```

따라서 다음 오차가 누적될 수 있습니다.

```mermaid
flowchart LR
    T["Tape / Tile Measurement Error"] --> R["Relative Dimension Error"]
    R --> C["CAD Transfer Error"]
    C --> M["Marker World Coordinate Error"]
    M --> P["Final Pose Bias"]
```

즉, PnP 계산이 수학적으로 안정적이어도 **Marker Database의 월드 좌표 자체가 틀리면 최종 위치에 systematic bias가 생길 수 있습니다.**

---

## 중요한 가정 / 한계

| 항목 | 현재 상태 |
|---|---|
| Marker reference | 중심 좌표로 가정 |
| Camera position | 카메라 광학 중심 |
| Body center | 직접 보정하지 않음 |
| 벽 모델 | 일부는 무한/단순화된 선 |
| 실제 벽 끝점 | 모두 실측된 것은 아님 |
| P10 | 평탄하지 않을 가능성 |
| P11 | Occlusion 때문에 이동 |
| 동적 장애물 | 사람/의자 자동 회피 없음 |
| 경유 판정 | 공간 polygon이 아니라 중심 거리 |
| 뒷문 | 현재 쪽문 좌표 사용 |
| HFOV 근사 | 실제 Calibration보다 정확도 낮음 |
| 3m near threshold | 과거 semantic용이며 현장 검증 필요 |
| 2.5m checkpoint | 현재 navigation 전용 |

---

## 프로젝트 파일 구조

```text
ArUco/
├─ aruco_world_map.py
├─ camera_calibration.py
├─ calibrate_webcam.py
├─ check_marker_database.py
├─ coordinate_system.py
├─ pose_localizer.py
├─ live_map_view.py
├─ webcam_localizer.py
├─ navigation_engine.py
├─ navigation_live.py
├─ video_localizer.py
├─ image_localizer.py
├─ batch_image_localizer.py
├─ compare_localization_results.py
├─ requirements.txt
├─ requirements_navigation.txt
├─ docs/
│  ├─ navigation_v7_ui_overview.png
│  └─ navigation_v7_straight_arrow_preview.png
└─ README.md
```

---

## Git 브랜치와 P11

```mermaid
gitGraph
    commit id: "common"
    branch p11-relocation
    checkout p11-relocation
    commit id: "P11 Y=6012"
    checkout main
    commit id: "P11 Y=5752"
```

실제 프로젝트에서는 런타임 `--p11-version` 옵션을 사용하지 않습니다.

```text
main / 이동 전 코드
→ P11 Y = 5752

p11-relocation / 이동 후 코드
→ P11 Y = 6012
```

> 팀 저장소 구조가 변경된 현재 main에서는 이 프로젝트가 `ArUco/` 폴더 아래에 위치합니다.

---

## 대표 현장 이슈와 해결

| # | 이슈 | 원인/상황 | 대응 |
|---:|---|---|---|
| 1 | 실측 누적오차 | 3m 줄자 + 상대 측정 | CAD/현장 재검증 |
| 2 | 방향 정의 오류 | 설치방향 문자열 해석 혼동 | FACE/TOP 직접 정의 |
| 3 | Marker Metadata 변경 | ID/크기/위치 수정 | DB 중앙관리 |
| 4 | P10 불안정 | 마커가 휘어짐 | 평탄 재부착 권장 |
| 5 | P11 가림 | 구조물 Occlusion | +Y 260 mm 이동 |
| 6 | 검출 실패 | 거리/픽셀/각도/조명 | 촬영조건 점검 |
| 7 | Calibration 차이 | 장비/해상도 불일치 | 동일 장비 재캘리브레이션 |
| 8 | 패키지 문제 | venv / contrib 누락 | 환경 명확화 |
| 9 | 영상 느림 | 모든 프레임 분석 | `all` / `realtime` 분리 |
| 10 | 내장캠 사용성 | 화면과 카메라 방향 문제 | 외장 웹캠 사용 |
| 11 | 결과 가독성 | 문 거리/Z 부족 | 최근접 문 + Z 강조 |
| 12 | 지도 수정 | TABLE1/앞문 좌표 보정 | Geometry 업데이트 |
| 13 | Git 충돌 | 팀 repo 구조 변경 | 최신 main 기반 통합 |
| 14 | `__pycache__` | 불필요 파일 추적 | `.gitignore` |

---

## 테스트 체크리스트

- [ ] 실제 사용 카메라와 Calibration 파일이 일치하는가
- [ ] Calibration 해상도와 실행 해상도가 같은가
- [ ] 인쇄 마커 실제 크기가 DB `size_mm`와 같은가
- [ ] 마커가 평평하게 붙어 있는가
- [ ] FACE/TOP이 실제 설치방향과 같은가
- [ ] 현재 브랜치의 P11 좌표가 현장과 맞는가
- [ ] 앞문/4번/2번/뒷문 2.5m 판정을 확인했는가
- [ ] 위치가 잠깐 끊겨도 안내가 유지되는가
- [ ] 진행선을 따라 뒤로 가면 초록색도 뒤로 줄어드는가
- [ ] 직선 방향 화살표가 시선 기준과 일치하는가
- [ ] `Navigation Control` 종료 버튼이 안내만 종료하는가
- [ ] FPS / 처리 ms가 정상 표시되는가
- [ ] P10 Reprojection Error가 비정상적으로 크지 않은가

---

# 상세 문서

아래는 기존 README에 있던 상세 설명을 삭제하지 않고 주제별로 접어 둔 영역입니다.

원하는 항목만 펼쳐 보면 됩니다.

<details>
<summary><strong>시스템 개요 · 위치추정 원리 · 좌표계</strong></summary>

<br>

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

</details>

<details>
<summary><strong>설치 · 요구환경 · 가상환경</strong></summary>

<br>

## 설치

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

</details>

<details>
<summary><strong>카메라 캘리브레이션</strong></summary>

<br>

## 카메라 캘리브레이션

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

</details>

<details>
<summary><strong>실시간 위치추정</strong></summary>

<br>

## 실시간 위치 추정

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

</details>

<details>
<summary><strong>실시간 미니맵</strong></summary>

<br>

## 실시간 미니맵

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

</details>

<details>
<summary><strong>녹화 동영상 분석</strong></summary>

<br>

## 녹화 동영상 테스트

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

</details>

<details>
<summary><strong>동영상 결과 파일 · CSV · JSON</strong></summary>

<br>

## 동영상 결과

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

</details>

<details>
<summary><strong>Best Single · Fusion · EMA 비교</strong></summary>

<br>

## 테스트 방법 비교

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

## 모델 / 방법 비교

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

</details>

<details>
<summary><strong>Pose Frames 해석</strong></summary>

<br>

## Pose Frames 해석

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

</details>

<details>
<summary><strong>프로젝트 구조 · 모듈 역할</strong></summary>

<br>

## 프로젝트 구조

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

</details>

<details>
<summary><strong>ArUco Marker Database</strong></summary>

<br>

## ArUco 데이터 관리

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

</details>

<details>
<summary><strong>정확도에 영향을 주는 요소</strong></summary>

<br>

## 정확도에 영향을 주는 요소

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

</details>

<details>
<summary><strong>시스템 특성 · 제한사항</strong></summary>

<br>

## 시스템 특성 및 제한

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

</details>

<details>
<summary><strong>기존 빠른 실행 요약</strong></summary>

<br>

## 빠른 실행 요약

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

</details>

<details>
<summary><strong>권장 테스트 시나리오</strong></summary>

<br>

## 권장 테스트 시나리오

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

</details>

<details>
<summary><strong>동영상 Preview 모드 상세</strong></summary>

<br>

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

</details>

<details>
<summary><strong>정지 이미지 판정</strong></summary>

<br>

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

</details>

<details>
<summary><strong>이미지 일괄 판정</strong></summary>

<br>

## 이미지 100장 일괄 판정

이미지 폴더 전체를 한 번에 판정하려면:

```powershell
python batch_image_localizer.py "C:\이미지폴더"
```

촬영 장비가 불명확한 이미지라면:

```powershell
python batch_image_localizer.py "C:\이미지폴더" --no-calibration
```

하위 폴더까지 포함하려면:

```powershell
python batch_image_localizer.py "C:\이미지폴더" --no-calibration --recursive
```

결과는 기본적으로 `image_results`에 저장된다.

```text
image_results/
├─ 0001_이미지명/
│  ├─ annotated.png
│  ├─ map.png
│  └─ result.json
├─ ...
├─ batch_summary.csv
├─ batch_summary.json
├─ batch_summary.html
└─ failed_images.txt
```

`batch_summary.html`을 브라우저에서 열면 각 이미지의 검출 결과, XYZ 좌표, Z 높이, 최근접 문과 거리를 100장 전체에 대해 빠르게 확인할 수 있다.

Windows에서 바로 열기:

```powershell
start .\image_results\batch_summary.html
```
---

</details>

<details>
<summary><strong>실시간 네비게이션 전체 상세</strong></summary>

<br>

## 실시간 실내 네비게이션

## 31. 네비게이션 개요

기존 ArUco 기반 위치추정 결과를 이용해 실내에서 사용자를 지정된 순서대로 안내합니다.

현재 경로:

```text
앞문 → 4번 강의실 → 2번 강의실 → 뒷문
```

현재 프로젝트에서 별도로 측정된 `뒷문` 좌표가 없기 때문에,
네비게이션의 최종 목적지는 기존 `쪽문` 좌표를 사용합니다.

```text
뒷문(현재 쪽문 좌표)
center = (20306, 6792) mm
```

네비게이션은 별도의 학습 모델을 사용하지 않습니다.

```text
ArUco Pose
→ 카메라 월드 좌표 XYZ / Yaw
→ 위치 안정화
→ 현재 위치를 경로에 투영
→ 다음 경유지/목적지 선택
→ 카메라 시선 대비 이동 방향 계산
→ 화면 방향 안내
```

```mermaid
flowchart LR
    A[ArUco Detection] --> B[Camera XYZ + Yaw]
    B --> C[Position Stabilizer]
    C --> D[Route Projection]
    D --> E[Checkpoint Decision]
    E --> F[Direction Guidance]
    F --> G[Camera UI]
    F --> H[Minimap]
    F --> I[Navigation Control]
```

---

## 32. 네비게이션 실행

외장 웹캠이 `1`번인 경우:

```powershell
python navigation_live.py --camera 1
```

카메라 번호 확인:

```powershell
python navigation_live.py --list-cameras
```

기존 캘리브레이션을 무시하고 HFOV 근사 카메라 모델을 사용하려면:

```powershell
python navigation_live.py --camera 1 --force-approx
```

한글 UI는 Pillow를 사용합니다.

```powershell
python -m pip install pillow
```

이미 설치되어 있다면 다시 설치할 필요가 없습니다.

---

## 33. 네비게이션 출발 및 경유 순서

출발점은 **앞문**입니다.

```text
앞문
 ↓
4번 강의실
 ↓
2번 강의실
 ↓
뒷문
```

네비게이션 시작 위치가 앞문 주변이 아니라면 먼저 앞문으로 이동하도록 안내합니다.

```text
먼저 앞문으로 이동하세요.
```

현재 출발/경유/도착 판정 반경:

```text
2500 mm = 2.5 m
```

즉, 각 경유지 중심의 XY 직선거리 기준 `2.5m 이내`에 들어오면 해당 지점을 통과하거나 도착한 것으로 판정합니다.

판정 순서:

```text
앞문 2.5m 이내
→ 출발

4번 강의실 2.5m 이내
→ 경유

2번 강의실 2.5m 이내
→ 경유

뒷문 2.5m 이내
→ 도착
```

---

## 34. 현재 이동 경로 중심선

실측된 지도 구조를 기준으로 사용하는 기본 경로는 다음과 같습니다.

```text
앞문
 → 서쪽 통로
 → TABLE1 서쪽
 → 4번 강의실
 → y≈1000 복도
 → 2번 강의실
 → x≈20500 동쪽 통로
 → 뒷문
```

경로는 `navigation_engine.py`의 waypoint 좌표로 정의되어 있습니다.

> 현재 경로는 등록된 벽과 구조물을 기준으로 만든 수동 경로입니다.  
> 좌표로 등록되지 않은 임시 장애물이나 사람은 자동으로 탐지하지 않습니다.

---

## 35. 급격한 위치 튐 제거

실시간 위치가 한 프레임에서 갑자기 크게 바뀌는 현상을 줄이기 위해 `PositionStabilizer`를 사용합니다.

처리 방식:

```text
새로운 XYZ 입력
 ↓
직전 안정 위치와 XY / Z 변화량 비교
 ↓
허용범위 이내
 → 정상 위치로 반영

허용범위 초과
 → 바로 반영하지 않음
 ↓
비슷한 새 위치가 여러 번 연속 검출되는지 확인
 ↓
연속 확인
 → 실제 이동으로 판단하여 위치 재획득
```

작은 흔들림은 다음을 이용해 완화합니다.

```text
최근 위치 Median
+
EMA
```

갑작스러운 값이 제거되면 화면에 다음 상태를 표시할 수 있습니다.

```text
급격한 위치 튐 제거
```

---

## 36. 위치 인식 실패 시 상태 유지

ArUco가 순간적으로 보이지 않거나 Pose 계산이 실패해도
네비게이션 화면을 즉시 초기화하지 않습니다.

유지되는 정보:

```text
마지막 정상 위치
현재 경유지
현재 경로 진행상태
현재 방향 안내
미니맵 진행선
```

다음 정상 위치가 입력될 때까지:

```text
위치 인식 대기 · 마지막 안내 상태 유지
```

상태로 대기합니다.

따라서 짧은 마커 미검출 때문에 경로 화면이 사라지지 않습니다.

---

## 37. 웹캠 방향 안내 UI

웹캠 영상 중앙을 가리는 AR 도로 방식은 사용하지 않습니다.

현재 방식은 **시선 기준 방향 카드**입니다.

카메라 화면 우측 상단에 다음 정보가 표시됩니다.

```text
현재 시선 기준 방향
목적지
목적지까지의 거리
이동 방향 설명
직접 이동 명령
수평 방향 게이지
```

예:

```text
왼쪽 35°

4번 강의실 · 5.2 m

현재 시선 기준 왼쪽 방향으로 가세요.

왼쪽으로 35° 가세요.
```

직진인 경우:

```text
직진

현재 보고 있는 방향이 이동 방향과 거의 같습니다.

직진하세요.
```

오른쪽인 경우:

```text
오른쪽 60°

현재 시선 기준 오른쪽 방향으로 가세요.

오른쪽으로 60° 가세요.
```

현재 시선의 뒤쪽에 목적지가 있는 경우에도 `회전하세요`라는 표현 대신 실제 이동 방향으로 표시합니다.

```text
왼쪽 165°

현재 시선의 왼쪽 뒤쪽 방향으로 가야 합니다.

왼쪽으로 165° 가세요.
```

---

## 38. 방향 카드의 그림 의미

방향 게이지:

```text
왼쪽 ─────────── │ ─────────── 오른쪽
                 정면
```

의미:

```text
가운데 흰 선
= 현재 카메라가 보고 있는 정면

노란 삼각형
= 가야 할 방향

초록 점
= 최근 정상 위치 변화로 계산한 실제 이동 방향
```

노란 삼각형이 가운데 흰 선과 가까우면 현재 바라보는 방향으로 이동하면 됩니다.

노란 삼각형이 왼쪽이면:

```text
현재 시선 기준 왼쪽 방향
```

오른쪽이면:

```text
현재 시선 기준 오른쪽 방향
```

입니다.

---

## 39. 직선 화살표만 사용

방향 아이콘은 **직선 화살표만 사용**합니다.

둥근 화살표나 U턴 화살표는 사용하지 않습니다.

```text
↑  직진
↖  왼쪽 앞
←  왼쪽
↙  왼쪽 뒤
↓  뒤쪽
↘  오른쪽 뒤
→  오른쪽
↗  오른쪽 앞
```

예를 들어 목표가 현재 시선의 왼쪽 뒤쪽에 있으면:

```text
↙

왼쪽 165°
왼쪽으로 165° 가세요.
```

처럼 표시합니다.

---

## 40. 네비게이션 화면 구성

네비게이션 실행 시 세 개의 창을 사용합니다.

### `ArUco Indoor Navigation`

웹캠 화면 전용입니다.

표시 정보:

```text
실제 웹캠 영상
현재 전체 경로
시선 기준 방향 카드
직선 방향 화살표
위치 인식 대기 상태
좌표 튐 제거 상태
벽/경로 경고
시작/경유/도착 이벤트
```

영상 중앙은 가능한 한 비워두어 실제 환경과 ArUco 마커를 확인하기 쉽게 구성합니다.

### `Indoor Navigation Map`

미니맵 전용 화면입니다.

표시 정보:

```text
실내 지도
전체 예정 경로
지나온 경로
현재 위치
경유지
다음 진행점
```

시작/종료 버튼은 미니맵 위에 표시하지 않습니다.

### `Navigation Control`

네비게이션 제어 및 상태 대시보드입니다.

표시 정보:

```text
현재 네비게이션 상태
전체 이동경로
현재 안내
현재 X / Y / Z
최근접 문
Frame
화면 FPS
처리시간 ms
처리 FPS
시작 / 다시 시작
네비게이션 종료
```

---

## 41. 네비게이션 시작 / 종료

`Navigation Control` 창에서 버튼을 사용합니다.

```text
[ 네비게이션 시작 / 다시 시작 ]

[ 네비게이션 종료 ]
```

버튼을 누르면 잠깐 작아지고 어두워지는 클릭 효과가 표시됩니다.

키보드:

```text
N
= 네비게이션 시작 / 다시 시작

X
= 네비게이션만 중간 종료

S
= 현재 카메라 화면 저장

Q / ESC
= 프로그램 전체 종료
```

`네비게이션 종료`는 웹캠 프로그램 자체를 종료하지 않고 안내 기능만 중단합니다.

---

## 42. 경로 진행 표시

미니맵의 전체 예정 경로와 지나온 경로는 서로 다른 색으로 표시합니다.

```text
전체 예정 경로
= 기본 경로 색

현재까지 지나온 경로
= 초록색

현재 위치
= 별도 위치점

다음 진행점
= 별도 목표점
```

진행률은 단순히 최고 진행값을 계속 누적하는 방식이 아닙니다.

현재 위치를 매번 경로 선에 투영하여 진행 위치를 다시 계산합니다.

따라서 사용자가 이동 중 뒤로 돌아가면:

```text
현재 위치가 경로의 이전 구간으로 이동
→ 초록색 진행선도 이전 위치까지 다시 감소
```

합니다.

---

## 43. 시작 / 경유 / 도착 이벤트

다음 이벤트는 웹캠 화면에 약 `2초` 동안 크게 표시합니다.

```text
네비게이션 시작
앞문 출발

앞문 도착
네비게이션 출발

4번 강의실 경유

2번 강의실 경유

뒷문 도착
안내 완료

네비게이션 종료
```

---

## 44. 벽 및 구조물 접근 경고

현재 위치가 등록된 벽 또는 구조물에 너무 가까워지면 경고합니다.

현재 기본 경고거리:

```text
650 mm
```

예:

```text
주의 · TABLE1과 너무 가깝습니다.
```

문 개구부는 벽에서 제외하여 문을 통과할 때 단순 벽 접근으로 판정하지 않도록 구성합니다.

> 이 기능은 등록된 구조물 좌표를 기준으로 계산합니다.  
> 사람, 의자, 이동식 장비 등 실시간 장애물을 카메라 영상에서 자동 인식하는 충돌 회피 기능은 아닙니다.

---

## 45. 경로 이탈 경고

현재 위치와 계획 경로의 거리가 기본 약 `1500 mm` 이상 벌어지면 경로로 복귀하도록 안내합니다.

```text
경로에서 벗어났습니다.
경로 선 쪽으로 복귀하세요.
```

---

## 46. 성능 표시

`Navigation Control` 창에서 처리 성능을 실시간으로 확인할 수 있습니다.

예:

```text
Frame 1248
화면 FPS 29.6
처리 24.3 ms
처리 FPS 41.2
```

의미:

| 항목 | 의미 |
|---|---|
| `Frame` | 현재 처리 중인 프레임 번호 |
| `화면 FPS` | 메인 루프 기준 실제 화면 갱신 속도 |
| `처리 ms` | 카메라 입력부터 위치추정, 네비게이션, 지도 생성까지 걸린 처리시간 |
| `처리 FPS` | 처리시간을 초당 프레임으로 환산한 값 |

---

## 47. P11 이동 전 / 이동 후 브랜치

P11은 장애물 가림 문제 때문에 현장에서 위치가 변경되었습니다.

이동 전:

```text
P11
ID = 12
XYZ = (19475, 5752, 1250) mm
FACE = +X
TOP = -Y
```

이동 후:

```text
P11
ID = 12
XYZ = (19475, 6012, 1250) mm
FACE = +X
TOP = -Y
```

변경량:

```text
+Y 260 mm
```

두 버전은 런타임 옵션으로 전환하지 않고 Git 브랜치별 `aruco_world_map.py`에 각각 고정된 좌표를 사용합니다.

---

## 48. 네비게이션 관련 파일

```text
navigation_engine.py
navigation_live.py
```

### `navigation_engine.py`

담당 기능:

```text
경로 waypoint
출발/경유/도착 판정
위치 튐 제거
현재 위치 경로 투영
이동 진행률 계산
시선 기준 방향 계산
실제 이동 방향 계산
벽/구조물 거리 경고
웹캠 방향 카드
미니맵 경로
Navigation Control UI
```

### `navigation_live.py`

담당 기능:

```text
웹캠 입력
ArUco 검출
PnP 위치추정
다중 마커 융합
Navigation Engine 호출
3개 화면 표시
시작/종료 입력
Frame/FPS/처리시간 측정
```

---

## 49. 네비게이션 제한사항

현재 네비게이션은 **실측 좌표와 사전에 정의한 경로**를 이용하는 방식입니다.

다음 사항을 고려해야 합니다.

- 실측/CAD 좌표 자체의 오차가 존재할 수 있음
- 카메라 캘리브레이션 오차가 위치 결과에 영향을 줌
- ArUco가 작게 보이거나 가려지면 위치 갱신이 중단될 수 있음
- 벽과 구조물은 등록된 좌표만 사용함
- 실시간 사람/물체 장애물 인식은 수행하지 않음
- 경유지 판정은 현재 `2.5m` XY 직선거리 기준
- 벽 접근 경고는 현재 `650mm` 기준
- 경로 이탈 경고는 현재 약 `1500mm` 기준
- 최종 `뒷문`은 현재 기존 `쪽문` 좌표를 사용함
- 카메라 위치는 사람 몸 중심이 아니라 카메라 광학 중심 기준임

---

## 통합 프로젝트 파일 구성

최신 프로젝트의 주요 파일은 다음과 같습니다.

```text
machine-vision1/
├─ aruco_world_map.py
├─ camera_calibration.py
├─ calibrate_webcam.py
├─ check_marker_database.py
├─ coordinate_system.py
├─ pose_localizer.py
├─ live_map_view.py
├─ webcam_localizer.py
├─ navigation_engine.py
├─ navigation_live.py
├─ video_localizer.py
├─ image_localizer.py
├─ batch_image_localizer.py
├─ compare_localization_results.py
├─ requirements.txt
├─ requirements_navigation.txt
├─ camera_calibration.npz        # 생성된 경우
└─ README.md
```

프로젝트에서는 **이 `README.md` 하나만 사용**합니다.

예전의 다음 문서는 최신 통합본에서는 제거합니다.

```text
README_NAVIGATION.md
README_NAVIGATION_V2.md
README_NAVIGATION_V3.md
README_NAVIGATION_V4.md
README_NAVIGATION_V5.md
README_NAVIGATION_V6.md
README_NAVIGATION_V7.md
```

</details>


---

## 프로젝트 한 줄 설명

> **ArUco 마커와 PnP 기반의 비학습식 실내 위치추정에, 다중 마커 융합·시간축 안정화·실측 지도·규칙식 네비게이션을 결합한 시스템입니다.**
