# ArUco Indoor Localizer V2

2026-09-16 현장 재확인 데이터를 반영한 버전입니다.

## 가장 중요한 변경점

기존 버전은 `+Y wall`, `-Y wall` 같은 설치 문자열에서 마커의
정면 방향을 추론했습니다. 이 방식 때문에 월드 방향이 잘못될 수 있었습니다.

V2는 마커마다 다음 두 방향을 직접 저장합니다.

- `face_world`: 마커 앞면이 향하는 방향 = marker local +Z
- `top_world`: ArUco 그림의 TOP이 향하는 방향 = marker local +Y
- `right_world`: `TOP × FACE`로 자동 계산 = marker local +X

## 2026-09-16 주요 수정

- P06: size 34 mm
- P10: ID 10, FACE -X, TOP +Y
- P11: ID 12, size 134 mm, FACE +X, TOP -Y
- 모든 P01~P12의 FACE/TOP 현장 확인값 반영
- P06 위치는 이전 수정값 `(19195, 1962, 820)` 유지

## 설치

```powershell
python -m pip install numpy opencv-contrib-python
```

## 데이터 확인

```powershell
python check_marker_database.py
```

## 새 웹캠 찾기

```powershell
python webcam_localizer.py --list-cameras
```

## 실행

카메라를 자동 검색하고 선택:

```powershell
python webcam_localizer.py
```

특정 카메라:

```powershell
python webcam_localizer.py --camera 1
```

카메라 캘리브레이션을 아직 하지 않았다면 자동으로 HFOV 60도 근사모드를 사용합니다.

```powershell
python webcam_localizer.py --camera 1 --hfov 60
```

실제 ArUco dictionary가 다르면 반드시 변경하세요.

```powershell
python webcam_localizer.py --camera 1 --dict DICT_5X5_100
```

## 화면에서 확인할 것

검출된 마커 옆에 다음이 표시됩니다.

- P번호
- ArUco ID
- 실제 크기
- FACE
- TOP
- RIGHT
- 그 마커 한 장으로 계산한 카메라 월드 좌표

`I`를 누르면 콘솔에 마커별 상세정보가 출력됩니다.
`S`를 누르면 현재 화면이 저장됩니다.

## 주의

캘리브레이션 파일이 없을 때 표시되는 위치는 근사값입니다.
이번 현장 점검에서는 좌표/방향 데이터 오류를 찾는 용도로 사용하고,
최종 정밀 위치검증은 카메라 캘리브레이션 이후 진행하는 것을 권장합니다.


## 새 외장 웹캠 캘리브레이션

새 웹캠은 기존 노트북 내장 웹캠과 내부 파라미터가 다르므로
새 웹캠 기준으로 다시 캘리브레이션하는 것이 좋습니다.

```powershell
python calibrate_webcam.py
```

카메라를 직접 지정하려면:

```powershell
python calibrate_webcam.py --camera 1 --cols 9 --rows 6 --square-mm 25
```

`cols`, `rows`는 체스보드의 칸 수가 아니라 **내부 교차점 수**이며,
`square-mm`는 실제 한 칸의 크기입니다.

- SPACE: 샘플 저장
- C: 8장 이상에서 계산/저장
- Q/ESC: 종료
- 15~30장의 다양한 거리/각도 샘플 권장

완료되면 프로젝트 폴더에 `camera_calibration.npz`가 저장되고,
`webcam_localizer.py`가 자동으로 이 파일을 사용합니다.

## 현장 장비 변경 배경

초기 테스트에서는 노트북 내장 웹캠을 사용했습니다.
내장 카메라는 노트북 화면과 같은 방향을 향하고 있어,
마커를 향해 노트북을 움직이면서 동시에 화면의 검출 결과를 확인하기가 어려웠습니다.

따라서 현장 검증 단계에서는 외장 웹캠을 사용하여
카메라 방향과 노트북 화면 방향을 분리했습니다.
이 변경은 마커를 촬영하면서 실시간 검출 결과와 방향을 동시에 확인하기 위한
현장 작업성 개선 목적입니다.


## 실시간 XY 지도 표시

`webcam_localizer.py` 실행 시 카메라 영상과 별도로 `Indoor XY Map` 창이 열립니다.

```powershell
python webcam_localizer.py --camera 1
```

지도에는 현재까지 제공된 실측/CAD 데이터를 기반으로 다음 항목이 표시됩니다.

- y=0 벽과 4번/3번/2번강의실 문 개구부
- y=6792 벽과 쪽문
- x=27660 동쪽 벽
- 회의실, 테이블, 정수기 등 등록 구조물
- P01~P12 ArUco 위치
- 현재 카메라 위치: 빨간 점
- 현재 카메라가 보는 XY 방향: 빨간 화살표
- 사용한 마커 P번호
- 여러 마커 결과의 spread

지도 창이 필요 없는 경우:

```powershell
python webcam_localizer.py --camera 1 --no-map
```

현재 점은 사람 몸의 중심점이 아니라 **외장 웹캠의 광학 중심 좌표**입니다.
외장 웹캠을 들고 이동하는 현재 사용 방식에서는 사용자의 대략적인 위치로 활용할 수 있습니다.


## 테스트 동영상 판정 / 모델 비교

녹화한 테스트 영상을 프레임 단위로 분석:

```powershell
python video_localizer.py test.mp4
```

결과:

```text
test_results/
├─ annotated.mp4
├─ positions.csv
└─ summary.json
```

`positions.csv`에는 동일한 프레임에 대해 세 가지 결과가 동시에 기록됩니다.

1. `best_single_*`
   - 가장 신뢰도가 높은 단일 ArUco 마커 기준

2. `fused_*`
   - 여러 마커의 이상치 제거 + 가중 평균

3. `ema_*`
   - fused 결과를 시간축 EMA로 평활화

따라서 같은 영상에서 각 방식의 안정성을 직접 비교할 수 있습니다.

처리 화면을 보면서 실행:

```powershell
python video_localizer.py test.mp4 --preview
```

EMA 반응성을 변경:

```powershell
python video_localizer.py test.mp4 --ema-alpha 0.25
```

여러 결과 CSV 비교:

```powershell
python compare_localization_results.py test_results/positions.csv
```

Ground Truth 좌표가 있는 경우 `frame,x_mm,y_mm,z_mm` 형식의 CSV를 만든 뒤:

```powershell
python compare_localization_results.py test_results/positions.csv --ground-truth gt.csv
```

로 MAE/RMSE/P95 오차까지 비교할 수 있습니다.

### 테스트 영상 촬영 권장

- 현재 캘리브레이션한 동일 외장 웹캠 사용
- 캘리브레이션과 동일한 해상도 사용
- 실제 위치를 알고 있는 지점을 몇 군데 포함
- 정지 구간과 이동 구간을 모두 촬영
- 단일 마커만 보이는 구간과 여러 마커가 동시에 보이는 구간 모두 포함
- 작은 34 mm 마커가 포함되는 구간도 별도로 촬영


### 다른 카메라로 촬영한 테스트 영상

현재 `camera_calibration.npz`와 다른 카메라로 촬영한 영상이라면
기존 캘리브레이션을 사용하지 않는 것이 좋습니다.

```powershell
python video_localizer.py test.mp4 --no-calibration
```

이 경우 영상 해상도를 기준으로 기본 수평 화각(HFOV) 60도를 가정하여
근사 카메라 행렬을 생성합니다.

다른 화각을 알고 있으면:

```powershell
python video_localizer.py test.mp4 --no-calibration --hfov 70
```

처럼 지정할 수 있습니다.

이 모드는 모델 비교/기능 테스트에는 사용할 수 있지만,
실제 mm 단위 위치 정확도는 해당 카메라를 직접 캘리브레이션한 경우보다 낮을 수 있습니다.


### 동영상 미리보기 조작

미니맵과 판정 결과를 실시간으로 보려면:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview
```

미리보기는 원본 해상도가 커도 기본 폭 1100px로 축소됩니다.

더 작게:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-width 800
```

더 크게:

```powershell
python video_localizer.py test.mp4 --no-calibration --preview --preview-width 1400
```

조작키:

- `SPACE`: 일시정지 / 재생
- `N` 또는 `.`: 일시정지 상태에서 다음 프레임 1장
- `[` 또는 `-`: 재생속도 감소
- `]` 또는 `+`: 재생속도 증가
- `R`: 1.0배속 복귀
- `S`: 현재 판정 화면 PNG 저장
- `Q` 또는 `ESC`: 종료

지원 속도 단계:

```text
0.125x / 0.25x / 0.5x / 1x / 1.5x / 2x / 4x / 8x
```

화면 오른쪽 위에는 현재 프레임 번호, 영상 시간, 재생속도,
`PLAYING`/`PAUSED` 상태가 크게 표시됩니다.

`S`로 저장한 이미지는 결과 폴더의 `snapshots` 안에 저장됩니다.


### 라이브 웹캠 화면 축소

라이브 위치 추정 화면은 표시할 때만 자동 축소됩니다.
위치 계산은 원본 카메라 프레임 해상도로 계속 수행됩니다.

기본 실행:

```powershell
python webcam_localizer.py --camera 1
```

카메라 미리보기를 더 작게:

```powershell
python webcam_localizer.py --camera 1 --preview-width 720
```

미니맵도 더 작게:

```powershell
python webcam_localizer.py --camera 1 --preview-width 720 --map-preview-width 700
```

기본값:

```text
camera preview = 960 px
map preview    = 900 px
```

두 창 모두 마우스로 직접 드래그하여 크기를 조절할 수 있습니다.
표시용 화면만 축소되므로 ArUco 검출 및 PnP 계산 해상도에는 영향을 주지 않습니다.


### 노트북 화면용 통합 라이브 창

라이브 실행 시 기본적으로 카메라 화면과 미니맵을 **하나의 창에 좌우로 합쳐서**
노트북 화면 안에 들어오도록 자동 축소합니다.

```powershell
python webcam_localizer.py --camera 1
```

기본 최대 표시 크기:

```text
1180 x 700 px
```

더 작은 노트북에서는:

```powershell
python webcam_localizer.py --camera 1 --display-width 950 --display-height 600
```

더 작게:

```powershell
python webcam_localizer.py --camera 1 --display-width 800 --display-height 520
```

예전처럼 카메라와 미니맵을 별도 창으로 띄우고 싶을 때만:

```powershell
python webcam_localizer.py --camera 1 --separate-windows
```

이 설정은 화면 표시만 축소합니다.
ArUco 검출과 PnP 위치 계산은 원본 카메라 프레임 해상도로 계속 수행됩니다.
