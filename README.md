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
