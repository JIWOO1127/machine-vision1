# classroom_locator.landmark_locator

출처: https://github.com/JIWOO1127/machine-vision1/tree/feature/seojiwoo-detection/seojiwoo/core
(같은 조 팀원 서지우님의 작업물, `feature/seojiwoo-detection` 브랜치)

원본 코드를 우리 `classroom_locator` 패키지 규약(상대 임포트, `src/` sys.path
추가 방식)에 맞게 옮겨 붙인 버전입니다. 우리 파이프라인(`classroom_locator.pipeline`,
`LandmarkGridPositionTracker`가 내부에서 이 `Locator`를 그대로 씀)과 독립
실행 데모(`scripts/run_landmark_demo.py`) 양쪽에서 씁니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `locator.py` | 프레임 → YOLO 탐지 → 거리 추정 → 최근 N프레임 투표 → near/far/none 판정 |
| `ocr_verify.py` | 표지판이 "근접(near)"으로 판정된 순간에만 OCR로 숫자 재확인(confirm/reject/unknown) |
| `navigator.py` | 시나리오 상태 머신(find_logo→reach_wall→room4→room3→room2→rear_door) + 좌/우/직진 방향 안내 |

(2026-09-17에 "실제 제품(웹 UI/CLI 격자 지도) 어디서도 안 씀"이라는 이유로
`navigator.py`를 한 번 삭제했었으나, 2026-09-18 `scripts/run_landmark_demo.py
--nav`용으로 다시 이식함. `classroom_locator.pipeline`의 격자 지도 판정
(`LandmarkGridPositionTracker`/`_describe_position()`)은 여전히 이 상태 머신을
쓰지 않고, 매 프레임 독립적으로 어느 랜드마크가 감지됐는지만 보고 판정한다 -
`Navigator`는 순서를 강제하는 시연용 시나리오가 필요한 `run_landmark_demo.py
--nav`에서만 쓰는 별도 경로다.)

(원본의 `demo_live.py`는 `scripts/run_landmark_demo.py`로 옮겨졌습니다.)

## 우리 자체 파이프라인과의 관계 / 다른 점

`classroom_locator.pipeline`(YOLO 직접분류 + 격자 지도)도 결국 이 `Locator`를
그대로 쓰지만(`LandmarkGridPositionTracker`), 원본과 비교했을 때 참고할 만한
차이점들:

- 거리 추정: 표지판/문의 **실측 물리 크기(m)**를 코드에 직접 박아두고
  (`REAL_SIZE`), bbox가 화면 위/아래 끝에 잘리면(clipped) 높이 대신 너비를
  쓰는 보정이 있음 (핀홀 카메라 근사).
- 안정화: 최근 N프레임 다수결 투표(기본 5프레임 중 4표 이상).
- **접근 추세 확인(approach_only)**: 최근 거리가 늘고 있으면("스쳐 지나감")
  근접 판정을 안 함 — 오탐 방지 로직.
- **동선 제약(route)**: 정해진 순서의 "현재/다음" 랜드마크만 인정.
  `LandmarkGridPositionTracker`도 격자 경로(x축/y축/x=cols축) 제약으로
  결과적으로 비슷한 걸 함. 좌/우 방향 안내는 `grid_tracker.py`의
  `_direction_from_box()`가 원본 `Navigator._steer()`와 같은 임계값(0.35/0.65)으로
  독립적으로 구현함.
- OCR: 표지판이 YOLO로 near 판정된 순간에만 OCR로 숫자 재확인하는 하이브리드
  (근접 시에만 돌려서 느린 OCR 비용을 최소화). 우리 `resolve_location_name()`
  (`pipeline/grid_tracker.py`)은 이 중 `OcrVerifier.read()`만 직접 불러서
  3강의실(YOLO 클래스 없음) 정정에 씀.
- TTS(음성 안내, `pyttsx3`): `scripts/run_landmark_demo.py --tts`에서만 씀.
