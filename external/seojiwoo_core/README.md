# seojiwoo_core (이식된 외부 코드)

출처: https://github.com/JIWOO1127/machine-vision1/tree/feature/seojiwoo-detection/seojiwoo/core
(같은 조 팀원 서지우님의 작업물, `feature/seojiwoo-detection` 브랜치)

우리 프로젝트(`src/classroom_locator/`)와는 임포트 구조가 달라서(상대 경로가
아니라 같은 폴더 안 모듈을 직접 import) **의도적으로 통합하지 않고 이 폴더에
독립적으로 이식**했습니다. 우리 파이프라인과 섞어 쓰려면 별도 통합 작업이
필요합니다 (아래 "우리 프로젝트와의 관계" 참고).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `locator.py` | 프레임 → YOLO 탐지 → 거리 추정 → 최근 N프레임 투표 → near/far/none 판정 |
| `navigator.py` | `Locator` 위에 얹는 시나리오 상태 머신 (find_logo→reach_wall→room4→room3→room2→rear_door), 좌/우/직진 방향 안내 |
| `ocr_verify.py` | 표지판이 "근접(near)"으로 판정된 순간에만 OCR로 숫자 재확인(confirm/reject/unknown) |
| `demo_live.py` | 영상 파일/웹캠/폰 스트림에 판정 결과를 오버레이해서 보여주는 실행 스크립트 (TTS 음성 안내 지원) |
| `final_best_v2_with_logo.pt` | **새로 발견한 모델**: 클래스에 `logo`가 추가된 6클래스 버전. 우리 프로젝트의 `models/yolo/final_best.pt`는 이 저장소의 `v1_best.pt`와 완전히 동일(해시 일치) — 즉 우리가 갖고 있던 "final_best.pt"는 사실 구버전이었음. Navigator의 find_logo/reach_wall 단계를 쓰려면 이 파일이 필요함 |

## 실행 방법

```powershell
conda activate classroom-locator
cd external\seojiwoo_core

# 추가 의존성 (우리 프로젝트엔 없던 것들)
pip install pyttsx3 pandas

# route만 지정해서 간단히 (로고 단계 없이 바로 시작)
python demo_live.py --weights ..\..\models\yolo\final_best.pt --source "경로\testvideo.mp4" ^
    --route front_door 4_class 2_class rear_door --ocr

# 로고 찾기부터 시작하는 전체 시나리오 (6클래스 모델 필요)
python demo_live.py --weights final_best_v2_with_logo.pt --source 0 --nav --ocr --tts
```

**주의**: `--nav`(Navigator, find_logo/reach_wall 단계 포함)를 쓰려면 `logo`
클래스가 있는 `final_best_v2_with_logo.pt`를 써야 합니다. 우리 기존
`models/yolo/final_best.pt`(5클래스, logo 없음)로 `--nav`를 돌리면 로고를
영원히 못 찾아서 그 단계에서 멈춥니다. `--route`만 쓰는 (Navigator 없는)
단순 모드는 5클래스 모델로도 정상 동작합니다.

## 우리 프로젝트와의 관계 / 다른 점

- 거리 추정: 우리는 `distance_data` 사진으로 뽑은 경험적 계수를 쓰는데,
  여기는 표지판/문의 **실측 물리 크기(m)**를 코드에 직접 박아두고(`REAL_SIZE`),
  bbox가 화면 위/아래 끝에 잘리면(clipped) 높이 대신 너비를 쓰는 보정이 있음
  — 우리보다 더 정교함
- 안정화: 우리는 시간 기반(3초 잠금, `LocationLocker`)인데, 여기는
  프레임 기반 다수결 투표(최근 5프레임 중 4표 이상)
- **접근 추세 확인(approach_only)**: 최근 거리가 늘고 있으면("스쳐 지나감")
  근접 판정을 안 함 — 우리한텐 없는 오탐 방지 로직
- **동선 제약(route)**: 정해진 순서의 "현재/다음" 랜드마크만 인정 —
  우리 GridPositionTracker도 결과적으로 비슷한 걸 하지만 이쪽은 명시적
  상태 머신(Navigator)으로 방향 안내(좌/우/직진)까지 포함
- OCR: 우리는 아예 배제했는데, 여기는 **YOLO가 near로 판정한 순간에만
  OCR로 숫자 재확인**하는 하이브리드 (근접 시에만 돌려서 느린 OCR 비용을 최소화)
- TTS(음성 안내): 우리한텐 없는 기능 (`pyttsx3`)

필요하면 이 중 일부(특히 접근 추세 확인, clipped bbox 처리)를 우리
`GridPositionTracker`/`visual_localization.py`에도 반영할 수 있습니다.
