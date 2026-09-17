# classroom-locator 웹 데모 (webapp)

`frontend/`는 [machine-vision1](https://github.com/JIWOO1127/machine-vision1/tree/main/frontend)
저장소의 React(Vite) 앱을 그대로 가져온 것입니다(수정 없음). 백엔드는
그 저장소의 Flask 서버(`backend/app.py`)와 같은 API 모양(`/api/health`,
`/api/analyze`, `/api/analyze-frame`, `/api/reset-tracking`, ...)을 따르지만,
`webapp/backend/`에 새로 작성했고 실제 판정은 전부 이 프로젝트의
`classroom_locator.pipeline.LandmarkGridPositionTracker`(=
`classroom_locator.landmark_locator` Locator 이식 + 3강의실 OCR 정정 + 격자 스냅)를
그대로 사용합니다.

## 구성

```
webapp/
  backend/
    app.py            # Flask 서버 (라우트만 담당)
    vision_bridge.py   # 우리 LandmarkGridPositionTracker <-> 프론트가 기대하는 JSON 모양 변환
    requirements.txt   # Flask, flask-cors만 추가 (나머지는 프로젝트 루트 requirements.txt)
    results/           # 분석 결과 이미지 저장 (/api/results/<파일명>로 서빙)
  frontend/             # machine-vision1/frontend를 그대로 복사 (수정 없음)
```

## 실행

```powershell
conda activate classroom-locator
cd webapp\backend
python app.py
```

`Running on http://0.0.0.0:5000`이 뜨면 서버 준비 완료. 브라우저에서
`http://localhost:5000`으로 접속하면 (프론트를 빌드해둔 경우) 바로 웹앱이
뜹니다.

### 프론트엔드 개발 모드

화면(React 코드)을 수정하면서 확인하려면 별도로 Vite 개발 서버를 켭니다
(`vite.config.js`에 `/api` -> `http://127.0.0.1:5000` 프록시가 이미
설정되어 있어서, 백엔드는 5000번 포트에 떠 있기만 하면 됩니다):

```powershell
cd webapp\frontend
npm install
npm run dev
```

`http://localhost:5173`으로 접속해서 확인합니다.

### 프로덕션 빌드 (Flask가 직접 서빙)

```powershell
cd webapp\frontend
npm install
npm run build
```

`frontend/dist`가 생기면 `webapp/backend/app.py`가 5000번 포트에서 그대로
서빙합니다(프론트 서버를 따로 켤 필요 없음).

## API (프론트가 그대로 기대하는 모양)

| 메서드/경로 | 설명 |
|---|---|
| GET `/api/health` | `model_ready`, `device`, `location_count` 등 서버 상태 |
| POST `/api/analyze` (`image` 파일) | 사진 1장 분석. `detections`, `location`, `annotated_image_url` 반환 |
| POST `/api/analyze-frame` (`frame` 파일, `time_seconds`) | 실시간 카메라/영상 재생 중 프레임 1장 분석. 최근 5프레임 다수결로 안정화한 `cue`(거리/방향/안내문/음성안내 대상) 반환 |
| POST `/api/reset-tracking` | 실시간 분석용 프레임 버퍼 초기화 (카메라 새로 시작할 때 프론트가 자동 호출) |
| GET `/api/locations` | `configs/locations.yaml`의 위치 목록 |
| GET `/api/results/<파일명>` | 분석 결과 이미지 서빙 |
| POST `/api/analyze-video` | **아직 미구현.** 501과 함께 안내 메시지 반환 (아래 "안 된 것" 참고) |

## 사진/실시간 카메라 분석이 우리 파이프라인과 다른 점

- 사진 1장(`/api/analyze`)은 프레임 투표(temporal smoothing)가 의미 없어서
  `window=1, min_votes=1, approach_only=False`로 설정한 별도
  `LandmarkGridPositionTracker`(`photo_tracker`)를 씀 — 그 한 장만 보고 바로
  판정.
- 실시간 카메라(`/api/analyze-frame`)는 `realtime_pipeline.py`와 동일한
  기본값(최근 5프레임 중 4표, 접근추세 확인, 3강의실 OCR 정정)의
  `live_tracker`를 프레임이 올 때마다 계속 재사용 — 브라우저가 200ms마다
  (초당 5프레임) 프레임을 보내는 구조라 원본 저장소의 5프레임 다수결 설계와
  정확히 맞아떨어짐.
- `location.candidates`/`landmarks`의 방향("왼쪽"/"오른쪽"/"앞쪽")은
  bbox 중심의 화면 내 위치로 계산 (`classroom_locator/landmark_locator/navigator.py`의
  좌/우 판정 임계값 0.35/0.65와 동일).

## 안 된 것 (필요하면 추가 요청)

- **영상 업로드 분석**(`/api/analyze-video`): 원본 저장소는 업로드된 영상을
  프레임별로 분석해서 오버레이 mp4 + 타임라인을 돌려주는데, 이 프로젝트엔
  아직 그 경로를 안 만들었습니다. 지금은 501 에러로 안내만 하고, 프론트의
  "영상 업로드" 버튼은 그대로 남아있지만 실제로 쓰면 에러 메시지가 뜹니다.
- 원본의 PaddleOCR 기반 텍스트 인식(`texts` 필드)은 안 씀 — 우리는
  "OCR은 3강의실 정정용으로만" 쓰기로 한 결정(2026-09-17)을 그대로 따름.
