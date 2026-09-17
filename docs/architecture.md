# 아키텍처 개요

```
[웹캠 프레임 / 사진 파일]
        │
        ▼
 ┌─────────────────┐
 │  Detector (YOLO)  │  configs/default.yaml: detector.backend
 └─────────────────┘
        │  bbox 목록 (표지판 후보 영역)
        ▼
 ┌─────────────────┐
 │  crop_bbox()     │  utils/image_utils.py
 └─────────────────┘
        │  crop된 이미지
        ▼
 ┌─────────────────┐
 │  OCR Reader      │  configs/default.yaml: ocr.backend
 └─────────────────┘
        │  인식된 텍스트 목록
        ▼
 ┌─────────────────┐
 │  SignMatcher     │  configs/locations.yaml 과 유사도 비교
 └─────────────────┘
        │
        ▼
   현재 위치 추정 결과
```

## 모듈 책임

- `detection/` : "이미지 -> bbox 목록" 만 책임진다. YOLO 버전이 바뀌어도
  `BaseDetector` 인터페이스만 지키면 나머지 코드는 영향받지 않는다.
- `ocr/` : "crop된 이미지 -> 텍스트 목록" 만 책임진다. OCR 라이브러리가
  바뀌어도 `BaseOcrReader` 인터페이스만 지키면 된다.
- `localization/` : "텍스트 목록 -> 위치" 매칭만 책임진다. YOLO/OCR과
  완전히 독립적이라 단위 테스트가 쉽다 (`tests/test_localization.py` 참고).
- `pipeline/` : 위 세 모듈을 조립한다. `core.py`가 "이미지 한 장 처리"의
  단일 진실 공급원(single source of truth)이고, `realtime_pipeline.py`와
  `batch_pipeline.py`는 입력 소스(웹캠 vs 폴더)만 다르다.

## 확장 아이디어

- 여러 프레임의 매칭 결과를 다수결/이동평균으로 스무딩해서 오탐 줄이기
- 표지판 위치 좌표를 이용해 간단한 실내 지도 위에 현재 위치 시각화
- 표지판 폰트/각도가 다양하면 OCR 전처리(원근 보정, 이진화) 단계 추가
