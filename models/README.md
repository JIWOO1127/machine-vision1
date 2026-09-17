# models/

- `yolo/` : 학습된 YOLO 가중치 파일 (예: `best.pt`). `configs/default.yaml`의
  `detector.weights` 값이 이 폴더의 파일을 가리키도록 설정하세요.
- `ocr/` : 별도 OCR 모델/가중치가 필요한 경우 (EasyOCR/PaddleOCR은 보통
  자동으로 다운로드하므로 대부분의 경우 비워둬도 됩니다).

용량이 큰 가중치 파일은 git에 커밋하지 않는 것을 권장합니다 (.gitignore 처리됨).
