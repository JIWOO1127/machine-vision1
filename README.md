# Mobile Vision Inspector

휴대폰에는 모델을 설치하지 않습니다. 휴대폰 브라우저에서 촬영한 사진을 PC 서버로 보내면 서버가 다음 순서로 처리하고 텍스트와 위치 후보를 반환합니다.

1. Faster R-CNN 객체 탐지
2. 숫자 표지판 후보에만 PaddleOCR 실행
3. 미니맵과 객체 크기를 이용한 현재 위치 후보 계산
4. 인식 텍스트, 위치 후보, 분석 결과 이미지를 휴대폰에 표시

사진 촬영뿐 아니라 휴대폰에 저장된 영상 업로드도 지원합니다. 영상은 최대 300MB, 5분까지 받으며 서버가 5프레임마다 분석한 뒤 모바일 재생용 H.264 MP4로 반환합니다.

## 실행

프로젝트 루트에서 아래 명령 하나만 실행합니다.

```powershell
.\backend\venv311\Scripts\python.exe backend\app.py
```

터미널에 `Running on http://...:5000`이 표시되면 서버가 준비된 것입니다. 휴대폰과 PC를 같은 Wi-Fi에 연결하고 휴대폰 브라우저에서 아래 주소로 접속합니다.

```text
http://PC의-IPv4-주소:5000
```

PC의 IPv4 주소는 PowerShell에서 `ipconfig`를 실행해 확인합니다. 예를 들어 IPv4 주소가 `192.168.0.15`라면 휴대폰에서 `http://192.168.0.15:5000`을 엽니다.

Windows 방화벽 메시지가 나오면 개인 네트워크에서 Python의 접근을 허용해야 합니다.

## 화면 수정 후 재빌드

일반 실행에는 프론트엔드 서버를 따로 켤 필요가 없습니다. `frontend/src`를 수정한 경우에만 아래 명령으로 화면을 다시 빌드합니다.

```powershell
cd frontend
npm run build
```

빌드 결과는 `frontend/dist`에 저장되며 Flask가 같은 5000번 포트에서 제공합니다.

## 사용 모델

- 객체 탐지: `backend/models/final_best.pt` (YOLOv8, 서버에서 최우선 사용)
- `backend/models/final.pt`가 추가되면 해당 파일을 `final_best.pt`보다 먼저 사용
- OCR: PaddleOCR Korean PP-OCRv5 mobile
- 결과 영상: H.264, yuv420p, MP4 fast-start
- 객체 탐지는 CUDA GPU를 사용하고 현재 Windows 환경의 PaddleOCR은 CPU를 사용합니다.
- 지도 좌표: `backend/data/minimap_coordinates.json`

서버 상태는 `http://PC-IP:5000/api/health`에서 확인할 수 있습니다.
