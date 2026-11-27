# vision — AI 비전

담당: **D (승준, @5Color)**

[5Color/dtra](https://github.com/5Color/dtra) `v3.1.0` 태그를 `git submodule`로 연결해 사용합니다 (`vision/third_party/dtra`, 코드 복제 없음).

| 카메라 | 역할 |
|---|---|
| CAM 1 상단 웹캠 | 벨트 전체, 컵 위치·분포·밀집도 |
| CAM 2 로봇팔 카메라 (`icSpring`) | 컵 위치·자세, 로봇팔 기준 좌표 |
| CAM 3 작업 위치 카메라 | 컵 도착 감지 및 음료 컵/기타 쓰레기 판별 |

## 모델 요약

- 고정된 ResNet 백본 + 물체별 기억 슬롯. 샘플 몇 장과 자연어 라벨로 등록하고, 등록이 끝난 슬롯은 잠근다.
- 배경·조명을 빼고 유사도로 판별: 기준선 안 → 음료 컵(뚜껑 유/무, 집을 위치), 밖 → 기타 쓰레기(통과).
- 컵 하나가 끝날 때마다 상태 초기화. 로봇팔 궤적도 B-spline으로 생성 가능.
- 자세한 설명은 [README 5절](../README.md#ai-비전-모델--dtra-v31).

## 1. 설치 및 서브모듈 받기

```bash
git submodule update --init --recursive
```

> **버전 주의 (`CLAUDE.md` 5절):** Pi 5 및 Docker 환경에서는 `numpy 1.26.4`, `protobuf 3.20.3`, `onnxruntime 1.19.2`, `onnx 1.16.2 (opset 17)` 버전을 유지합니다.

## 2. 모듈 구성

- [dtra_adapter.py](dtra_adapter.py): `5Color/dtra@v3.1.0` 직접 참조 어댑터 (동결 `ResNet-18 GAP` 특징, 소량 등록 `No-Encroachment`, 슬롯당 `<=96` 대표 벡터 매칭, `U_geom` OOD 거절, `episode_snap_back`)
- [camera.py](camera.py): 실제 USB 카메라(`OpenCVCameraSource`) 및 하드웨어 없는 테스트용 합성 카메라(`FakeCameraSource`)
- [enroll.py](enroll.py): 1~30장 샘플(동영상/폴더/웹캠/fake)만으로 기존 슬롯을 건드리지 않고 새 물체를 등록하는 CLI
- [infer.py](infer.py): PyTorch / ONNX Runtime 공용 실시간 추론 CLI
- [export_onnx.py](export_onnx.py): `ResNet-18 GAP` 특징 추출기를 `opset 17` ONNX로 변환 및 오차 검증
- [config.yaml](config.yaml): 카메라 인덱스, `U_geom` 임계치 등 실측 설정값 모음

## 3. 실행 방법

저장소에 사전 등록 컵 슬롯(`vision/weights/dtra_bank.npz`, 124 KB)과 ONNX 백본(`vision/models/dtra_visual_encoder_opset17.onnx`, 42.6 MB)이 포함되어 있어 **별도 데이터 수집·학습 없이 바로 실행**할 수 있습니다.

```bash
# 0) [가장 빠름] git pull 후 바로 실시간 웹캠 화면 창으로 시연하기 (q 또는 ESC 로 종료)
python -m vision.infer --onnx vision/models/dtra_visual_encoder_opset17.onnx --camera-index 0 --show --max-frames 0

# 1) 새 컵/물체를 추가로 등록하거나 처음부터 등록할 때
python -m vision.enroll --fake
python -m vision.enroll --label lid_clear_straw --has-lid --video raw_data/영상.mp4

# 2) ONNX (opset 17) 재변환 및 PyTorch vs ONNX 오차·용량 확인
python -m vision.export_onnx

# 3) ONNX 백엔드로 가짜 카메라 추론 테스트 (화면 띄우려면 --show 추가)
python -m vision.infer --onnx vision/models/dtra_visual_encoder_opset17.onnx --fake --max-frames 4

# 4) 촬영 동영상(.mp4)으로 시연하거나 결과 영상 저장하기
python -m vision.infer --video raw_data/영상.mp4 --show --loop --max-frames 0
python -m vision.infer --video raw_data/영상.mp4 --save-out demo_out.mp4 --max-frames 0

# 5) 단위 테스트 실행
python -m unittest discover -s vision/tests -v
```

## 남은 실기 할 일
- 실제 벨트·컵 샘플 촬영 및 기준선(`u_geom_threshold`) 실측 조정
- 이 프로젝트 환경에서 정확도·처리 시간 측정 (`test` 이슈로 `docs/experiments/` 기록)

> 학습 원본 데이터·실패 사진·영상(`raw_data/`)은 Git에 올리지 않고 팀 공유 드라이브에 보관합니다.
