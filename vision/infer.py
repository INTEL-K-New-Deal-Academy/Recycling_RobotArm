"""실시간 비전 추론 CLI (PyTorch 및 ONNX Runtime 공용, fake 카메라 지원).

사용 예:
  python -m vision.infer --fake --max-frames 4
  python -m vision.infer --onnx vision/models/dtra_visual_encoder_opset17.onnx --fake --max-frames 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vision.camera import FakeCameraSource, OpenCVCameraSource
from vision.dtra_adapter import DTRAVisionRouter


def main() -> None:
    ap = argparse.ArgumentParser(description="DTRA 비전 실시간 판별 CLI")
    ap.add_argument("--config", default="vision/config.yaml")
    ap.add_argument("--bank", default="", help="등록된 슬롯 뱅크 .npz 경로")
    ap.add_argument("--onnx", default="", help="ONNX 인코더 경로 (지정 시 onnxruntime 사용)")
    ap.add_argument("--fake", action="store_true", help="하드웨어 없이 합성 카메라 사용")
    ap.add_argument("--fake-mode", default="cycle", help="cycle / cup_with_lid / cup_no_lid / other_trash")
    ap.add_argument("--camera-index", type=int, default=-1)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    route_cfg = cfg["routing"]
    bank_path = Path(args.bank or model_cfg["bank_path"])
    if not bank_path.exists():
        raise SystemExit(
            f"슬롯 뱅크 파일이 없습니다: {bank_path}. 먼저 `python -m vision.enroll --fake` 를 실행하세요."
        )

    router = DTRAVisionRouter(
        d_feat=int(model_cfg["d_feat"]),
        d_lang=int(model_cfg["d_lang"]),
        seed=int(model_cfg["seed"]),
        input_size=int(model_cfg["input_size"]),
        u_geom_threshold=float(route_cfg["u_geom_threshold"]),
        pretrained_backbone=(not args.no_pretrained) and bool(model_cfg["pretrained_backbone"]),
        onnx_path=args.onnx or None,
    )
    router.load_bank(bank_path)

    if args.fake:
        cam = FakeCameraSource(mode=args.fake_mode)
    else:
        cam_idx = args.camera_index if args.camera_index >= 0 else int(cfg["camera"]["main_index"])
        cam = OpenCVCameraSource(
            index=cam_idx,
            width=int(cfg["camera"]["width"]),
            height=int(cfg["camera"]["height"]),
            exposure=cfg["camera"]["exposure"],
        )

    try:
        for step in range(args.max_frames):
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            pred = router.predict(frame)
            print(json.dumps({"step": step, **pred.to_dict()}, ensure_ascii=False))
        snap = router.episode_snap_back()
        print(json.dumps({"event": "episode_snap_back", "post_snap_norm": snap}))
    finally:
        cam.release()


if __name__ == "__main__":
    main()
