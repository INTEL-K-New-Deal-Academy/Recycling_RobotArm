"""실시간 비전 추론 & 시각화 CLI (PyTorch / ONNX Runtime 공용, 웹캠·동영상·fake 지원).

사용 예:
  # 웹캠 실시간 화면 띄워서 시연 (q 또는 ESC 로 종료)
  python -m vision.infer --onnx vision/models/dtra_visual_encoder_opset17.onnx --camera-index 0 --show --max-frames 0

  # 촬영 동영상으로 화면 시연 또는 결과 영상 저장
  python -m vision.infer --video raw_data/영상.mp4 --show
  python -m vision.infer --video raw_data/영상.mp4 --save-out demo_out.mp4

  # 가짜 카메라 테스트
  python -m vision.infer --fake --max-frames 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from vision.camera import FakeCameraSource, OpenCVCameraSource, VideoFileCameraSource
from vision.dtra_adapter import DTRAVisionRouter, VisionPrediction


def _banner_style(pred: VisionPrediction) -> tuple[str, tuple[int, int, int]]:
    if pred.status == "EMPTY":
        return "EMPTY", (150, 150, 150)
    if not pred.is_cup:
        return f"PASS: {pred.status}", (45, 95, 240)
    if pred.has_lid is True:
        return f"CUP [WITH LID] ({pred.status})", (55, 195, 70)
    if pred.has_lid is False:
        return f"CUP [NO LID] ({pred.status})", (225, 185, 35)
    return f"CUP ({pred.status})", (55, 195, 70)


def draw_prediction_overlay(
    frame_bgr: np.ndarray,
    pred: VisionPrediction,
    u_geom_threshold: float = 0.32,
) -> np.ndarray:
    """원본 프레임 사본에 중앙 크롭 가이드, 바운딩 박스, 상태 배너, 수치를 그린다."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    scale = max(0.55, min(w, h) / 640.0)
    thick = max(2, int(round(2 * scale)))

    # 1) 중앙 정사각형 인식 영역 가이드 (점선 느낌의 코너 브라켓)
    m = min(h, w)
    x0, y0 = (w - m) // 2, (h - m) // 2
    cv2.rectangle(vis, (x0, y0), (x0 + m, y0 + m), (210, 210, 210), 1)

    title, color = _banner_style(pred)

    # 2) 검출 바운딩 박스 & 중심점 표시
    bx, by, bw, bh = pred.bbox
    if pred.status != "EMPTY" and bw > 0 and bh > 0:
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, thick)
        cx_px, cy_px = int(pred.center_xy[0] * w), int(pred.center_xy[1] * h)
        cv2.circle(vis, (cx_px, cy_px), max(4, thick * 2), color, -1)

    # 3) 상단 상태 배너
    banner_h = int(round(68 * scale))
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 22, 26), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)
    cv2.rectangle(vis, (0, banner_h - 4), (w, banner_h), color, -1)

    fps = 1000.0 / max(pred.latency_ms, 0.1)
    line1 = title
    line2 = (
        f"sim={pred.confidence:.3f}  u_geom={pred.u_geom:.3f} (<{u_geom_threshold:.2f})  "
        f"{pred.latency_ms:.1f}ms ({fps:.0f}fps)  [{pred.backend}]"
    )
    cv2.putText(
        vis,
        line1,
        (14, int(round(28 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * scale,
        color,
        max(2, thick),
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        line2,
        (14, int(round(54 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        (235, 238, 242),
        max(1, thick - 1),
        cv2.LINE_AA,
    )
    return vis


def _resize_for_display(frame_bgr: np.ndarray, max_height: int = 720) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if h <= max_height:
        return frame_bgr
    ratio = max_height / float(h)
    return cv2.resize(frame_bgr, (int(w * ratio), max_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description="DTRA 비전 실시간 판별 & 시각화 CLI")
    ap.add_argument("--config", default="vision/config.yaml")
    ap.add_argument("--bank", default="", help="등록된 슬롯 뱅크 .npz 경로")
    ap.add_argument("--onnx", default="", help="ONNX 인코더 경로 (지정 시 onnxruntime 사용)")
    ap.add_argument("--fake", action="store_true", help="하드웨어 없이 합성 카메라 사용")
    ap.add_argument("--fake-mode", default="cycle", help="cycle / cup_with_lid / cup_no_lid / other_trash")
    ap.add_argument("--video", default="", help="재생할 동영상(.mp4) 경로")
    ap.add_argument("--loop", action="store_true", help="동영상 반복 재생")
    ap.add_argument("--camera-index", type=int, default=-1)
    ap.add_argument("--max-frames", type=int, default=8, help="0 이면 q/ESC 또는 영상 끝까지 계속 실행")
    ap.add_argument("--show", action="store_true", help="OpenCV 실시간 창 띄우기")
    ap.add_argument("--save-out", default="", help="시각화 결과를 .mp4 동영상으로 저장")
    ap.add_argument("--display-height", type=int, default=720, help="--show 창 최대 세로 높이")
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

    u_thresh = float(route_cfg["u_geom_threshold"])
    router = DTRAVisionRouter(
        d_feat=int(model_cfg["d_feat"]),
        d_lang=int(model_cfg["d_lang"]),
        seed=int(model_cfg["seed"]),
        input_size=int(model_cfg["input_size"]),
        u_geom_threshold=u_thresh,
        pretrained_backbone=(not args.no_pretrained) and bool(model_cfg["pretrained_backbone"]),
        onnx_path=args.onnx or None,
    )
    router.load_bank(bank_path)

    if args.fake:
        cam = FakeCameraSource(mode=args.fake_mode)
    elif args.video:
        cam = VideoFileCameraSource(args.video, loop=args.loop)
    else:
        cam_idx = args.camera_index if args.camera_index >= 0 else int(cfg["camera"]["main_index"])
        cam = OpenCVCameraSource(
            index=cam_idx,
            width=int(cfg["camera"]["width"]),
            height=int(cfg["camera"]["height"]),
            exposure=cfg["camera"]["exposure"],
        )

    writer: Optional[cv2.VideoWriter] = None
    step = 0
    try:
        while args.max_frames <= 0 or step < args.max_frames:
            ok, frame = cam.read()
            if not ok or frame is None:
                break
            pred = router.predict(frame)
            print(json.dumps({"step": step, **pred.to_dict()}, ensure_ascii=False))

            if args.show or args.save_out:
                vis = draw_prediction_overlay(frame, pred, u_geom_threshold=u_thresh)
                if args.save_out:
                    if writer is None:
                        out_p = Path(args.save_out)
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        fps = getattr(cam, "fps", 15.0)
                        writer = cv2.VideoWriter(
                            str(out_p),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (vis.shape[1], vis.shape[0]),
                        )
                    writer.write(vis)
                if args.show:
                    cv2.imshow("DTRA Vision Demo (q/ESC to quit)", _resize_for_display(vis, args.display_height))
                    key = cv2.waitKey(1 if not args.fake else 400) & 0xFF
                    if key in (ord("q"), 27):
                        break
            step += 1

        snap = router.episode_snap_back()
        print(json.dumps({"event": "episode_snap_back", "post_snap_norm": snap}))
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    main()
