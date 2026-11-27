"""소량 샘플(1~30장)로 DTRA 슬롯 뱅크를 생성/확장 등록하는 CLI.

사용 예:
  python -m vision.enroll --fake --out vision/weights/dtra_bank.npz
  python -m vision.enroll --label lid_clear_straw --has-lid --video raw_data/20261127_094718.mp4
  python -m vision.enroll --label cup_with_lid --frames-dir path/to/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

from vision.camera import FakeCameraSource, OpenCVCameraSource
from vision.dtra_adapter import DTRAVisionRouter


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_frames_from_dir(frames_dir: Path, limit: int) -> List[np.ndarray]:
    files = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )[:limit]
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            frames.append(img)
    return frames


def collect_frames_from_video(video_path: Path, limit: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1), max(1, limit), dtype=int)
    frames = []
    try:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="DTRA 소량 샘플 슬롯 등록 CLI")
    ap.add_argument("--config", default="vision/config.yaml")
    ap.add_argument("--out", default="", help="저장할 .npz 경로 (기본값: config.yaml)")
    ap.add_argument("--label", default="", help="등록할 클래스 이름")
    ap.add_argument("--prompt", default="", help="선택적 자연어 설명")
    ap.add_argument("--has-lid", action="store_true", help="뚜껑 있음 여부 지정")
    ap.add_argument("--no-lid", action="store_true", help="뚜껑 없음 여부 지정")
    ap.add_argument("--not-cup", action="store_true", help="기타/EMPTY 슬롯 등록 시 지정")
    ap.add_argument("--frames-dir", default="", help="샘플 이미지 폴더")
    ap.add_argument("--video", default="", help="샘플 동영상(.mp4) 경로")
    ap.add_argument("--camera-index", type=int, default=-1, help="실제 웹캠에서 바로 수집할 인덱스")
    ap.add_argument("--num-frames", type=int, default=12)
    ap.add_argument("--fake", action="store_true", help="하드웨어 없이 합성 컵 샘플로 등록")
    ap.add_argument("--no-pretrained", action="store_true", help="오프라인 단위 테스트용 랜덤 백본")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    route_cfg = cfg["routing"]
    out_path = Path(args.out or model_cfg["bank_path"])

    router = DTRAVisionRouter(
        d_feat=int(model_cfg["d_feat"]),
        d_lang=int(model_cfg["d_lang"]),
        seed=int(model_cfg["seed"]),
        input_size=int(model_cfg["input_size"]),
        u_geom_threshold=float(route_cfg["u_geom_threshold"]),
        pretrained_backbone=(not args.no_pretrained) and bool(model_cfg["pretrained_backbone"]),
    )
    if out_path.exists():
        router.load_bank(out_path)

    if args.fake:
        classes = cfg["classes"]
        for cls_name, meta in classes.items():
            if cls_name in router.class_names:
                continue
            cam = FakeCameraSource(mode=cls_name)
            frames = [cam.read()[1] for _ in range(args.num_frames)]
            valid = [f for f in frames if f is not None]
            stat = router.enroll_class(
                label=cls_name,
                frames_bgr=valid,
                prompt=meta.get("prompt", ""),
                is_cup=bool(meta.get("is_cup", True)),
                has_lid=meta.get("has_lid"),
            )
            print("[ENROLL-FAKE]", stat)
    else:
        if not args.label:
            raise SystemExit("--fake 가 아니면 --label 을 반드시 지정해야 합니다.")
        if args.video:
            frames = collect_frames_from_video(Path(args.video), args.num_frames)
        elif args.frames_dir:
            frames = collect_frames_from_dir(Path(args.frames_dir), args.num_frames)
        elif args.camera_index >= 0:
            cam = OpenCVCameraSource(index=args.camera_index)
            try:
                frames = []
                for _ in range(args.num_frames):
                    ok, frame = cam.read()
                    if ok and frame is not None:
                        frames.append(frame)
            finally:
                cam.release()
        else:
            raise SystemExit("--video, --frames-dir, --camera-index 중 하나를 지정하세요.")

        cls_meta = cfg["classes"].get(args.label, {})
        has_lid = True if args.has_lid else (False if args.no_lid else cls_meta.get("has_lid"))
        is_cup = False if args.not_cup else bool(cls_meta.get("is_cup", True))
        stat = router.enroll_class(
            label=args.label,
            frames_bgr=frames,
            prompt=args.prompt or cls_meta.get("prompt", ""),
            is_cup=is_cup,
            has_lid=has_lid,
        )
        print("[ENROLL]", stat)

    saved = router.save_bank(out_path)
    print(f"저장 완료: {saved} (총 {len(router.class_names)}개 슬롯)")


if __name__ == "__main__":
    main()
