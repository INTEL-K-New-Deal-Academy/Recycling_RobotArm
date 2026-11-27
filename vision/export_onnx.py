"""DTRA 시각 백본(ResNet-18 L1-L3 + GAP/2x2/4x4/DoG)을 ONNX opset 17로 내보내고 검증한다."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from vision.dtra_adapter import DTRAVisualEncoder


def export_visual_encoder_onnx(
    out_path: str | Path,
    input_size: int = 128,
    opset: int = 17,
    pretrained: bool = True,
    encoder: Optional[DTRAVisualEncoder] = None,
) -> float:
    """PyTorch 시각 인코더를 ONNX로 내보낸 뒤 PyTorch vs ONNX Runtime 최대 오차를 반환한다."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    enc = (encoder if encoder is not None else DTRAVisualEncoder(pretrained=pretrained)).eval()
    dummy_main = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_wrist = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    with torch.no_grad():
        pt_vis1, pt_vis2 = enc(dummy_main, dummy_wrist)

    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["img_main", "img_wrist"],
        output_names=["vis_stage1", "vis_stage2"],
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        enc,
        (dummy_main, dummy_wrist),
        str(out),
        **export_kwargs,
    )

    model = onnx.load(str(out))
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_vis1, ort_vis2 = sess.run(
        None,
        {"img_main": dummy_main.numpy(), "img_wrist": dummy_wrist.numpy()},
    )
    diff1 = float(np.max(np.abs(pt_vis1.numpy() - ort_vis1)))
    diff2 = float(np.max(np.abs(pt_vis2.numpy() - ort_vis2)))
    return max(diff1, diff2)


def main() -> None:
    ap = argparse.ArgumentParser(description="DTRA 시각 인코더 ONNX 변환기 (opset 17)")
    ap.add_argument("--config", default="vision/config.yaml")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    out_path = args.out or model_cfg["onnx_path"]
    max_diff = export_visual_encoder_onnx(
        out_path=out_path,
        input_size=int(model_cfg["input_size"]),
        opset=int(model_cfg["onnx_opset"]),
        pretrained=(not args.no_pretrained) and bool(model_cfg["pretrained_backbone"]),
    )
    print(f"ONNX 저장 완료: {out_path} (opset={model_cfg['onnx_opset']}, max_abs_diff={max_diff:.6e})")


if __name__ == "__main__":
    main()
