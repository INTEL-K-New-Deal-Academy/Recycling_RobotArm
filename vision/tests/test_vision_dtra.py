"""하드웨어 없이 DTRA v3.1.0 비전 파이프라인 전체를 검증하는 단위 테스트."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vision.camera import FakeCameraSource, VideoFileCameraSource, render_synthetic_frame
from vision.dtra_adapter import DTRAVisionRouter
from vision.export_onnx import export_visual_encoder_onnx
from vision.infer import draw_prediction_overlay


class TestDTRAVisionPipeline(unittest.TestCase):
    def _build_enrolled_router(self, pretrained: bool = False) -> DTRAVisionRouter:
        router = DTRAVisionRouter(
            u_geom_threshold=0.32 if pretrained else 0.003,
            pretrained_backbone=pretrained,
        )
        for cls_name, has_lid, prompt in (
            ("cup_with_lid", True, "disposable beverage cup with lid"),
            ("cup_no_lid", False, "open disposable beverage cup without lid"),
        ):
            frames = [
                render_synthetic_frame(cls_name, shift_xy=(dx, 0), seed=10 + i)
                for i, dx in enumerate((-8, 0, 8, -4, 4))
            ]
            router.enroll_class(
                label=cls_name,
                frames_bgr=frames,
                prompt=prompt,
                is_cup=True,
                has_lid=has_lid,
            )
        return router

    def test_enroll_no_encroachment_zero_drift_and_classify(self) -> None:
        router = self._build_enrolled_router(pretrained=False)
        before_keys = [k.clone() for k in router.keys]

        # 신규 쓰레기(팝콘 통) 추가 등록 시 기존 키가 단 1bit도 변하지 않는지 검증 (L_inf == 0.0)
        popcorn_frames = [
            render_synthetic_frame("popcorn_bucket", shift_xy=(dx, 0), seed=100 + i)
            for i, dx in enumerate((-6, 0, 6))
        ]
        router.enroll_class(
            label="popcorn_bucket",
            frames_bgr=popcorn_frames,
            prompt="popcorn bucket",
            is_cup=False,
            has_lid=None,
        )
        for idx, k_before in enumerate(before_keys):
            drift = float(torch.max(torch.abs(router.keys[idx] - k_before)).item())
            self.assertEqual(drift, 0.0)

        # 합성 프레임 분류 검증
        p_lid = router.predict(render_synthetic_frame("cup_with_lid", seed=99))
        p_nolid = router.predict(render_synthetic_frame("cup_no_lid", seed=99))
        p_other = router.predict(render_synthetic_frame("other_trash", seed=99))

        self.assertEqual(p_lid.status, "cup_with_lid")
        self.assertTrue(p_lid.is_cup)
        self.assertTrue(p_lid.has_lid)

        self.assertEqual(p_nolid.status, "cup_no_lid")
        self.assertTrue(p_nolid.is_cup)
        self.assertFalse(p_nolid.has_lid)

        # 미등록 기타 쓰레기는 기하학적 거리(u_geom) 임계치 초과로 OTHER_TRASH 거절
        self.assertEqual(p_other.status, "OTHER_TRASH")
        self.assertFalse(p_other.is_cup)

        # Snap-Back 상태 초기화 검증 (0.0000)
        self.assertEqual(router.episode_snap_back(), 0.0)

    def test_bank_roundtrip_and_onnx_opset17_parity(self) -> None:
        router = self._build_enrolled_router(pretrained=False)
        test_frame = render_synthetic_frame("cup_with_lid", seed=77)
        pt_pred = router.predict(test_frame)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bank_path = tmp_dir / "bank.npz"
            onnx_path = tmp_dir / "encoder.onnx"
            router.save_bank(bank_path)

            max_diff = export_visual_encoder_onnx(
                out_path=onnx_path,
                input_size=router.input_size,
                opset=17,
                pretrained=False,
                encoder=router.visual_encoder,
            )
            self.assertLess(max_diff, 1e-4)

            # ONNX 백엔드로 동일 뱅크 로드 후 결과 일치 검증
            ort_router = DTRAVisionRouter(
                u_geom_threshold=router.u_geom_threshold,
                pretrained_backbone=False,
                onnx_path=onnx_path,
            )
            ort_router.load_bank(bank_path)
            ort_pred = ort_router.predict(test_frame)

            self.assertEqual(ort_pred.backend, "onnxruntime")
            self.assertEqual(ort_pred.status, pt_pred.status)
            self.assertAlmostEqual(ort_pred.confidence, pt_pred.confidence, places=3)

    def test_fake_camera_cycle(self) -> None:
        cam = FakeCameraSource(mode="cycle")
        frames = [cam.read()[1] for _ in range(4)]
        cam.release()
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(isinstance(f, np.ndarray) and f.shape == (480, 640, 3) for f in frames))

    def test_overlay_render_does_not_mutate_input(self) -> None:
        router = self._build_enrolled_router(pretrained=False)
        frame = render_synthetic_frame("cup_with_lid", seed=55)
        frame_orig = frame.copy()
        pred = router.predict(frame)
        vis = draw_prediction_overlay(frame, pred, u_geom_threshold=router.u_geom_threshold)
        self.assertEqual(vis.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, frame_orig))
        self.assertFalse(np.array_equal(vis, frame))
        self.assertIsNotNone(VideoFileCameraSource)


if __name__ == "__main__":
    unittest.main()
