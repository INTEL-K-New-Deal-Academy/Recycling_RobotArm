"""카메라 입력 추상화 (실제 OpenCV 카메라 + 하드웨어 없는 개발용 Fake 카메라).

- 실제 카메라 설정(CAP_DSHOW, BUFFERSIZE=1 등)은 autonomous-delivery/arm/vision.py 패턴을 참고함.
- FakeCameraSource 는 실기 없이도 cup_with_lid / cup_no_lid / other_trash / empty 시나리오를
  검증할 수 있도록 결정론적 합성 프레임을 생성하거나 폴더 이미지를 순회한다.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraSource(ABC):
    """비전 파이프라인용 공통 카메라 인터페이스."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """BGR uint8 프레임 (H, W, 3) 반환."""

    @abstractmethod
    def release(self) -> None:
        """자원 해제."""


class OpenCVCameraSource(CameraSource):
    """USB 웹캠 / icSpring 팔 카메라 입력."""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        exposure: str | float = "auto",
    ) -> None:
        if isinstance(index, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: index={index}")
        self._set_exposure(exposure)

    def _set_exposure(self, value: str | float | None) -> None:
        if value in (None, "auto", ""):
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if os.name == "nt" else 3)
            return
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if os.name == "nt" else 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.cap.release()


def render_synthetic_frame(
    kind: str,
    width: int = 640,
    height: int = 480,
    shift_xy: Tuple[int, int] = (0, 0),
    noise_std: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """하드웨어 없이 동작 검증을 하기 위한 합성 컨베이어 벨트 장면 생성."""
    rng = np.random.default_rng(seed)
    # 어두운 회색 컨베이어 벨트 배경 + 칸 구분선
    frame = np.full((height, width, 3), (48, 50, 54), dtype=np.uint8)
    cv2.line(frame, (0, 80), (width, 80), (85, 88, 92), 4)
    cv2.line(frame, (0, height - 80), (width, height - 80), (85, 88, 92), 4)

    cx = width // 2 + shift_xy[0]
    cy = height // 2 + shift_xy[1]

    if kind == "cup_with_lid":
        # 흰색/연회색 컵 몸통 (테이퍼 사다리꼴) + 빨간 뚜껑 + 빨대
        body_pts = np.array(
            [
                [cx - 65, cy - 55],
                [cx + 65, cy - 55],
                [cx + 48, cy + 95],
                [cx - 48, cy + 95],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [body_pts], (225, 228, 232))
        cv2.polylines(frame, [body_pts], True, (180, 185, 190), 2)
        # 콜라 라벨 밴드
        cv2.rectangle(frame, (cx - 56, cy - 10), (cx + 56, cy + 35), (30, 35, 200), -1)
        # 돔/플랫 뚜껑 (선명한 상단 림 + 뚜껑 캡)
        cv2.ellipse(frame, (cx, cy - 55), (72, 20), 0, 0, 360, (240, 240, 245), -1)
        cv2.ellipse(frame, (cx, cy - 66), (62, 16), 0, 0, 360, (210, 215, 220), -1)
        cv2.ellipse(frame, (cx, cy - 66), (62, 16), 0, 0, 360, (120, 120, 130), 2)
        # 빨대
        cv2.line(frame, (cx + 8, cy - 66), (cx + 28, cy - 135), (40, 200, 40), 6)

    elif kind == "cup_no_lid":
        # 같은 컵 몸통이지만 뚜껑 없이 안쪽 어두운 액체/림 노출
        body_pts = np.array(
            [
                [cx - 65, cy - 55],
                [cx + 65, cy - 55],
                [cx + 48, cy + 95],
                [cx - 48, cy + 95],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [body_pts], (225, 228, 232))
        cv2.polylines(frame, [body_pts], True, (180, 185, 190), 2)
        cv2.rectangle(frame, (cx - 56, cy - 10), (cx + 56, cy + 35), (30, 35, 200), -1)
        # 열린 컵 안쪽(어두운 콜라색 원형)
        cv2.ellipse(frame, (cx, cy - 55), (65, 18), 0, 0, 360, (235, 235, 235), 4)
        cv2.ellipse(frame, (cx, cy - 54), (58, 14), 0, 0, 360, (25, 20, 35), -1)

    elif kind == "popcorn_bucket":
        # 넓은 사다리꼴 노란/빨간 줄무늬 팝콘 통 (추후 확장 테스트용)
        pts = np.array(
            [
                [cx - 105, cy - 75],
                [cx + 105, cy - 75],
                [cx + 75, cy + 95],
                [cx - 75, cy + 95],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (40, 195, 240))
        for x_off in (-60, -20, 20, 60):
            cv2.line(frame, (cx + x_off, cy - 75), (cx + int(x_off * 0.72), cy + 95), (35, 35, 215), 14)
        cv2.circle(frame, (cx - 25, cy - 82), 18, (235, 245, 250), -1)
        cv2.circle(frame, (cx + 18, cy - 85), 20, (230, 242, 248), -1)

    elif kind == "other_trash":
        # 비정형 과자봉지/구겨진 종이 쓰레기
        poly = np.array(
            [
                [cx - 110, cy - 25],
                [cx - 20, cy - 70],
                [cx + 95, cy - 45],
                [cx + 120, cy + 35],
                [cx + 10, cy + 75],
                [cx - 95, cy + 50],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [poly], (60, 150, 75))
        cv2.line(frame, (cx - 60, cy - 20), (cx + 65, cy + 25), (30, 80, 40), 5)
        cv2.line(frame, (cx - 40, cy + 35), (cx + 45, cy - 35), (190, 220, 180), 4)

    elif kind == "empty":
        pass
    else:
        raise ValueError(f"지원하지 않는 합성 프레임 종류입니다: {kind}")

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame


class FakeCameraSource(CameraSource):
    """하드웨어 없이 폴더 이미지 또는 합성 프레임을 순차 공급하는 가짜 카메라."""

    DEFAULT_SEQUENCE = ("cup_with_lid", "cup_no_lid", "other_trash", "empty")

    def __init__(
        self,
        mode: str = "cycle",
        image_dir: Optional[str | Path] = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.mode = mode
        self.width = width
        self.height = height
        self._step = 0
        self._images: List[Path] = []
        if image_dir is not None:
            p = Path(image_dir)
            self._images = sorted(
                [f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._images:
            img_path = self._images[self._step % len(self._images)]
            self._step += 1
            frame = cv2.imread(str(img_path))
            return (frame is not None), frame

        kind = (
            self.DEFAULT_SEQUENCE[self._step % len(self.DEFAULT_SEQUENCE)]
            if self.mode == "cycle"
            else self.mode
        )
        shift = ((self._step % 3 - 1) * 8, ((self._step // 3) % 3 - 1) * 5)
        frame = render_synthetic_frame(
            kind=kind,
            width=self.width,
            height=self.height,
            shift_xy=shift,
            seed=42 + self._step,
        )
        self._step += 1
        return True, frame

    def release(self) -> None:
        return None


class VideoFileCameraSource(CameraSource):
    """저장된 동영상(.mp4/.avi)을 카메라처럼 순차 재생하는 입력 소스."""

    def __init__(self, video_path: str | Path, loop: bool = False) -> None:
        self.path = Path(video_path)
        self.loop = loop
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"동영상 파일을 열 수 없습니다: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.cap.release()
