"""DTRA v3.1.0 어댑터 모듈 (ResNet-18 GAP + No-Encroachment + 경량 검증 벡터 매칭).

`5Color/dtra` (`vision/third_party/dtra`, tag v3.1.0) 의 검증된 모듈을 직접 import해
재사용하며, 코드를 복제하지 않는다 (`CLAUDE.md` 5절 준수).
- 특징 추출: 동결 ResNet-18 GAP(layer4, 512-d) + L1-L3 GAP/DoG(512-d)
- 침범 금지 등록: _refine_key_with_contrastive_margin (No-Encroachment, <=96 vectors/slot)
- 기하학적 OOD 거절: U_geom = 1.0 - max_cos >= u_geom_threshold -> OTHER_TRASH
- 에피소드 경계 초기화: episode_snap_back() -> exact 0.0000
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

VISION_DIR = Path(__file__).resolve().parent
DTRA_SUBMODULE_ROOT = VISION_DIR / "third_party" / "dtra"

if not (DTRA_SUBMODULE_ROOT / "dtra").exists():
    raise RuntimeError(
        f"DTRA v3.1.0 서브모듈을 찾을 수 없습니다: {DTRA_SUBMODULE_ROOT}. "
        "`git submodule update --init --recursive` 를 먼저 실행하세요."
    )

if str(DTRA_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DTRA_SUBMODULE_ROOT))

# 5Color/dtra v3.1.0 공식 모듈 직접 참조 (코드 복사 금지)
from dtra.core.bspline_core import BSplineDecoder  # noqa: E402, F401
from dtra.experiments.libero.benchmark_dtra_libero import (  # noqa: E402
    LightweightTextEncoder,
    _refine_key_with_contrastive_margin,
)


def center_square_crop(frame_bgr: np.ndarray) -> np.ndarray:
    """세로(1080x1920)/가로(640x480) 비율 왜곡 방지를 위한 중앙 정사각형 크롭."""
    h, w = frame_bgr.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    return frame_bgr[y0 : y0 + m, x0 : x0 + m]


def preprocess_bgr_to_norm_nchw(frame_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """BGR uint8 -> 중앙 정사각 크롭 -> ImageNet 정규화 float32 (1, 3, size, size)."""
    cropped = center_square_crop(frame_bgr)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def estimate_cup_center(
    frame_bgr: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[int, int, int, int]]:
    """전경 물체의 대략적 중심 좌표(정규화 0~1)와 바운딩 박스를 추정한다."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0.5, 0.5), (w // 4, h // 4, w // 2, h // 2)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (h * w * 0.005):
        return (0.5, 0.5), (w // 4, h // 4, w // 2, h // 2)

    x, y, bw, bh = cv2.boundingRect(largest)
    cx = (x + bw * 0.5) / float(w)
    cy = (y + bh * 0.5) / float(h)
    return (round(cx, 4), round(cy, 4)), (int(x), int(y), int(bw), int(bh))


class DTRAVisualEncoder(nn.Module):
    """동결 ResNet-18 (GAP 512-d + L3 DoG 512-d) 특징 추출기 (어텐션 없음, ONNX opset 17 호환)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.stem_to_l3 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.layer4 = backbone.layer4
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, img_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """정규화 RGB 텐서 (B, 3, H, W) -> L2 정규화된 (vis_gap [B, 512], vis_cmc [B, 512])."""
        f3 = self.stem_to_l3(img_norm)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)

        gap4 = f4.mean(dim=(2, 3))  # 위치/크기 변화에 강건한 GAP (B, 512)
        gap4 = F.normalize(gap4, dim=-1)

        gap3 = f3.mean(dim=(2, 3))
        dog3 = (f3 - F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)).abs().mean(dim=(2, 3))
        vis_cmc = F.normalize(torch.cat([gap3, dog3], dim=-1), dim=-1)
        return gap4, vis_cmc


@dataclass
class VisionPrediction:
    """메인 제어기(interfaces/main)로 전달할 구조화 인식 결과."""

    status: str  # 등록된 클래스명(예: cup_with_lid) 또는 OTHER_TRASH / EMPTY
    is_cup: bool
    has_lid: Optional[bool]
    confidence: float
    u_geom: float
    margin: float
    center_xy: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    latency_ms: float
    backend: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


class DTRAVisionRouter:
    """DTRA v3.1.0 기반 소량 등록·실시간 비전 라우터 (PyTorch & ONNX Runtime 공용)."""

    def __init__(
        self,
        d_feat: int = 512,
        d_lang: int = 128,
        seed: int = 42,
        input_size: int = 128,
        u_geom_threshold: float = 0.32,
        pretrained_backbone: bool = True,
        onnx_path: Optional[str | Path] = None,
    ) -> None:
        self.d_feat = d_feat
        self.d_lang = d_lang
        self.seed = seed
        self.input_size = input_size
        self.u_geom_threshold = u_geom_threshold
        self.device = torch.device("cpu")

        self.text_encoder = LightweightTextEncoder(dim=d_lang)
        self.visual_encoder: Optional[DTRAVisualEncoder] = None
        self.ort_session = None
        self.backend = "pytorch"

        if onnx_path is not None and Path(onnx_path).exists():
            import onnxruntime as ort

            self.ort_session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self.backend = "onnxruntime"
        else:
            self.visual_encoder = DTRAVisualEncoder(pretrained=pretrained_backbone)

        # 등록된 클래스 슬롯 및 불변량 상태 버퍼 (과거 슬롯 드립프트 0.0 보장)
        self.class_names: List[str] = []
        self.class_meta: Dict[str, Dict[str, Any]] = {}
        self.keys: List[torch.Tensor] = []
        self.anchors: List[torch.Tensor] = []
        self.self_sims: List[float] = []
        self.val_queries: List[torch.Tensor] = []  # 슬롯당 <=96개 경량 대표 벡터
        self.cmc_sigs: List[np.ndarray] = []
        self.lang_sigs: List[np.ndarray] = []

        # 에피소드 내 임시 관성/상태 버퍼 (episode_snap_back 시 exact 0.0000 초기화)
        self.episodic_trace = np.zeros(0, dtype=np.float32)
        self.prev_selected: Optional[int] = None

    def _extract_visual_features(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = preprocess_bgr_to_norm_nchw(frame_bgr, size=self.input_size)
        if self.ort_session is not None:
            vis_gap, vis_cmc = self.ort_session.run(None, {"img_main": x})
            return vis_gap[0].astype(np.float32), vis_cmc[0].astype(np.float32)

        assert self.visual_encoder is not None
        with torch.no_grad():
            g_t, cmc_t = self.visual_encoder(torch.from_numpy(x))
        return g_t[0].numpy().astype(np.float32), cmc_t[0].numpy().astype(np.float32)

    def enroll_class(
        self,
        label: str,
        frames_bgr: Sequence[np.ndarray],
        prompt: str = "",
        is_cup: bool = True,
        has_lid: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """1~30장의 샘플 프레임만으로 기존 슬롯을 침범/수정하지 않고 새 클래스 슬롯을 등록한다."""
        if not frames_bgr:
            raise ValueError("등록할 샘플 프레임이 최소 1장 이상 필요합니다.")
        if label in self.class_names:
            raise ValueError(f"이미 등록된 클래스입니다: {label}")

        feats = [self._extract_visual_features(f) for f in frames_bgr]
        gap_arr = np.stack([g for g, _ in feats], axis=0)
        cmc_arr = np.stack([c for _, c in feats], axis=0)

        q_tensor = torch.from_numpy(gap_arr[:96])
        raw_key = F.normalize(q_tensor.mean(dim=0), dim=-1)

        # DTRA v3.1.0 No-Encroachment 정제 (기존 keys / anchors / val_queries 불변)
        refined_key, stats = _refine_key_with_contrastive_margin(
            raw_key=raw_key,
            existing_keys=self.keys,
            sample_queries=q_tensor,
            existing_anchors=self.anchors,
            existing_self_sims=self.self_sims,
            existing_val_queries=self.val_queries,
        )
        anchor_mu = getattr(refined_key, "_anchor_mu", raw_key).detach().clone()
        self_sim = float(getattr(refined_key, "_self_sim", 1.0))
        val_q = getattr(refined_key, "_val_queries", q_tensor).detach().clone()
        refined_key = refined_key.detach().clone()

        lang_vec = self.text_encoder.encode(prompt or label, self.device).numpy().astype(np.float32)
        cmc_mean = _l2_normalize_np(cmc_arr.mean(axis=0))

        self.class_names.append(label)
        self.class_meta[label] = {
            "prompt": prompt or label,
            "is_cup": bool(is_cup),
            "has_lid": has_lid,
        }
        self.keys.append(refined_key)
        self.anchors.append(anchor_mu)
        self.self_sims.append(self_sim)
        self.val_queries.append(val_q)
        self.cmc_sigs.append(cmc_mean)
        self.lang_sigs.append(lang_vec)
        self.episodic_trace = np.zeros(len(self.class_names), dtype=np.float32)
        return {"label": label, "num_samples": len(frames_bgr), **stats}

    def predict(self, frame_bgr: np.ndarray) -> VisionPrediction:
        """단일 프레임을 GAP 대표 벡터 + No-Encroachment 키 + U_geom 게이트로 판별한다."""
        t0 = time.perf_counter()
        center_xy, bbox = estimate_cup_center(frame_bgr)

        if not self.keys:
            raise RuntimeError("등록된 클래스 슬롯이 없습니다. enroll_class()를 먼저 호출하세요.")

        q_gap, q_cmc = self._extract_visual_features(frame_bgr)
        k_count = len(self.keys)

        # 각 슬롯에 저장된 경량 대표 벡터(<=96개) 중 최대 유사도 + 정제 키 유사도 결합
        exemplar_scores = np.array(
            [float((vq.numpy() @ q_gap).max()) for vq in self.val_queries],
            dtype=np.float32,
        )
        key_bank = np.stack([k.numpy() for k in self.keys], axis=0)
        key_scores = key_bank @ q_gap
        scores = 0.85 * exemplar_scores + 0.15 * key_scores

        order = np.argsort(-scores)
        top1_idx = int(order[0])
        top1_score = float(scores[top1_idx])
        top2_score = float(scores[int(order[1])]) if k_count >= 2 else -1.0
        margin = max(0.0, top1_score - top2_score)
        selected_idx = top1_idx

        u_geom = float(max(0.0, 1.0 - float(exemplar_scores[selected_idx])))
        if u_geom >= self.u_geom_threshold:
            status = "OTHER_TRASH"
            is_cup = False
            has_lid = None
        else:
            status = self.class_names[selected_idx]
            meta = self.class_meta[status]
            is_cup = bool(meta["is_cup"])
            has_lid = meta["has_lid"]
            if len(self.episodic_trace) == k_count:
                self.episodic_trace[selected_idx] = 1.0
            self.prev_selected = selected_idx

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return VisionPrediction(
            status=status,
            is_cup=is_cup,
            has_lid=has_lid,
            confidence=round(float(exemplar_scores[selected_idx]), 4),
            u_geom=round(u_geom, 4),
            margin=round(margin, 4),
            center_xy=center_xy,
            bbox=bbox,
            latency_ms=latency_ms,
            backend=self.backend,
        )

    def episode_snap_back(self) -> float:
        """에피소드 종료 시 임시 관성/추적 버퍼를 exact 0.0000 으로 초기화한다."""
        self.episodic_trace.fill(0.0)
        self.prev_selected = None
        if len(self.episodic_trace) == 0:
            return 0.0
        return float(np.max(np.abs(self.episodic_trace)))

    def save_bank(self, path: str | Path) -> Path:
        """등록된 초경량 슬롯 뱅크(.npz) 저장 (원본 이미지 미저장, <=96 벡터/슬롯)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(
            {
                "class_names": self.class_names,
                "class_meta": self.class_meta,
                "d_feat": self.d_feat,
                "d_lang": self.d_lang,
                "seed": self.seed,
                "input_size": self.input_size,
                "u_geom_threshold": self.u_geom_threshold,
            },
            ensure_ascii=False,
        )
        # 슬롯마다 개수가 다를 수 있는 val_queries를 고정 패딩 없이 안전하게 직렬화
        val_lengths = np.array([int(vq.shape[0]) for vq in self.val_queries], dtype=np.int32)
        val_concat = np.concatenate([vq.numpy() for vq in self.val_queries], axis=0)
        np.savez(
            out_path,
            meta_json=np.array([meta_json]),
            keys=np.stack([k.numpy() for k in self.keys], axis=0),
            anchors=np.stack([a.numpy() for a in self.anchors], axis=0),
            self_sims=np.array(self.self_sims, dtype=np.float32),
            val_lengths=val_lengths,
            val_concat=val_concat.astype(np.float32),
            cmc_sigs=np.stack(self.cmc_sigs, axis=0),
            lang_sigs=np.stack(self.lang_sigs, axis=0),
        )
        return out_path

    def load_bank(self, path: str | Path) -> None:
        """저장된 슬롯 뱅크(.npz)를 로드해 즉시 추론 준비 상태로 복원한다."""
        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(data["meta_json"][0]))
        self.class_names = list(meta["class_names"])
        self.class_meta = dict(meta["class_meta"])
        self.u_geom_threshold = float(meta.get("u_geom_threshold", self.u_geom_threshold))

        keys_arr = data["keys"]
        anchors_arr = data["anchors"]
        self_sims_arr = data["self_sims"]
        val_lengths = data["val_lengths"]
        val_concat = data["val_concat"]
        cmc_arr = data["cmc_sigs"]
        lang_arr = data["lang_sigs"]

        self.keys = [torch.from_numpy(keys_arr[i].copy()) for i in range(len(self.class_names))]
        self.anchors = [torch.from_numpy(anchors_arr[i].copy()) for i in range(len(self.class_names))]
        self.self_sims = [float(x) for x in self_sims_arr]

        self.val_queries = []
        offset = 0
        for length in val_lengths:
            n = int(length)
            self.val_queries.append(torch.from_numpy(val_concat[offset : offset + n].copy()))
            offset += n

        self.cmc_sigs = [cmc_arr[i].copy() for i in range(len(self.class_names))]
        self.lang_sigs = [lang_arr[i].copy() for i in range(len(self.class_names))]
        self.episodic_trace = np.zeros(len(self.class_names), dtype=np.float32)
