"""DTRA v3.1.0 어댑터 모듈.

`5Color/dtra` (`vision/third_party/dtra`, tag v3.1.0) 의 검증된 모듈을 직접 import해
재사용하며, 코드를 복제하지 않는다 (`CLAUDE.md` 5절 준수).
- 특징 추출: FrozenResNet18DualCam (L1-L3) + MultimodalMagneticKeyExtractor
- 침범 금지 등록: _refine_key_with_contrastive_margin (No-Encroachment, <=96 vectors/class)
- 2단계 라우팅: Top-M Common-Mode Cancellation (CMC) + 기하학적 OOD 거절(U_geom)
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
    FrozenResNet18DualCam,
    LightweightTextEncoder,
    MultimodalMagneticKeyExtractor,
    _refine_key_with_contrastive_margin,
)


def preprocess_bgr_to_u8_nchw(frame_bgr: np.ndarray, size: int = 128) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> RGB uint8 (1, 3, size, size)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    chw = np.transpose(resized, (2, 0, 1)).astype(np.uint8)
    return torch.from_numpy(chw).unsqueeze(0)


def preprocess_bgr_to_norm_nchw(frame_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> ImageNet 정규화 float32 (1, 3, size, size) (ONNX 공용)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def estimate_cup_center(
    frame_bgr: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[int, int, int, int], bool]:
    """벨트 중앙 영역에서 전경 물체 중심 좌표(정규화 0~1), 바운딩 박스, 물체 존재 여부를 추정한다."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0.5, 0.5), (w // 4, h // 4, w // 2, h // 2), False

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (h * w * 0.01):
        return (0.5, 0.5), (w // 4, h // 4, w // 2, h // 2), False

    x, y, bw, bh = cv2.boundingRect(largest)
    cx = (x + bw * 0.5) / float(w)
    cy = (y + bh * 0.5) / float(h)
    return (round(cx, 4), round(cy, 4)), (int(x), int(y), int(bw), int(bh)), True


class DTRAVisualEncoder(nn.Module):
    """동결 ResNet-18 (L1-L3) + Stage-1 풀링 + Stage-2 중심화 격자/DoG 추출기.

    ONNX(opset 17) 내보내기 호환을 위해 128x128 입력의 고정 8x8 특징맵에 대해
    avg_pool2d를 명시적으로 사용한다.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            cam_encoder = FrozenResNet18DualCam()
            self.trunk = cam_encoder.trunk
        else:
            import torchvision.models as tv_models

            backbone = tv_models.resnet18(weights=None)
            self.trunk = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            )
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self, img_main_norm: torch.Tensor, img_wrist_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """정규화 RGB 텐서 (B, 3, 128, 128) 2개를 받아 Stage-1/Stage-2 시각 특징 반환."""
        fa = self.trunk(img_main_norm)  # (B, 256, 8, 8)
        fe = self.trunk(img_wrist_norm)  # (B, 256, 8, 8)

        # Stage-1: GAP(256) + 2x2(1024) per camera -> 2560
        fa_gap = fa.mean(dim=(2, 3))
        fa_p2 = F.avg_pool2d(fa, kernel_size=4, stride=4).flatten(1)
        fe_gap = fe.mean(dim=(2, 3))
        fe_p2 = F.avg_pool2d(fe, kernel_size=4, stride=4).flatten(1)
        vis_stage1 = torch.cat([fa_gap, fa_p2, fe_gap, fe_p2], dim=-1)

        # Stage-2 CMC: 중심화 2x2(1024) + 4x4(4096) + 3x3 DoG(256) + wrist 2x2(1024) -> 6400
        fa_c = fa - fa.mean(dim=(2, 3), keepdim=True)
        fa_c2 = F.avg_pool2d(fa_c, kernel_size=4, stride=4).flatten(1)
        fa_c4 = F.avg_pool2d(fa_c, kernel_size=2, stride=2).flatten(1)
        fa_dog = (fa - F.avg_pool2d(fa, kernel_size=3, stride=1, padding=1)).abs().mean(dim=(2, 3))
        fe_c = fe - fe.mean(dim=(2, 3), keepdim=True)
        fe_c2 = F.avg_pool2d(fe_c, kernel_size=4, stride=4).flatten(1)
        vis_stage2 = torch.cat([fa_c2, fa_c4, fa_dog, fe_c2], dim=-1)

        return vis_stage1, vis_stage2


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
        d_topo: int = 64,
        d_lang: int = 128,
        seed: int = 42,
        input_size: int = 128,
        u_geom_threshold: float = 0.08,
        pretrained_backbone: bool = True,
        onnx_path: Optional[str | Path] = None,
    ) -> None:
        self.d_topo = d_topo
        self.d_lang = d_lang
        self.seed = seed
        self.input_size = input_size
        self.u_geom_threshold = u_geom_threshold
        self.device = torch.device("cpu")

        self.text_encoder = LightweightTextEncoder(dim=d_lang)
        self.key_extractor = MultimodalMagneticKeyExtractor(
            d_topo=d_topo, d_lang=d_lang, seed=seed
        )
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

        # 등록된 클래스 슬롯 및 불변량 상태 버퍼
        self.class_names: List[str] = []
        self.class_meta: Dict[str, Dict[str, Any]] = {}
        self.keys: List[torch.Tensor] = []
        self.anchors: List[torch.Tensor] = []
        self.self_sims: List[float] = []
        self.val_queries: List[torch.Tensor] = []

        # 에피소드 내 임시 관성/상태 버퍼 (episode_snap_back 시 0.0000 초기화)
        self.episodic_trace = np.zeros(0, dtype=np.float32)
        self.prev_selected: Optional[int] = None

    def _extract_visual_features(
        self, frame_main_bgr: np.ndarray, frame_wrist_bgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        wrist = frame_main_bgr if frame_wrist_bgr is None else frame_wrist_bgr
        x_main = preprocess_bgr_to_norm_nchw(frame_main_bgr, size=self.input_size)
        x_wrist = preprocess_bgr_to_norm_nchw(wrist, size=self.input_size)

        if self.ort_session is not None:
            vis1, vis2 = self.ort_session.run(
                None, {"img_main": x_main, "img_wrist": x_wrist}
            )
            return vis1.astype(np.float32), vis2.astype(np.float32)

        assert self.visual_encoder is not None
        with torch.no_grad():
            vis1_t, vis2_t = self.visual_encoder(
                torch.from_numpy(x_main), torch.from_numpy(x_wrist)
            )
        return vis1_t.numpy().astype(np.float32), vis2_t.numpy().astype(np.float32)

    def _build_query_and_sig(
        self,
        frame_main_bgr: np.ndarray,
        frame_wrist_bgr: Optional[np.ndarray] = None,
        prompt: str = "",
        proprio: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vis1, vis2 = self._extract_visual_features(frame_main_bgr, frame_wrist_bgr)
        prop = (
            np.zeros((1, 12), dtype=np.float32)
            if proprio is None
            else np.asarray(proprio, dtype=np.float32).reshape(1, 12)
        )
        lang_t = self.text_encoder.encode(prompt, self.device).unsqueeze(0)
        lang_np = lang_t.numpy().astype(np.float32)

        w_vis = self.key_extractor.w_vis.numpy()
        w_prop = self.key_extractor.w_prop.numpy()
        w_lang = self.key_extractor.w_lang.numpy()
        w_rerank = self.key_extractor.w_rerank.numpy()

        kv = _l2_normalize_np(vis1 @ w_vis)
        kp = _l2_normalize_np(prop @ w_prop) if np.any(prop) else np.zeros_like(kv)
        kl = _l2_normalize_np(lang_np @ w_lang) if prompt else np.zeros_like(kv)

        # 카메라 단독 추론(prompt="") 시에도 일관되게 매칭되도록 1차 라우팅 키는 시각+고유수용성 기준,
        # 자연어 라벨은 lang_sig_bank에 저장해 명령 조건부 매칭 시 합산한다.
        if np.any(prop):
            q_stage1 = _l2_normalize_np(0.80 * kv + 0.20 * kp)
        else:
            q_stage1 = kv
        vis_sig = _l2_normalize_np(vis2 @ w_rerank)
        return q_stage1[0], vis_sig[0], prop[0], lang_np[0]

    def enroll_class(
        self,
        label: str,
        frames_bgr: Sequence[np.ndarray],
        prompt: str = "",
        is_cup: bool = True,
        has_lid: Optional[bool] = None,
        wrist_frames_bgr: Optional[Sequence[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """1~30장의 샘플 프레임만으로 기존 슬롯을 침범/수정하지 않고 새 클래스 슬롯을 등록한다."""
        if not frames_bgr:
            raise ValueError("등록할 샘플 프레임이 최소 1장 이상 필요합니다.")
        if label in self.class_names:
            raise ValueError(f"이미 등록된 클래스입니다: {label}")

        queries, vis_sigs, prop_sigs, lang_sigs = [], [], [], []
        for idx, frame in enumerate(frames_bgr):
            w_frame = (
                wrist_frames_bgr[idx]
                if wrist_frames_bgr is not None and idx < len(wrist_frames_bgr)
                else None
            )
            q1, vs, ps, ls = self._build_query_and_sig(frame, w_frame, prompt=prompt)
            queries.append(q1)
            vis_sigs.append(vs)
            prop_sigs.append(ps)
            lang_sigs.append(ls)

        q_tensor = torch.from_numpy(np.stack(queries, axis=0))
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
        val_q = getattr(refined_key, "_val_queries", q_tensor[:96]).detach().clone()
        refined_key = refined_key.detach().clone()

        self.class_names.append(label)
        self.class_meta[label] = {
            "prompt": prompt,
            "is_cup": bool(is_cup),
            "has_lid": has_lid,
        }
        self.keys.append(refined_key)
        self.anchors.append(anchor_mu)
        self.self_sims.append(self_sim)
        self.val_queries.append(val_q)

        self.key_extractor.register_slot_signature(
            torch.from_numpy(np.stack(vis_sigs, axis=0)),
            torch.from_numpy(np.stack(prop_sigs, axis=0)),
            torch.from_numpy(np.stack(lang_sigs, axis=0)),
            self.device,
        )
        self.episodic_trace = np.zeros(len(self.class_names), dtype=np.float32)
        return {"label": label, "num_samples": len(frames_bgr), **stats}

    def predict(
        self,
        frame_main_bgr: np.ndarray,
        frame_wrist_bgr: Optional[np.ndarray] = None,
        prompt: str = "",
        proprio: Optional[np.ndarray] = None,
    ) -> VisionPrediction:
        """단일 프레임을 Stage-1 + 조건부 Stage-2 CMC + U_geom 게이트로 판별한다."""
        t0 = time.perf_counter()
        center_xy, bbox, has_foreground = estimate_cup_center(frame_main_bgr)

        if not self.keys:
            raise RuntimeError("등록된 클래스 슬롯이 없습니다. enroll_class()를 먼저 호출하세요.")

        if not has_foreground:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return VisionPrediction(
                status="EMPTY",
                is_cup=False,
                has_lid=None,
                confidence=0.0,
                u_geom=1.0,
                margin=0.0,
                center_xy=center_xy,
                bbox=bbox,
                latency_ms=latency_ms,
                backend=self.backend,
            )

        q1, vis_sig, prop_sig, lang_sig = self._build_query_and_sig(
            frame_main_bgr, frame_wrist_bgr, prompt=prompt, proprio=proprio
        )
        key_bank = np.stack([k.numpy() for k in self.keys], axis=0)  # (K, 64)
        s1 = key_bank @ q1  # (K,)

        k_count = len(self.keys)
        order = np.argsort(-s1)
        top1_idx = int(order[0])
        top1_score = float(s1[top1_idx])
        top2_score = float(s1[int(order[1])]) if k_count >= 2 else -1.0
        margin = top1_score - top2_score

        # 2단계 Top-M Common-Mode Cancellation (후보 간 공통 배경/조명 차감 후 DoG 격자 재순위)
        vis_bank_all = self.key_extractor.vis_sig_bank.numpy()  # (K, 256)
        vis_sims_all = vis_bank_all @ vis_sig  # (K,)
        selected_idx = top1_idx
        if k_count >= 2 and margin < 0.12:
            m_eff = min(5, k_count)
            cand_idx = order[:m_eff]
            vis_bank = vis_bank_all[cand_idx]
            mu_vis = vis_bank.mean(axis=0, keepdims=True)
            vis_res = _l2_normalize_np(vis_bank - mu_vis)
            q_vis_res = _l2_normalize_np(vis_sig[None, :] - mu_vis)
            s_vis_cm = (q_vis_res @ vis_res.T)[0]

            s_comb = s1[cand_idx] + 0.25 * s_vis_cm
            best_local = int(np.argmax(s_comb))
            selected_idx = int(cand_idx[best_local])

        # 기하학적 OOD/미등록 물체 판정:
        # 배경이 제거된 Stage-2 중심화 격자+DoG 유사도(vis_sim)와 Stage-1 키 유사도(top1_score)를 결합해
        # 벨트 배경이 동일해도 물체 형태가 다르면 확실히 OTHER_TRASH로 분리한다.
        sel_s1 = float(s1[selected_idx])
        sel_vis_sim = float(vis_sims_all[selected_idx])
        combined_sim = 0.35 * sel_s1 + 0.65 * sel_vis_sim
        u_geom = float(max(0.0, 1.0 - combined_sim))
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
            confidence=round(combined_sim, 4),
            u_geom=round(u_geom, 4),
            margin=round(max(0.0, margin), 4),
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
        """학습/등록된 초경량 슬롯 뱅크(.npz) 저장 (원본 이미지 미저장)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(
            {
                "class_names": self.class_names,
                "class_meta": self.class_meta,
                "d_topo": self.d_topo,
                "d_lang": self.d_lang,
                "seed": self.seed,
                "input_size": self.input_size,
                "u_geom_threshold": self.u_geom_threshold,
            },
            ensure_ascii=False,
        )
        np.savez(
            out_path,
            meta_json=np.array([meta_json]),
            keys=np.stack([k.numpy() for k in self.keys], axis=0),
            anchors=np.stack([a.numpy() for a in self.anchors], axis=0),
            self_sims=np.array(self.self_sims, dtype=np.float32),
            vis_sig_bank=self.key_extractor.vis_sig_bank.numpy(),
            prop_sig_bank=self.key_extractor.prop_sig_bank.numpy(),
            lang_sig_bank=self.key_extractor.lang_sig_bank.numpy(),
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
        self.keys = [torch.from_numpy(keys_arr[i].copy()) for i in range(len(self.class_names))]
        self.anchors = [torch.from_numpy(anchors_arr[i].copy()) for i in range(len(self.class_names))]
        self.self_sims = [float(x) for x in self_sims_arr]
        self.val_queries = [a.unsqueeze(0) for a in self.anchors]

        self.key_extractor.reset_slot_signatures()
        vis_b = data["vis_sig_bank"]
        prop_b = data["prop_sig_bank"]
        lang_b = data["lang_sig_bank"]
        for i in range(len(self.class_names)):
            self.key_extractor.register_slot_signature(
                torch.from_numpy(vis_b[i].copy()),
                torch.from_numpy(prop_b[i].copy()),
                torch.from_numpy(lang_b[i].copy()),
                self.device,
            )
        self.episodic_trace = np.zeros(len(self.class_names), dtype=np.float32)
