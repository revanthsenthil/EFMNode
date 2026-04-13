import os
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger

from core.inference.inference_engine import InferenceEngine
from core.inference.pytorch_engine import PyTorchEngine
from core.inference.tensorrt_engine import TensorRTEngine

def _resolve_trt_engine_paths(ckpt_path: str) -> tuple[str, str]:
    engine_pairs = [
        (
            "galaxea_zero_encoder_opt.fp16.engine",
            "galaxea_zero_predictor_opt.fp16.engine",
        ),
        ("prefill.fp16.engine", "decode.fp16.engine"),
    ]

    for encoder_name, predictor_name in engine_pairs:
        encoder_path = os.path.join(ckpt_path, encoder_name)
        predictor_path = os.path.join(ckpt_path, predictor_name)
        if os.path.exists(encoder_path) and os.path.exists(predictor_path):
            logger.info(
                "Using TensorRT engines: {} and {}",
                encoder_name,
                predictor_name,
            )
            return encoder_path, predictor_path

    # Fall back to the legacy names so downstream errors still point at a
    # concrete expected path when no known engine pair exists yet.
    return (
        os.path.join(ckpt_path, engine_pairs[0][0]),
        os.path.join(ckpt_path, engine_pairs[0][1]),
    )

def create_inference_engine(
    config: Dict[str, Any],
    cfg: DictConfig,
    use_trt: bool = False,
    trt_config: Dict[str, Any] = None,
    role: Optional[str] = None
) -> InferenceEngine:
    if use_trt:
        logger.info("Creating TensorRT inference engine")
        default_trt_config = {}
        ckpt_path = config["model"]["ckpt_dir"]
        encoder_path, predictor_path = _resolve_trt_engine_paths(ckpt_path)
        default_trt_config["encoder_path"] = encoder_path
        default_trt_config["predictor_path"] = predictor_path
        default_trt_config["device"] = "cuda:0"
        default_trt_config["precision"] = "fp16"
        default_trt_config["plugin_path"] = os.path.join(ckpt_path, "gemma_rmsnorm.so")
        default_trt_config["use_cuda_graph"] = True
        if trt_config is None:
            trt_config = default_trt_config
        return TensorRTEngine(
            config=config,
            cfg=cfg,
            encoder_path=trt_config.get("encoder_path"),
            predictor_path=trt_config.get("predictor_path"),
            device=trt_config.get("device", "cuda:0"),
            precision=trt_config.get("precision", "fp16"),
            plugin_path=trt_config.get("plugin_path"),
            use_cuda_graph=trt_config.get("use_cuda_graph", True),
        )
    elif use_trt is False:
        logger.info("Creating PyTorch inference engine")
        return PyTorchEngine(config, cfg)
