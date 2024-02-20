""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import Union, List

import torch

from .model import build_model_from_openai_state_dict
from .pretrained import (
    get_pretrained_url,
    list_pretrained_tag_models,
    download_pretrained,
)

__all__ = ["list_openai_models", "load_openai_model"]

#사용 가능한 OpenAI CLIP 모델의 이름을 반환
def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models("openai")

# CLIP 모델을 로드하고, 해당 모델의 텍스트 사전 훈련 부분을 보존하고, CLAP 모델에 설정
def load_openai_model(
    name: str,
    model_cfg,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
    cache_dir=os.path.expanduser("~/.cache/clip"),
    enable_fusion: bool = False,
    fusion_type: str = "None",
):
    """
    Parameters
    ----------
    name : str
        `clip.available_models()`에 나열된 모델 이름 또는 상태 사전을 포함하는 모델 체크포인트의 경로
    device : Union[str, torch.device]
        로드된 모델을 배치할 장치
    jit : bool
        최적화된 JIT 모델(기본값) 또는 더 편집 가능한 비-JIT 모델을 로드할지 여부

    Returns
    -------
    model : torch.nn.Module
        CLAP 모델
    preprocess : Callable[[PIL.Image], torch.Tensor]
        PIL 이미지를 반환된 모델이 입력으로 취할 수 있는 텐서로 변환하는 torchvision 변환
    """
    
    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained(
            get_pretrained_url(name, "openai"), root=cache_dir
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {list_openai_models()}"
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict(), model_cfg, enable_fusion, fusion_type
            ).to(device)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(
                sd, model_cfg, enable_fusion, fusion_type
            ).to(device)

        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_audio)
    patch_device(model.encode_text)

    # # CPU에서 dtype을 float32로 패치
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_audio)
        patch_float(model.encode_text)
        model.float()

    model.audio_branch.audio_length = model.audio_cfg.audio_length
    return model
