import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import torch

# 모듈에서 클래스,함수 가져오기
from .model import CLAP, convert_weights_to_fp16
from .openai import load_openai_model
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform

# 모델 설정 경로 
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

# 자연순서 정렬 key 함수
def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]

# _MODEL_CONFIGS 딕셔너리 생성 함수 
def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
		# 경로 안의 파일, 디렉토리에 관해 확인 후 config_files 리스트에 json파일 추가
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))
		#config_files리스트에서 
    for cf in config_files:
        if os.path.basename(cf)[0] == ".":
            continue  # Ignore hidden files
				#.으로 시작하지 않는 파일 열기 
        with open(cf, "r") as f:
            model_cfg = json.load(f)
						#파일에 "embed_dim", "audio_cfg", "text_cfg" 키가 모두 존재할 때 
            if all(a in model_cfg for a in ("embed_dim", "audio_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg 
						# 파일이름을 키, 모델 구성을 value로 하는 dict에 추가
#파일 정렬
    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }

# 모델 구성 파일 레지스트리 초기화
_rescan_model_configs()  

'''
check point = 주어진 딥러닝 모델의 상태를 나타내는 데이터 파일. 주로 모델의 weight와 bias 등을 저장
나중에 복원하여 모델을 이전 상태로 복구하거나, 이어서 학습을 진행하는 데 사용. 
'''

#체크포인트를 로드하여 모델의 상태 반환하는 함수   
def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    #체크 포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:#state_dict의 첫 번째 항목이 'module'로 시작
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}#key의 7번째부터를 state_dict으로
    # for k in state_dict:
    #     if k.startswith('transformer'):
    #         v = state_dict.pop(k)
    #         state_dict['text_branch.' + k[12:]] = v
    return state_dict

#CLAP 모델 생성 함수
def create_model(
    amodel_name: str,
    tmodel_name: str,
    pretrained: str = "",
    precision: str = "fp32",#가중치 정밀도
    device: torch.device = torch.device("cpu"),#장치 기본값 cpu
    jit: bool = False,
    force_quick_gelu: bool = False,
    openai_model_cache_dir: str = os.path.expanduser("~/.cache/clip"),#openai모델 캐시 디렉토리
    skip_params=True,
    pretrained_audio: str = "",# 사전 훈련된 오디오 모델 경로
    pretrained_text: str = "",# 사전 훈련된 텍스트 모델 경로
    enable_fusion: bool = False,# 퓨전 사용 여부 (기본값: False)
    fusion_type: str = "None"# 퓨전 타입 (기본값: "None")
    # pretrained_image: bool = False,
):
    # ViT 이름에 '/'가 있는 경우 '-'로 변경
    amodel_name = amodel_name.replace(
        "/", "-"
    )  # 원래의 pretrained 값을 소문자로 변경
    pretrained_orig = pretrained
    pretrained = pretrained.lower()
    # pretrained가 "openai"인 경우
    if pretrained == "openai":
        if amodel_name in _MODEL_CONFIGS:# _MODEL_CONFIGS에 amodel_name이 있는지 확인
            logging.info(f"Loading {amodel_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])# OpenAI에서 사전 훈련된 모델을 로드하기 위한 모델 구성 복사
        else:
            logging.error( # 모델 구성을 찾을 수 없는 경우 에러 발생
                f"Model config for {amodel_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {amodel_name} not found.")
         # 로그에 정보 출력
        logging.info(f"Loading pretrained ViT-B-16 text encoder from OpenAI.")
        # Hard Code in model name
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model = load_openai_model(# OpenAI에서 제공하는 모델을 로드
            "ViT-B-16",
            model_cfg,
            device=device,
            jit=jit,
            cache_dir=openai_model_cache_dir,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type,
        )
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()# 정밀도가 "amp" 또는 "fp32"인 경우 모델을 float으로 변환
    # pretrained가 "openai"가 아닌 경우
    else:
        if amodel_name in _MODEL_CONFIGS:# 모델 구성이 있는 경우 정보 로깅
            logging.info(f"Loading {amodel_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            logging.error(# 모델 구성을 찾을 수 없는 경우 에러 발생
                f"Model config for {amodel_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {amodel_name} not found.")

        if force_quick_gelu:
            # QuickGELU 사용 설정
            model_cfg["quick_gelu"] = True 

        # 텍스트 모델 유형,퓨전 설정,CLAP 모델 생성
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model_cfg["enable_fusion"] = enable_fusion
        model_cfg["fusion_type"] = fusion_type
        model = CLAP(**model_cfg)

        if pretrained:# 사전 훈련된 가중치가 있는 경우
            checkpoint_path = ""# 체크포인트 경로 초기화
            url = get_pretrained_url(amodel_name, pretrained)# 사전 훈련된 가중치 다운로드 URL 가져오기
            if url:
                checkpoint_path = download_pretrained(url, root=openai_model_cache_dir)# 사전 훈련된 가중치 다운로드
            elif os.path.exists(pretrained_orig):
                checkpoint_path = pretrained_orig
            if checkpoint_path:# 체크포인트 경로가 존재
                logging.info( # 정보 로깅
                    f"Loading pretrained {amodel_name}-{tmodel_name} weights ({pretrained})."
                )
                ckpt = load_state_dict(checkpoint_path, skip_params=True) # 가중치 로드
                model.load_state_dict(ckpt)# 모델에 가중치 적용
                param_names = [n for n, p in model.named_parameters()]# 모델의 파라미터 이름 가져오기
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
            else:# 체크포인트 경로가 존재하지 않는 경우
                logging.warning(# 경고 로깅
                    f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
                )
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
                )

        if pretrained_audio:# 사전 훈련된 오디오 가중치가 있는 경우
            if amodel_name.startswith("PANN"):# 모델이 PANN
                if "Cnn14_mAP" in pretrained_audio:# 공식 체크포인트
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu") # 오디오 체크포인트 로드
                    audio_ckpt = audio_ckpt["model"]# 모델 파트 추출
                    keys = list(audio_ckpt.keys())# 모든 키 가져오기
                    for key in keys:# 각 키에 대해 반복
                        if (
                            "spectrogram_extractor" not in key
                            and "logmel_extractor" not in key
                        ):
                            v = audio_ckpt.pop(key) # 키 삭제
                            audio_ckpt["audio_branch." + key] = v # 새로운 키 생성
                elif os.path.basename(pretrained_audio).startswith(
                    "PANN"
                ):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")# 오디오 체크포인트 로드
                    audio_ckpt = audio_ckpt["state_dict"]# 상태 사전 추출
                    keys = list(audio_ckpt.keys()) # 모든 키 가져오기
                    for key in keys:
                        if key.startswith("sed_model"): # 키가 "sed_model"로 시작
                            v = audio_ckpt.pop(key)# 키 삭제
                            audio_ckpt["audio_branch." + key[10:]] = v# 새로운 키 생성
                elif os.path.basename(pretrained_audio).startswith(
                    "finetuned"
                ):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                else:
                    raise ValueError("Unknown audio checkpoint")
            elif amodel_name.startswith("HTSAT"):# 모델이 HTSAT
                if "HTSAT_AudioSet_Saved" in pretrained_audio: # 공식 체크포인트
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")# 오디오 체크포인트 로드
                    audio_ckpt = audio_ckpt["state_dict"]# 모델 파트 추출
                    keys = list(audio_ckpt.keys())# 모든 키 가져오기
                    for key in keys:# 각 키에 대해 반복
                        if key.startswith("sed_model") and (# 키가 "sed_model"로 시작하는 경우
                            "spectrogram_extractor" not in key
                            and "logmel_extractor" not in key
                        ):  
                            v = audio_ckpt.pop(key)# 키 삭제
                            audio_ckpt["audio_branch." + key[10:]] = v# 새로운 키 생성
                elif os.path.basename(pretrained_audio).startswith(
                    "HTSAT"
                ):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")# 오디오 체크포인트 로드
                    audio_ckpt = audio_ckpt["state_dict"]
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith("sed_model"):
                            v = audio_ckpt.pop(key)
                            audio_ckpt["audio_branch." + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                    "finetuned"
                ):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")# 오디오 체크포인트 로드
                else:
                    raise ValueError("Unknown audio checkpoint")# 알 수 없는 오디오 체크포인트
            else:
                raise f"this audio encoder pretrained checkpoint is not support"

            model.load_state_dict(audio_ckpt, strict=False)# 오디오 가중치 모델에 로드 
            logging.info(
                f"Loading pretrained {amodel_name} weights ({pretrained_audio})."
            )
            param_names = [n for n, p in model.named_parameters()]
            for n in param_names:
                print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")

        model.to(device=device)# 장치 설정
        if precision == "fp16":# 가중치 정밀도가 "fp16"
            assert device.type != "cpu"
            convert_weights_to_fp16(model)# 가중치를 FP16으로 변환

        if jit: # JIT 컴파일 설정
            model = torch.jit.script(model)# 모델을 스크립트로

    return model, model_cfg# 모델과 모델 구성 반환

# 모델 생성, 이미지 전처리 파이프라인 생성
def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    # pretrained_image: bool = False,
):
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        jit,
        force_quick_gelu=force_quick_gelu,
        # pretrained_image=pretrained_image
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val

# 사용 가능한 모델 아키텍처를 열거
def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())

#모델 구성 경로나 파일을 추가하고 레지스트리를 업데이트
def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
