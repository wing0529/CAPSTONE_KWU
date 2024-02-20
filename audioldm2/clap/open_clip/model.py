""" CLAP Model

Adapted from CLIP: https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
Adapted to the Audio Task.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import logging
from .utils import freeze_batch_norm_2d

from .pann_model import create_pann_model
from .htsat import create_htsat_model
from transformers import BertModel, RobertaModel, BartModel, RobertaConfig

#다중 레이어 퍼셉트론(MLP)을 정의
class MLPLayers(nn.Module):
    #선형 레이어, 활성화 함수, 드롭아웃을 설정
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)
    #입력을 받아서 MLP 레이어를 통과
    def forward(self, X):
        X = self.sequential(X)
        return X

#ResNet의 Bottleneck 블록을 정의 : 모델이 깊어질 때 적은 파라미터로 더 깊은 네트워크를 학습할 수 있도록 설계
#ResNet 모델을 구축할 때 Bottleneck 블록을 여러 번 사용하여 네트워크를 구성할 수 있음
class Bottleneck(nn.Module):
    expansion = 4#블록 내의 차원 확장 비율
    #Bottleneck 블록의 구성 요소를 초기화
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        #1x1, 3x3, 1x1 컨볼루션 레이어와 배치 정규화 레이어
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        
        #입력과 출력의 차원이 다를 경우 다운샘플링 레이어를 추가로 정의
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )
    # 주어진 입력에 대해 Bottleneck 블록 순전파
    def forward(self, x: torch.Tensor):
        identity = x
        #컨볼루션 및 배치 정규화
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        #다운샘플링이 있는 경우에는 입력을 변경
        if self.downsample is not None:
            identity = self.downsample(x)
        #ReLU 활성화 함수를 적용하여 반환
        out += identity
        out = self.relu(out)
        return out

# 2D 어텐션 풀링 레이어를 정의 -> self-attention을 수행하고, 결과를 출력
class AttentionPool2d(nn.Module):
    #어텐션 풀링 레이어의 구성 요소를 초기화
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    #어텐션 풀링을 수행   
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # height * width, batch_size, channels) 형태로 변환-> self attention 수행 -> 어텐션의 결과는 출력 프로젝션 레이어를 통해 변환된 후 반환
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


# ModifiedResNet 클래스 생성
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    #클래스 초기화    
    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        #Stem 레이어 변경 : 3개의 "stem" 컨볼루션 레이어를 사용
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)#Batch Normalization이 적용
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)#ReLU 활성화 함수가 사용

        # residual layers
        #Bottleneck 클래스로 구현
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)
        '''
        layers:각 레이어의 블록 수를 포함하는 리스트
        output_dim:모델의 출력 차원
        heads:QKV 어텐션 풀링 레이어에서 사용되는 어텐션 헤드의 수
        image_size: 이미지의 크기 
        width:모델의 초기 너비
        '''
        self.init_parameters()
   
    #ResNet의 각 residual layer를 생성
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)] # 새로운 Bottleneck 레이어 생성

        self._inplanes = planes * Bottleneck.expansion #inplanes 업데이트
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers) # 생성된 레이어를 시퀀셜 레이어로 반환
    #모델의 파라미터를 초기화
    def init_parameters(self):
        if self.attnpool is not None:#attnpool 레이어가 존재
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std) # 가중치 초기화
            nn.init.normal_(self.attnpool.k_proj.weight, std=std) # 가중치 초기화
            nn.init.normal_(self.attnpool.v_proj.weight, std=std) # 가중치 초기화
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
 # 가중치 초기화
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:# 각 residual 블록
            for name, param in resnet_block.named_parameters(): # 블록 내의 각 파라미터
                if name.endswith("bn3.weight"):# 만약 bn3.weight
                    nn.init.zeros_(param)# 0으로 초기화
    #모델의 파라미터를 잠금 처리
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False # 그래디언트 계산 비활성화
        if freeze_bn_stats:
            freeze_batch_norm_2d(self) # Batch Normalization 통계 고정
    #입력 이미지를 받아서 stem 레이어를 통과시키고 평균 풀링을 적용
    def stem(self, x):
        for conv, bn in [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
        ]: # 각각의 컨볼루션 레이어와 Batch Normalization
            x = self.relu(bn(conv(x)))# 컨볼루션, ReLU, Batch Normalization 순서로 연산
        x = self.avgpool(x) # 평균 풀링 적용
        return x
    #입력을 모델에 전달하고 출력을 생성
    def forward(self, x):
        x = self.stem(x) # 입력 이미지에 stem 레이어 적용
        x = self.layer1(x)# residual layer 적용
        x = self.layer2(x)# residual layer 적용
        x = self.layer3(x)# residual layer 적용
        x = self.layer4(x)# residual layer 적용
        x = self.attnpool(x) # QKV attention pooling 적용

        return x # 최종 출력 반환

#PyTorch의 LayerNorm 클래스를 상속하여 fp16(반정밀도)을 처리
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    #입력 데이터에 LayerNorm을 적용
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)#원본 데이터 타입으로 변환하여 반환

# QuickGELU 활성화 함수 계산 및 반환
class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)## QuickGELU 활성화 함수 계산 및 반환

# Transformer 구조를 구현 :  Residual Attention Block
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()
        ## Multi-head Attention 
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # Layer Normalization 
        self.ln_1 = LayerNorm(d_model)
        # MLP (Multi-Layer Perceptron)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),# Fully Connected Layer
                    ("gelu", act_layer()),# 활성화 함수 (GELU)
                    ("c_proj", nn.Linear(d_model * 4, d_model)),# Fully Connected Layer
                ]
            )
        )
        ## Layer Normalization 레이어
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Multi-head Attention 수행
        
        Args:
            x: 입력 텐서
            attn_mask: 어텐션 마스크
            
        Returns:
            어텐션 결과 텐서
        """
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Residual Attention Block의 forward 연산
        
        Args:
            x: 입력 텐서
            attn_mask: 어텐션 마스크
            
        Returns:
            출력 텐서
        """
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)# 입력과 어텐션 결과를 더함 (Residual Connection)
        x = x + self.mlp(self.ln_2(x))# 입력과 MLP 결과를 더함 (Residual Connection)
        return x

#여러 개의 ResidualAttentionBlock을 쌓아 올린 전체 Transformer 모델을 정의
class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        # ResidualAttentionBlock을 layers 개수만큼 반복하여 리스트로 저장
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Transformer의 forward 연산
        
        Args:
            x: 입력 텐서
            attn_mask: 어텐션 마스크
            
        Returns:
            출력 텐서
        """
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x

#VisualTransformer 모델을 정의
class VisualTransformer(nn.Module):
    #VisualTransformer 클래스 생성자 : 모델의 구조를 초기화
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
         # 이미지를 패치로 분할하기 위한 컨볼루션 레이어
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        # 클래스 임베딩
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 위치 임베딩
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        # Layer Normalization 
        self.ln_pre = LayerNorm(width)
        # Transformer
        self.text_branch = Transformer(width, layers, heads, act_layer=act_layer)
        # Layer Normalization 
        self.ln_post = LayerNorm(width)
        # Projection
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    #모델의 파라미터를 잠금
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False
            
    #입력에 대해 모델의 순전파 연산을 정의
    def forward(self, x: torch.Tensor):
        '''
        1. 입력 이미지에 컨볼루션 레이어를 적용하여 이미지를 패치로 분할
        2. 각 패치에 클래스 임베딩을 추가하고, 위치 임베딩을 더함
        3. 입력 패치에 Layer Normalization을 적용
        4. Transformer 모델에 입력을 전달하여 각 패치의 임베딩을 계산
        5. 출력 임베딩에 Layer Normalization을 적용하고, 필요하다면 Projection 레이어를 적용하여 최종 출력을 생성
    '''
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        
        # 위치 임베딩
        x = x + self.positional_embedding.to(x.dtype)
        
        # 입력 패치에 Layer Normalization
        x = self.ln_pre(x)
        
        # 입력 차원 순서 변경 (NLD -> LND)
        x = x.permute(1, 0, 2) 
         # Transformer 모델 실행
        x = self.text_branch(x)
        
        # 출력 차원 순서 변경 (LND -> NLD)
        x = x.permute(1, 0, 2) 
        # 출력에 Layer Normalization
        x = self.ln_post(x[:, 0, :])
        # Projection
        if self.proj is not None:
            x = x @ self.proj

        return x


# 데이터 클래스 CLAPVisionCfg 정의
@dataclass
class CLAPVisionCfg:
    #비전 트랜스포머의 레이어 수
    layers: Union[Tuple[int, int, int, int], int] = 12
    #비전 트랜스포머의 너비
    width: int = 768
    #이미지를 분할할 때 사용되는 패치의 크기
    patch_size: int = 16
    #입력 이미지의 크기
    image_size: Union[Tuple[int, int], int] = 224
    #Timm 라이브러리에서 사용할 모델의 이름
    timm_model_name: str = (
        None  
    )
    #Timm 모델의 사전 훈련 가중치 사용 여부
    timm_model_pretrained: bool = (
        False  
    )
    # Timm 모델의 특성 풀링 방법
    timm_pool: str = (
        "avg"  
    )
    #Timm 모델의 출력에 적용할 선형 변환 방법
    timm_proj: str = (
        "linear"
    )

# Audio Config Class CLAP의 다른 모듈에 대한 설정을 저장
@dataclass
class CLAPAudioCfp:
    model_type: str = "PANN"
    model_name: str = "Cnn14"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000

# CLAP의 다른 모듈에 대한 설정을 저장
@dataclass
class CLAPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    model_type: str


class CLAP(nn.Module):
    #주어진 인수를 사용하여 CLAP 모델을 초기화
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfp,
        text_cfg: CLAPTextCfg,
        quick_gelu: bool = False,
        enable_fusion: bool = False,
        fusion_type: str = "None",
        joint_embed_shape: int = 512,
        mlp_act: str = "relu",
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfp(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if mlp_act == "relu":
            mlp_act_layer = nn.ReLU()
        elif mlp_act == "gelu":
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # audio branch
        # audio branch parameters
        if audio_cfg.model_type == "PANN":
            self.audio_branch = create_pann_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(
                audio_cfg, enable_fusion, fusion_type
            )
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        # text branch
        # text branch parameters
        if text_cfg.model_type == "transformer":
            self.text_branch = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, text_cfg.width)
            )
            self.ln_final = LayerNorm(text_cfg.width)
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(text_cfg.width, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bert":
            self.text_branch = BertModel.from_pretrained("bert-base-uncased")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "roberta":
            self.text_branch = RobertaModel(
                RobertaConfig.from_pretrained("roberta-base")
            )
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bart":
            self.text_branch = BartModel.from_pretrained("facebook/bart-base")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        else:
            logging.error(f"Model config for {text_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {text_cfg.model_type} not found.")
        self.text_branch_type = text_cfg.model_type
        # text branch parameters

        # audio branch parameters
        self.audio_transform = MLPLayers(
            units=[
                self.joint_embed_shape,
                self.joint_embed_shape,
                self.joint_embed_shape,
            ],
            dropout=0.1,
        )

        # below here is text branch parameters

        # ============================================================================================================
        self.audio_projection = nn.Sequential(
            nn.Linear(embed_dim, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
        )

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_text_branch_parameters()
    #텍스트 분기에 대한 초기 파라미터를 설정
    def init_text_branch_parameters(self):
        #트랜스포머 모델인 경우, 초기화에 필요한 파라미터를 설정
        if self.text_branch_type == "transformer":
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (self.text_branch.width**-0.5) * (
                (2 * self.text_branch.layers) ** -0.5
            )
            attn_std = self.text_branch.width**-0.5
            fc_std = (2 * self.text_branch.width) ** -0.5
            for block in self.text_branch.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #BERT, RoBERTa, BART 등 특정 모델의 경우 해당 모델의 초기화를 수행
        if self.text_branch_type == "bert" or self.text_branch_type == "roberta":
            self.text_branch.embeddings.word_embeddings.weight.shape[-1]
        elif self.text_branch_type == "bart":
            self.text_branch.shared.weight.shape[-1]
        else:
            self.text_branch.width
        #로짓 스케일링 파라미터를 설정
        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))

        # deprecated
        # if hasattr(self.visual, 'init_parameters'):
        # self.visual.init_parameters()

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=width**-0.5)
    #어텐션 마스크를 생성
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    #오디오를 인코딩하여 오디오 피처를 반환
    def encode_audio(self, audio, device):
        return self.audio_branch(
            audio, mixup_lambda=None, device=device #주어진 오디오 입력을 사용하여 오디오 분기의 모델을 통해 오디오 피처를 추출
        )  # mix lambda needs to add

    # def list_of_dict_of_tensor2dict_of_tensor(self, x, device):
    #     tmp = {}
    #     for k in x[0].keys():
    #         tmp[k] = []
    #         for i in range(len(x)):
    #             tmp[k].append(x[i][k][:77])
    #     for k in x[0].keys():
    #         tmp[k] = torch.tensor(tmp[k]).to(device=device, non_blocking=True)
    #     return tmp
    #텍스트를 인코딩하여 텍스트 피처를 반환
    def encode_text(self, text, device):
        if self.text_branch_type == "transformer":
            text = text.to(device=device, non_blocking=True)
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_branch(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        elif self.text_branch_type == "bert":
            # text = self.list_of_dict_of_tensor2dict_of_tensor(text, device)
            # text = BatchEncoding(text)
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
                token_type_ids=text["token_type_ids"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "bart":
            x = torch.mean(
                self.text_branch(
                    input_ids=text["input_ids"].to(device=device, non_blocking=True),
                    attention_mask=text["attention_mask"].to(
                        device=device, non_blocking=True
                    ),
                )["encoder_last_hidden_state"],
                axis=1,
            )
            x = self.text_projection(x)
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x
    #CLAP 모델의 순방향 전달을 정의 특성을 추출하고, MLP 레이어를 통해 변환된 특성을 반환
    def forward(self, audio, text, device=None):
        """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: torch.Tensor () // need to add
            the text token input
        """
        if device is None:
            if audio is not None:
                device = audio.device
            elif text is not None:
                device = text.device
        if audio is None and text is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            return self.encode_text(text, device=device)
        elif text is None:
            return self.audio_projection(
                self.encode_audio(audio, device=device)["embedding"]
            )
        audio_features = self.audio_projection(
            self.encode_audio(audio, device=device)["embedding"]
        )
        audio_features = F.normalize(audio_features, dim=-1)

        text_features = self.encode_text(text, device=device)
        # print("text_features", text_features)
        # print("text_features.shape", text_features.shape)
        # print("text_features.type", type(text_features))
        text_features = F.normalize(text_features, dim=-1)

        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )
    #로짓 스케일링 값을 반환
    def get_logit_scale(self):
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()
    #입력 데이터로부터 텍스트 임베딩을 반환
    def get_text_embedding(self, data):
        """Get the text embedding from the model

        Parameters
        ----------
        data: torch.Tensor
            a tensor of text embedding

        Returns
        ----------
        text_embed: torch.Tensor
            a tensor of text_embeds (N, D)

        """
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
        text_embeds = self.encode_text(data, device=device)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds
    #입력 데이터로부터 오디오 임베딩을 반환
    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
        device = next(self.parameters()).device
        # input_dict = {}
        # keys = data[0].keys()
        # for k in keys:
        #     input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(
        #         device
        #     )
        audio_embeds = self.audio_projection(
            self.encode_audio(data, device=device)["embedding"]
        )
        audio_embeds = F.normalize(audio_embeds, dim=-1)

        return audio_embeds
    #추론 모드에서 오디오를 전달하고 오디오 임베딩을 생성 : 모델에 따라 오디오 입력을 처리하고 오디오 임베딩을 반환
    def audio_infer(self, audio, hopsize=None, device=None):
        """Forward one audio and produce the audio embedding

        Parameters
        ----------
        audio:  (audio_length)
            the time-domain audio input, notice that it must be only one input
        hopsize: int
            the overlap hopsize as the sliding window

        Returns
        ----------
        output_dict: {
            key: [n, (embedding_shape)] if "HTS-AT"
            or
            key: [(embedding_shape)] if "PANN"
        }
            the list of key values of the audio branch

        """

        assert not self.training, "the inference mode must be run at eval stage"
        output_dict = {}
        # PANN
        if self.audio_cfg.model_type == "PANN":
            audio_input = audio.unsqueeze(dim=0)
            output_dict[key] = self.encode_audio(audio_input, device=device)[
                key
            ].squeeze(dim=0)
        elif self.audio_cfg.model_type == "HTSAT":
            # repeat
            audio_len = len(audio)
            k = self.audio_cfg.clip_samples // audio_len
            if k > 1:
                audio = audio.repeat(k)
                audio_len = len(audio)

            if hopsize is None:
                hopsize = min(hopsize, audio_len)

            if audio_len > self.audio_cfg.clip_samples:
                audio_input = [
                    audio[pos : pos + self.audio_cfg.clip_samples].clone()
                    for pos in range(
                        0, audio_len - self.audio_cfg.clip_samples, hopsize
                    )
                ]
                audio_input.append(audio[-self.audio_cfg.clip_samples :].clone())
                audio_input = torch.stack(audio_input)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key]
            else:
                audio_input = audio.unsqueeze(dim=0)
                output_dict[key] = self.encode_audio(audio_input, device=device)[
                    key
                ].squeeze(dim=0)

        return output_dict

#주어진 모델의 일부 파라미터를 FP16 형식으로 변환
def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)#레이어의 가중치와 편향을 FP16로 변환


#OpenAI의 사전 학습된 상태 사전을 사용하여 CLAP 모델을 구축
def build_model_from_openai_state_dict(
    state_dict: dict, model_cfg, enable_fusion: bool = False, fusion_type: str = "None"
):
    # 모델 구성에서 필요한 정보 추출
    embed_dim = model_cfg["embed_dim"]
    audio_cfg = model_cfg["audio_cfg"]
    text_cfg = model_cfg["text_cfg"]

    # 상태 사전에서 특정 값의 속성 추출
    state_dict["positional_embedding"].shape[0]  # 위치 임베딩의 크기 확인
    state_dict["token_embedding.weight"].shape[0]  # 토큰 임베딩의 크기 확인
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 변형기의 너비 확인
    transformer_width // 64  # 변형기의 너비를 64로 나눔
    transformer_layers = len(  # 변형기의 레이어 수 계산
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    # CLAPAudioCfp 및 CLAPTextCfg 인스턴스화
    audio_cfg = CLAPAudioCfp(**audio_cfg)
    text_cfg = CLAPTextCfg(**text_cfg)

    # CLAP 모델 구축
    model = CLAP(
        embed_dim,
        audio_cfg=audio_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI 모델은 QuickGELU로 훈련
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )

    # logit_scale_a와 logit_scale_t를 설정
    state_dict["logit_scale_a"] = state_dict["logit_scale"]
    state_dict["logit_scale_t"] = state_dict["logit_scale"]

    # 상태 사전에서 제거할 키 목록 설정
    pop_keys = list(state_dict.keys())[::]
    for key in pop_keys:
        # 시각 분기 저장된 가중치 제거
        if key.startswith("visual."):
            state_dict.pop(key, None)

    # 사용하지 않는 키 제거
    for key in ["logit_scale", "input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    # 가중치를 로드하고 평가 모드로 설정하여 모델 반환
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


#모델을 추적하고 TorchScript로 변환
def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    audio_length = model.audio_cfg.audio_length
    example_audio = torch.ones((batch_size, audio_length), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(#모델을 추적
        model,
        inputs=dict(
            forward=(example_audio, example_text),
            encode_text=(example_text,),
            encode_image=(example_audio,),
        ),
    )
    model.audio_cfg.audio_length = audio_length#  추적 시 오디오 길이를 설정하는 데 사용
    return model
