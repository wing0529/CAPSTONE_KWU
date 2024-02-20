
# PANNs: 대규모 사전 학습된 오디오 신경망 모델
# https://github.com/qiuqiangkong/audioset_tagging_cnn 에서 참조된 코드를 기반
# CLAP에 맞게 일부 레이어를 재설계.

import os
# NUMBA_CACHE_DIR 환경 변수를 설정하여 Numba 캐시 디렉토리를 임시 디렉토리로 설정.
# Numba는 JIT 컴파일을 제공하는 라이브러리로, 최적화된 코드를 생성하기 위해 캐시를 사용
os.environ["NUMBA_CACHE_DIR"] = "/tmp/"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from .utils import do_mixup, interpolate
from .feature_fusion import iAFF, AFF, DAF

#선형 또는 합성곱 레이어를 초기화 
def init_layer(layer):
     # Xavier 초기화 방법을 사용하여 레이어의 가중치를 초기화
    nn.init.xavier_uniform_(layer.weight)
    # 레이어가 bias를 가지고 있는지 확인
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)# 0으로 설정

#배치 정규화 레이어를 초기화
def init_bn(bn):

    bn.bias.data.fill_(0.0)#bias를 0으로
    bn.weight.data.fill_(1.0)#가중치를 1로

# 컨볼루션 블록 클래스 (3x3 커널을 사용)
class ConvBlock(nn.Module):
    ''' Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
    '''
    # 클래스 초기화
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(#첫 번째 합성곱 레이어
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),#3x3 커널을 사용
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(#두 번째 합성곱 레이어
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)#첫 번째 배치 정규화 레이어
        self.bn2 = nn.BatchNorm2d(out_channels)#두 번째 배치 정규화 레이어

        self.init_weight()#레이어들의 가중치 초기화

    # 가중치 초기화
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    #Forward pass 연산을 수행
    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        # 합성곱 레이어를 통과시키고 ReLU 활성화 함수를 적용
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        # 풀링 연산을 적용
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x

# 5x5 커널을 사용하는 합성곱 블록 클래스
class ConvBlock5x5(nn.Module):
    #클래스를 초기화
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),#5x5 커널을 사용
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
    # 가중치 초기화
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x

#어텐션 블록 클래스
class AttBlock(nn.Module):
    #
    def __init__(self, n_in, n_out, activation="linear", temperature=1.0):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
    #가중치를 초기화
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        # 입력에 대해 어텐션 가중치를 계산
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        # 입력을 가중합하고, 그 결과에 활성화 함수를 적용하여 출력을 생성
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla
    
    
    #활성화 함수를 적용하여 입력을 변환
    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

#음향 데이터의 스펙트로그램을 입력으로
class Cnn14(nn.Module):
    # 모델의 초기화를 수행
    def __init__(
        self,
        sample_rate,            # 샘플링 속도
        window_size,            # 윈도우 크기
        hop_size,               # Hop 크기
        mel_bins,               # Mel bins 수
        fmin,                   # 최소 주파수
        fmax,                   # 최대 주파수
        classes_num,            # 클래스 수
        enable_fusion=False,    # 퓨전 활성화 여부
        fusion_type="None",     # 퓨전 타입
    ):
        #부모 클래스인 nn.Module의 초기화 메서드를 호출하여 초기화
        super(Cnn14, self).__init__()
        #스펙트로그램 및 로그멜 특성 추출을 위한 파라미터를 정의
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        #퓨전을 활성화하고 퓨전 타입을 설정
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        # 스펙트로그램 추출기를 초기화
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # 로그멜 특성 추출기를 초기화
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # 스펙트로그램 데이터 증강기를 초기화
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        #Batch Normalization을 위한 bn0을 초기화
        self.bn0 = nn.BatchNorm2d(64)

        #합성곱 블록을 초기화
        if (self.enable_fusion) and (self.fusion_type == "channel_map"):
            self.conv_block1 = ConvBlock(in_channels=4, out_channels=64)
        else:
            self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        ## 1D 퓨전 모델을 위한 초기화
        if (self.enable_fusion) and (
            self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]
        ):
            self.mel_conv1d = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1d(64),  # No Relu
            )
            if self.fusion_type == "daf_1d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_1d":
                self.fusion_model = AFF(channels=64, type="1D")
            elif self.fusion_type == "iaff_1d":
                self.fusion_model = iAFF(channels=64, type="1D")
        # 2D 퓨전 모델을 위한 초기화
        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            self.mel_conv2d = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(6, 2), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            if self.fusion_type == "daf_2d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_2d":
                self.fusion_model = AFF(channels=64, type="2D")
            elif self.fusion_type == "iaff_2d":
                self.fusion_model = iAFF(channels=64, type="2D")
        self.init_weight()
    #가중치를 초기화
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
    #입력에 대해 순전파를 수행
    def forward(self, input, mixup_lambda=None, device=None):
        """
        Input: (batch_size, data_length)"""

        if self.enable_fusion and input["longer"].sum() == 0:
            #퓨전이 활성화되어 있고 입력 데이터 중 어떤 오디오도 10초보다 길지 않은 경우, 무작위로 선택된 오디오 하나를 더 길게 만듦
            input["longer"][torch.randint(0, input["longer"].shape[0], (1,))] = True

        # 퓨전이 비활성화된 경우
        if not self.enable_fusion:#waveform으로부터 스펙트로그램을 추출
            x = self.spectrogram_extractor(
                input["waveform"].to(device=device, non_blocking=True)
            )  # (batch_size, 1, time_steps, freq_bins)
            #로그 멜 스펙트로그램 특성을 추출
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            #Batch Normalization
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
        # 퓨전이 활성화된 경우
        else:
            #퓨전된 멜 스펙트로그램 데이터를 사용
            longer_list = input["longer"].to(device=device, non_blocking=True)
            x = input["mel_fusion"].to(device=device, non_blocking=True)
            longer_list_idx = torch.where(longer_list)[0]
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            #1차원 퓨전 모델
            if self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]:
                # 새로운 텐서를 생성
                new_x = x[:, 0:1, :, :].clone().contiguous()

                # 오디오가 10초 이상인 경우에만 로컬 처리를 수행
                if len(longer_list_idx) > 0:
                
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(
                        fusion_x_local, (0, 2, 1)
                    ).contiguous()
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(
                        FB, FC, FF, fusion_x_local.size(-1)
                    )
                    fusion_x_local = (
                        torch.permute(fusion_x_local, (0, 2, 1, 3))
                        .contiguous()
                        .flatten(2)
                    )
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat(
                            [
                                fusion_x_local,
                                torch.zeros(
                                    (FB, FF, FT - fusion_x_local.size(-1)),
                                    device=device,
                                ),
                            ],
                            dim=-1,
                        )
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.fusion_model(
                        new_x[longer_list_idx], fusion_x_local
                    )
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                #10초 이상이 아니라면
                else:
                    #새로운 텐서로 설정한 것을 그대로 사용
                    x = new_x
            # 2차원 퓨전 모델
            elif self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d", "channel_map"]:
                x = x  # no change
        
        #데이터에 SpecAugmentation을 적용
        if self.training:
            x = self.spec_augmenter(x)
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            global_x = x[:, 0:1, :, :]

            # global processing
            B, C, H, W = global_x.shape
            global_x = self.conv_block1(global_x, pool_size=(2, 2), pool_type="avg")
            if len(longer_list_idx) > 0:
                local_x = x[longer_list_idx, 1:, :, :].contiguous()
                TH = global_x.size(-2)
                # local processing
                B, C, H, W = local_x.shape
                local_x = local_x.view(B * C, 1, H, W)
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.view(
                    B, C, local_x.size(1), local_x.size(2), local_x.size(3)
                )
                local_x = local_x.permute((0, 2, 1, 3, 4)).contiguous().flatten(2, 3)
                TB, TC, _, TW = local_x.size()
                if local_x.size(-2) < TH:
                    local_x = torch.cat(
                        [
                            local_x,
                            torch.zeros(
                                (TB, TC, TH - local_x.size(-2), TW),
                                device=global_x.device,
                            ),
                        ],
                        dim=-2,
                    )
                else:
                    local_x = local_x[:, :, :TH, :]

                global_x[longer_list_idx] = self.fusion_model(
                    global_x[longer_list_idx], local_x
                )
            x = global_x
        else:
            x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        latent_x = latent_x.transpose(1, 2)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 32)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": embedding,
            "fine_grained_embedding": latent_output,
        }
        return output_dict


class Cnn6(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        enable_fusion=False,
        fusion_type="None",
    ):
        super(Cnn6, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None, device=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        latent_x = latent_x.transpose(1, 2)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 16)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": embedding,
            "fine_grained_embedding": latent_output,
        }

        return output_dict


class Cnn10(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        enable_fusion=False,
        fusion_type="None",
    ):
        super(Cnn10, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None, device=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        latent_x = latent_x.transpose(1, 2)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 32)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            "clipwise_output": clipwise_output,
            "embedding": embedding,
            "fine_grained_embedding": latent_output,
        }

        return output_dict


def create_pann_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    try:
        ModelProto = eval(audio_cfg.model_name)
        model = ModelProto(
            sample_rate=audio_cfg.sample_rate,
            window_size=audio_cfg.window_size,
            hop_size=audio_cfg.hop_size,
            mel_bins=audio_cfg.mel_bins,
            fmin=audio_cfg.fmin,
            fmax=audio_cfg.fmax,
            classes_num=audio_cfg.class_num,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type,
        )
        return model
    except:
        raise RuntimeError(
            f"Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough."
        )
