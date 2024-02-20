"""
Feature Fusion for Varible-Length Data Processing
AFF/iAFF is referred and modified from https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
According to the paper: Yimian Dai et al, Attentional Feature Fusion, IEEE Winter Conference on Applications of Computer Vision, WACV 2021
"""
#가변 길이 데이터 처리를 위한 특징 퓨전
import torch
import torch.nn as nn


class DAF(nn.Module):
    """
    DAF(Direct Add Fusion):직접 더하기
    """

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    """
    iAFF (Improved Attentional Feature Fusion): 다중 특징 퓨전 
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":# 1차원 경우
            # 로컬 어텐션
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # 글로벌 어텐션
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            #두 번째 로컬 어텐션
            self.local_att2 = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            # 두 번째 글로벌 어텐션
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == "2D":# 2차원 경우
            # 로컬 어텐션
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 글로벌 어텐션
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 두 번째 로컬 어텐션
            self.local_att2 = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            # 두 번째 글로벌 어텐션
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f"the type is not supported"

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual# 입력과 잔여 값을 더함
        if xa.size(0) == 1:# 배치 크기가 1
            xa = torch.cat([xa, xa], dim=0)# 배치 크기를 2로
            flag = True # 플래그를 설정하여 나중에 결과를 다시 조정할 필요가 있음을 나타냄
        xl = self.local_att(xa)# 로컬 어텐션을 수행
        xg = self.global_att(xa)# 글로벌 어텐션을 수행
        xlg = xl + xg# 로컬 및 글로벌 어텐션의 결과를 더함
        wei = self.sigmoid(xlg) # 시그모이드 함수를 적용하여 가중치를 얻
        xi = x * wei + residual * (1 - wei)# 입력과 잔여 값을 가중합하여 중간 결과를 얻

        xl2 = self.local_att2(xi)# 두 번째 로컬 어텐션을 수행
        xg2 = self.global_att(xi)# 두 번째 글로벌 어텐션을 수행
        xlg2 = xl2 + xg2 # 두 번째 로컬 및 글로벌 어텐션의 결과를 더함
        wei2 = self.sigmoid(xlg2)# 두 번째 어텐션의 가중치를 얻
        xo = x * wei2 + residual * (1 - wei2)# 입력과 잔여 값을 두 번째 어텐션의 가중합으로 조합하여 최종 결과를 얻
        if flag:
            xo = xo[0].unsqueeze(0)# 배치 크기를 다시 원래대로
        return xo


class AFF(nn.Module):
    """
    AFF (Attentional Feature Fusion):다중 특징 퓨전
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":
            self.local_att = nn.Sequential(# 로컬 어텐션
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(# 글로벌 어텐션
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == "2D":
            self.local_att = nn.Sequential(# 로컬 어텐션
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(# 글로벌 어텐션
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f"the type is not supported."

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual# 입력과 잔여 값을 더함
        if xa.size(0) == 1:# 배치 크기가 1
            xa = torch.cat([xa, xa], dim=0)# 배치 크기를 2로
            flag = True# 플래그를 설정
        xl = self.local_att(xa)# 로컬 어텐션을 수행
        xg = self.global_att(xa)# 글로벌 어텐션을 수행
        xlg = xl + xg#결과를 더함
        wei = self.sigmoid(xlg) #시그모이드 함수를 적용하여 가중치를 얻
        xo = 2 * x * wei + 2 * residual * (1 - wei)# 입력과 잔여 값을 가중합
        if flag:
            xo = xo[0].unsqueeze(0)# 배치 크기를 다시 원래대로
        return xo
