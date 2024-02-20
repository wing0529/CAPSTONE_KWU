import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
#분산 학습을 위한 특성 수집
def gather_features(
    audio_features,
    text_features,
    audio_features_mlp=None,
    text_features_mlp=None,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
    mlp_loss=False,
):
    # 분산 학습을 위한 특성 수집 메서드
    # Horovod를 사용할 경우
    if use_horovod:
        assert hvd is not None, "Please install horovod"  # Horovod가 설치되어 있는지 확인
        if gather_with_grad:
        # 그래디언트와 함께 특성을 수집하는 경우
            all_audio_features = hvd.allgather(audio_features)  # 모든 오디오 특성을 수집
            all_text_features = hvd.allgather(text_features)  # 모든 텍스트 특성을 수집
            if mlp_loss:
            # MLP 손실을 사용하는 경우
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)  # 모든 오디오 MLP 특성을 수집
                all_text_features_mlp = hvd.allgather(text_features_mlp)  # 모든 텍스트 MLP 특성을 수집
        else:
        # 그래디언트 없이 특성을 수집하는 경우
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)  # 모든 오디오 특성을 수집
                all_text_features = hvd.allgather(text_features)  # 모든 텍스트 특성을 수집
                if mlp_loss:
                # MLP 손실을 사용하는 경우
                    all_audio_features_mlp = hvd.allgather(audio_features_mlp)  # 모든 오디오 MLP 특성을 수집
                    all_text_features_mlp = hvd.allgather(text_features_mlp)  # 모든 텍스트 MLP 특성을 수집
            if not local_loss:
            # 로컬 손실이 아닌 경우
                gathered_audio_features = list(
                    all_audio_features.chunk(world_size, dim=0)
                )  # 오디오 특성을 월드 사이즈만큼 쪼개어 리스트로 모음
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )  # 텍스트 특성을 월드 사이즈만큼 쪼개어 리스트로 모음
                gathered_audio_features[rank] = audio_features  # 현재 랭크에 대한 오디오 특성을 설정
                gathered_text_features[rank] = text_features  # 현재 랭크에 대한 텍스트 특성을 설정
                all_audio_features = torch.cat(gathered_audio_features, dim=0)  # 수집된 오디오 특성을 합침
                all_text_features = torch.cat(gathered_text_features, dim=0)  # 수집된 텍스트 특성을 합침
            if mlp_loss:
                # MLP 손실을 사용하는 경우
                gathered_audio_features_mlp = list(
                    all_audio_features_mlp.chunk(world_size, dim=0)
                )  # 오디오 MLP 특성을 월드 사이즈만큼 쪼개어 리스트로 모음
                gathered_text_features_mlp = list(
                    all_text_features_mlp.chunk(world_size, dim=0)
                )  # 텍스트 MLP 특성을 월드 사이즈만큼 쪼개어 리스트로 모음
                gathered_audio_features_mlp[rank] = audio_features_mlp  # 현재 랭크에 대한 오디오 MLP 특성을 설정
                gathered_text_features_mlp[rank] = text_features_mlp  # 현재 랭크에 대한 텍스트 MLP 특성을 설정
                all_audio_features_mlp = torch.cat(
                    gathered_audio_features_mlp, dim=0
                )  # 수집된 오디오 MLP 특성을 합침
                all_text_features_mlp = torch.cat(
                    gathered_text_features_mlp, dim=0
                )  # 수집된 텍스트 MLP 특성을 합침
    # Horovod를 사용하지 않을 경우
    else:
    
        if gather_with_grad:
        # 그래디언트와 함께 특성을 수집하는 경우
            all_audio_features = torch.cat(
            torch.distributed.nn.all_gather(audio_features), dim=0
            )  # 모든 오디오 특성을 수집
            all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
            )  # 모든 텍스트 특성을 수집
            if mlp_loss:
            # MLP 손실을 사용하는 경우
                all_audio_features_mlp = torch.cat(
                torch.distributed.nn.all_gather(audio_features_mlp), dim=0
                )  # 모든 오디오 MLP 특성을 수집
                all_text_features_mlp = torch.cat(
                torch.distributed.nn.all_gather(text_features_mlp), dim=0
                )  # 모든 텍스트 MLP 특성을 수집
        else:
        # 그래디언트 없이 특성을 수집하는 경우
            gathered_audio_features = [
            torch.zeros_like(audio_features) for _ in range(world_size)
            ]  # 오디오 특성을 월드 사이즈만큼 빈 텐서로 초기화한 리스트 생성
            gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
            ]  # 텍스트 특성을 월드 사이즈만큼 빈 텐서로 초기화한 리스트 생성
            dist.all_gather(gathered_audio_features, audio_features)  # 오디오 특성을 수집하여 리스트에 저장
            dist.all_gather(gathered_text_features, text_features)  # 텍스트 특성을 수집하여 리스트에 저장
        if mlp_loss:
            # MLP 손실을 사용하는 경우
            gathered_audio_features_mlp = [
                torch.zeros_like(audio_features_mlp) for _ in range(world_size)
            ]  # 오디오 MLP 특성을 월드 사이즈만큼 빈 텐서로 초기화한 리스트 생성
            gathered_text_features_mlp = [
                torch.zeros_like(text_features_mlp) for _ in range(world_size)
            ]  # 텍스트 MLP 특성을 월드 사이즈만큼 빈 텐서로 초기화한 리스트 생성
            dist.all_gather(
                gathered_audio_features_mlp, audio_features_mlp
            )  # 오디오 MLP 특성을 수집하여 리스트에 저장
            dist.all_gather(
                gathered_text_features_mlp, text_features_mlp
            )  # 텍스트 MLP 특성을 수집하여 리스트에 저장
        if not local_loss:
            # 로컬 손실이 아닌 경우
            gathered_audio_features[rank] = audio_features  # 현재 랭크에 대한 오디오 특성을 설정
            gathered_text_features[rank] = text_features  # 현재 랭크에 대한 텍스트 특성을 설정
            if mlp_loss:
                # MLP 손실을 사용하는 경우
                gathered_audio_features_mlp[rank] = audio_features_mlp  # 현재 랭크에 대한 오디오 MLP 특성을 설정
                gathered_text_features_mlp[rank] = text_features_mlp  # 현재 랭크에 대한 텍스트 MLP 특성을 설정
            all_audio_features = torch.cat(
                gathered_audio_features, dim=0
            )  # 수집된 오디오 특성을 합침
            all_text_features = torch.cat(
                gathered_text_features, dim=0
            )  # 수집된 텍스트 특성을 합침
            if mlp_loss:
                # MLP 손실을 사용하는 경우
                all_audio_features_mlp = torch.cat(
                    gathered_audio_features_mlp, dim=0
                )  # 수집된 오디오 MLP 특성을 합침
                all_text_features_mlp = torch.cat(
                    gathered_text_features_mlp, dim=0
                )  # 수집된 텍스트 MLP 특성을 합침
        # MLP 손실을 사용하는 경우
        if mlp_loss:
            return (
        all_audio_features,
        all_text_features,
        all_audio_features_mlp,
        all_text_features_mlp,
            )
        # MLP 손실을 사용하지 않는 경우
        else:
            return all_audio_features, all_text_features

#손실함수 종류에 따른 로짓, 크로스 엔트로피 계산 후 전체 손실 반환
class ClipLoss(nn.Module):
    #ClipLoss 클래스 초기화
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa != 0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    #오디오 특성, 텍스트 특성, 로짓 스케일 등을 입력으로 받아 손실 계산
    def forward(
        self,
        audio_features,
        text_features,
        logit_scale_a,
        logit_scale_t=None,
        audio_features_mlp=None,
        text_features_mlp=None,
    ):
        device = audio_features.device
        # MLP 손실을 사용 : 분산 학습을 고려하여 특성을 수집 -> 오디오와 텍스트 특성을 MLP를 사용하여 특성을 조합하고 로짓, Cross Entropy 손실 계산 
        if self.mlp_loss:
            if self.world_size > 1:
                (
                    all_audio_features,
                    all_text_features,
                    all_audio_features_mlp,
                    all_text_features_mlp,
                ) = gather_features(
                    audio_features=audio_features,
                    text_features=text_features,
                    audio_features_mlp=audio_features_mlp,
                    text_features_mlp=text_features_mlp,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )
                #로짓을 계산
                if self.local_loss:
                    a_logits_per_audio = (
                        logit_scale_a * audio_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = (
                        logit_scale_a * text_features_mlp @ all_audio_features.T
                    )
                    t_logits_per_audio = (
                        logit_scale_t * audio_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = (
                        logit_scale_t * text_features @ all_audio_features_mlp.T
                    )
                else:
                    a_logits_per_audio = (
                        logit_scale_a * all_audio_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = a_logits_per_audio.T
                    t_logits_per_audio = (
                        logit_scale_t * all_audio_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = t_logits_per_audio.T
            #로짓을 계산
            else:
                a_logits_per_audio = (
                    logit_scale_a * audio_features @ text_features_mlp.T
                )
                a_logits_per_text = logit_scale_a * text_features_mlp @ audio_features.T
                t_logits_per_audio = (
                    logit_scale_t * audio_features_mlp @ text_features.T
                )
                t_logits_per_text = logit_scale_t * text_features @ audio_features_mlp.T

            #로짓을 계산
            num_logits = a_logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            
            # Cross Entropy 손실을 계산
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels)
                    + F.cross_entropy(a_logits_per_text, labels)
                    + F.cross_entropy(t_logits_per_audio, labels)
                    + F.cross_entropy(t_logits_per_text, labels)
                ) / 4
            else:
                audio_weight = (audio_features @ audio_features.T).detach()
                audio_weight = (
                    torch.exp(
                        torch.sum(audio_weight, axis=1)
                        / (self.weight_loss_kappa * len(audio_weight))
                    )
                ).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels, weight=audio_weight)
                    + F.cross_entropy(a_logits_per_text, labels, weight=audio_weight)
                    + F.cross_entropy(t_logits_per_audio, labels, weight=text_weight)
                    + F.cross_entropy(t_logits_per_text, labels, weight=text_weight)
                ) / 4
                
        # MLP 손실 사용 x :  단순히 오디오와 텍스트 특성의 조합에 따라 로짓을 계산하고, Cross Entropy 손실을 계산
        else:
            if self.world_size > 1:
                all_audio_features, all_text_features = gather_features(
                    audio_features=audio_features,
                    text_features=text_features,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )

                if self.local_loss:
                    logits_per_audio = (
                        logit_scale_a * audio_features @ all_text_features.T
                    )
                    logits_per_text = (
                        logit_scale_a * text_features @ all_audio_features.T
                    )
                else:
                    logits_per_audio = (
                        logit_scale_a * all_audio_features @ all_text_features.T
                    )
                    logits_per_text = logits_per_audio.T
            else:
                logits_per_audio = logit_scale_a * audio_features @ text_features.T
                logits_per_text = logit_scale_a * text_features @ audio_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels)
                    + F.cross_entropy(logits_per_text, labels)
                ) / 2
            else:
                audio_weight = (all_audio_features @ all_audio_features.T).detach()
                audio_weight = (
                    torch.exp(
                        torch.sum(audio_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_audio_features))
                    )
                ).detach()
                text_weight = (all_text_features @ all_text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels, weight=text_weight)
                    + F.cross_entropy(logits_per_text, labels, weight=audio_weight)
                ) / 2
        #전체 손실을 반환
        return total_loss

#분산 환경에서 예측값과 타겟값을 수집하고, 이를 기반으로 평균 정밀도(Mean Average Precision, mAP)를 계산
def lp_gather_features(pred, target, world_size=1, use_horovod=False):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        with torch.no_grad():#예측값과 타겟값을 수집
            all_preds = hvd.allgather(pred)
            all_targets = hvd.allgath(target)
    else:#Horovod를 사용 X 
        gathered_preds = [torch.zeros_like(pred) for _ in range(world_size)]#예측값과 타겟값을 수집
        gathered_targets = [torch.zeros_like(target) for _ in range(world_size)]

        dist.all_gather(gathered_preds, pred)
        dist.all_gather(gathered_targets, target)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)

    return all_preds, all_targets

#평균 정밀도(mAP)를 계산
def get_map(pred, target):#예측 값: 시그모이드 함수 -> 확률 값 -> 넘파이 배열, 타겟 값: 넘파이 배열 
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))#평균 정밀도(mAP)를 계산

#정확도(accuracy)를 계산
def get_acc(pred, target):
    pred = torch.argmax(pred, 1).numpy()
    target = torch.argmax(target, 1).numpy()
    return accuracy_score(target, pred)

#다중 클래스 AUC 평균(mean AUC, MAUC)을 계산
def get_mauc(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(roc_auc_score(target, pred, average=None))

#평가할 지표(metric_names)를 지정하여 객체를 생성
class LPMetrics(object):
    def __init__(self, metric_names=["map", "acc", "mauc"]):
        self.metrics = []
        for name in metric_names:
            self.metrics.append(self.get_metric(name))
        self.metric_names = metric_names

    def get_metric(self, name):
        if name == "map":
            return get_map
        elif name == "acc":
            return get_acc
        elif name == "mauc":
            return get_mauc
        else:
            raise ValueError(f"the metric should be at least one of [map, acc, mauc]")

    def evaluate_mertics(self, pred, target):
        metric_dict = {}
        for i in range(len(self.metric_names)):
            metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
        return metric_dict

#다중 클래스 분류를 위한 Cross Entropy Loss를 계산
def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)

#loss_name에 따라 손실 함수(loss_func)를 선택 -> 손실을 계산하고 반환
class LPLoss(nn.Module):
    #loss_name에 따라 손실 함수(loss_func)를 선택하여 객체를 생성
    def __init__(self, loss_name):
        super().__init__()
        if loss_name == "bce":
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == "ce":
            self.loss_func = calc_celoss
        elif loss_name == "mse":
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f"the loss func should be at least one of [bce, ce, mse]")
    #선택된 손실 함수를 사용하여 손실을 계산하고 반환
    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss
