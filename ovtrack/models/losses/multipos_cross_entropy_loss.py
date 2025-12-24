import torch
import torch.nn as nn
from mmdet.models import LOSSES, weight_reduce_loss


def multi_pos_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    version="ori",
    pos_normalize=True,
):

    if version == "unbiased":

        valid_mask = label.sum(1) != 0
        pred = pred[valid_mask]
        label = label[valid_mask]
        weight = weight[valid_mask]
        logits_max, _ = torch.max(pred, dim=1, keepdim=True)
        logits = pred - logits_max.detach()

        if pos_normalize:
            pos_norm = torch.div(label, label.sum(1).reshape(-1, 1))
            exp_logits = (torch.exp(logits)) * pos_norm + (
                torch.exp(logits)
            ) * torch.logical_not(label)
        else:
            exp_logits = torch.exp(logits)
        exp_logits_input = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_input)

        mean_log_prob_pos = (label * log_prob).sum(1) / label.sum(1)
        loss = -mean_log_prob_pos

    elif version == "ori":
        # a more numerical stable implementation.
        pos_inds = label == 1
        neg_inds = label == 0
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float("inf")
        pred_neg[pos_inds] = pred_neg[pos_inds] + float("-inf")

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])

        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
        loss = torch.logsumexp(x, dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def multi_pos_cross_entropy_with_uncertainty(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    version="unbiased",
    pos_normalize=True,
    gamma_scale=2,
    alpha=0.25
):
    """
    :param pred:
    :param label:
    :param weight: the certainty matrix: 0 means no certainty, 1 means high certainty
    :param reduction:
    :param avg_factor:
    :param pos_normalize:
    :param gamma_scale:
    :param alpha:
    :return:
    """
    ## creating the sigmoid prediction matrix
    #  scale the inputs
    scaling_factor = pred.abs().max().detach()
    scaled_feats =  pred / scaling_factor  # scale to [-1, 1]
    # optional temprature factor, T > 0
    T = 1.0
    scaled_feats = scaled_feats / T
    p = torch.sigmoid(scaled_feats)
    # 3. Stabilize probs
    p = p.clamp(min=1e-8, max=1 - 1e-8)


    valid_mask = label.sum(1) != 0
    pred = pred[valid_mask]
    label = label[valid_mask]
    weight = weight[valid_mask]
    p = p[valid_mask]
    logits_max, _ = torch.max(pred, dim=1, keepdim=True)
    logits = pred - logits_max.detach()

    if pos_normalize:
        pos_norm = torch.div(label, label.sum(1).reshape(-1, 1))
        exp_logits = (torch.exp(logits)) * pos_norm + (
            torch.exp(logits)
        ) * torch.logical_not(label)
    else:
        exp_logits = torch.exp(logits)
    exp_logits_input = exp_logits.sum(1, keepdim=True)
    log_prob = logits - torch.log(exp_logits_input)

    # adding the focal attention here
    # getting the gamma
    if len(weight) > 0:
        eps = 1e-6
        gamma = (weight - weight.min()) / (weight.max() - weight.min() + eps) * gamma_scale
        gamma = gamma.view(-1, 1)
    else:
        gamma = 0

    alpha_tensor = alpha * label+ (1 - alpha) * (1 - label)
    log_prob = alpha_tensor * ((1 - p) ** gamma) * log_prob

    mean_log_prob_pos = (label * log_prob).sum(1) / label.sum(1)
    loss = -mean_log_prob_pos


    # apply weights and do the reduction
    # if weight_unnorm is not None:
    #     weight = weight_unnorm.float()
    loss = weight_reduce_loss(
        loss, weight=None, reduction=reduction, avg_factor=avg_factor
    )

    return loss


@LOSSES.register_module(force=True)
class MultiPosCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, version="v3"):
        super(MultiPosCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if True not in (label.sum(1) != 0):
            return 0.0 * torch.sum(cls_score)
        loss_cls = self.loss_weight * multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            version=self.version,
            **kwargs
        )
        return loss_cls
@LOSSES.register_module(force=True)
class MultiPosCrossEntropyLossUncertainty(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, version="v3", gamma_scale=2, alpha=0.25):
        super(MultiPosCrossEntropyLossUncertainty, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version
        self.gamma_scale = gamma_scale
        self.alpha = alpha

    def forward(
            self,
            cls_score,
            label,
            weight=None,
            avg_factor=None,
            reduction_override=None,
            **kwargs
    ):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * multi_pos_cross_entropy_with_uncertainty(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            version=self.version,
            gamma_scale=self.gamma_scale,
            alpha=self.alpha,
            **kwargs
        )
        return loss_cls
