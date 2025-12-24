import cv2
import mmcv
import numpy as np
from functools import partial
import os
import random
import seaborn as sns
import torch
from collections import defaultdict
from mmcv.image import imread, imwrite
from mmcv.visualization import color_val, imshow
from mmdet.core import bbox_overlaps
from addict import Dict

from ovtrack.core import cal_similarity
from .sort_tracker import SortTracker
from ..builder import TRACKERS, build_motion


@TRACKERS.register_module()
class OVTracker(object):
    def __init__(
        self,
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        distractor_nms_thr=0.3,
        distractor_score_thr=0.5,
        match_metric="bisoftmax",
        match_with_cosine=True,
        contrastive_thr=0.5,
    ):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ["bisoftmax", "cosine"]
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine
        self.contrastive_thr = contrastive_thr

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v["ids"])
        return list(set(valid_ids))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, cls_embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            cls_embeds[tracklet_inds],
            labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]["bboxes"].append(bbox)
                self.tracklets[id]["labels"].append(label)
                self.tracklets[id]["embeds"] = (
                    1 - self.momentum_embed
                ) * self.tracklets[id]["embeds"] + self.momentum_embed * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["frame_ids"].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    cls_embeds=cls_embed,
                    frame_ids=[frame_id],
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["frame_ids"][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        memo_cls_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v["bboxes"][-1][None, :])
            memo_labels.append(v["labels"][-1].view(1, 1))
            memo_embeds.append(v["embeds"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
        )

    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_new_objs, dtype=torch.long
        )
        self.num_tracklets += num_new_objs
        return ids

    def match(
        self,
        bboxes,
        labels,
        embeds,
        cls_embeds,
        frame_id,
        temperature=-1,
        method="ovtrack-teta",
        **kwargs
    ):
        """

        Args:
            bboxes:
            labels:
            track_feats: if use transformer method, the track_feats will be the encoder feats
            cls_feats: if use transformer method, the cls_feats will be the decoder feats
            frame_id:
            temperature:
            method: 'TETer'| 'oracle' | 'appearance' | 'contrastive'
            **kwargs:

        Returns:

        """

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        if method == "ovtrack-teta":
            # match if buffer is not empty
            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter"
            )
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo

                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )

                    exps = torch.exp(sims)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    cls_sims = cal_similarity(
                        cls_embeds,
                        memo_cls_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = cls_sims > self.contrastive_thr
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0

            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        elif method == "ovtrack-tao":

            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="intra"
            )
            # match if buffer is not empty
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo
                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    exps = torch.exp(sims) * cat_same.to(sims.device)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cos_scores *= cat_same.to(cos_scores.device)
                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0
                        m = self.momentum_obj_score
                        bboxes[i, -1] = (
                            m * bboxes[i, -1] + (1 - m) * memo_bboxes[memo_ind, -1]
                        )
            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        else:
            raise NotImplementedError

        return bboxes, labels, ids

    def remove_distractor(
        self,
        bboxes,
        labels,
        track_feats,
        cls_feats,
        object_score_thr=0.5,
        distractor_nms_thr=0.3,
        softmax_feats=None,
        nms="inter",
    ):

        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        if bboxes.shape[1] == 6:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-2], bboxes[:, :-2])
            else:
                raise NotImplementedError
        else:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
            elif nms == "intra":
                cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
                ious *= cat_same.to(ious.device)
            else:
                raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]
        cls_embeds = cls_feats[valid_inds]
        if softmax_feats is not None:
            softmax_feats = softmax_feats[valid_inds]

        return bboxes, labels, embeds, cls_embeds, softmax_feats



@TRACKERS.register_module()
class OVTrackerUncertainty(object):
    def __init__(
            self,
            init_score_thr=0.0001,
            obj_score_thr=0.0001,
            match_score_thr=0.5,
            memo_frames=10,
            momentum_embed=0.8,
            momentum_obj_score=0.5,
            distractor_nms_thr=0.3,
            distractor_score_thr=0.5,
            match_metric="bisoftmax",
            match_with_cosine=True,
            contrastive_thr=0.5,
            confused_features=False,
            vis=True,
    ):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ["bisoftmax", "cosine"]
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine
        self.contrastive_thr = contrastive_thr

        self.confused_features = confused_features
        self.vis = vis

        self.reset()

        if self.vis and os.path.exists('/data1/clark/dataset/openDomain/TAO/tao/annotations/tao_validation_ours_v1_filename2ann.pth'):
            self.filename2ann = torch.load('/data1/clark/dataset/openDomain/TAO/tao/annotations/tao_validation_ours_v1_filename2ann.pth')

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v["ids"])
        return list(set(valid_ids))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, cls_embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
                ids[tracklet_inds],
                bboxes[tracklet_inds],
                embeds[tracklet_inds],
                cls_embeds[tracklet_inds],
                labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]["bboxes"].append(bbox)
                self.tracklets[id]["labels"].append(label)
                self.tracklets[id]["embeds"] = (
                                                       1 - self.momentum_embed
                                               ) * self.tracklets[id]["embeds"] + self.momentum_embed * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["frame_ids"].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    cls_embeds=cls_embed,
                    frame_ids=[frame_id],
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["frame_ids"][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        memo_cls_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v["bboxes"][-1][None, :])
            memo_labels.append(v["labels"][-1].view(1, 1))
            memo_embeds.append(v["embeds"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
        )

    def set_fusion_head(self, fusion_head, loss_cyc):
        self.fusion_head = fusion_head
        self.loss_cyc = loss_cyc
    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_new_objs, dtype=torch.long
        )
        self.num_tracklets += num_new_objs
        return ids

    def match(
            self,
            bboxes,
            labels,
            embeds,
            cls_embeds,
            frame_id,
            temperature=-1,
            method="ovtrack-teta",
            **kwargs
    ):
        """

        Args:
            bboxes:
            labels:
            track_feats: if use transformer method, the track_feats will be the encoder feats
            cls_feats: if use transformer method, the cls_feats will be the decoder feats
            frame_id:
            temperature:
            method: 'TETer'| 'oracle' | 'appearance' | 'contrastive'
            **kwargs:

        Returns:

        """

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        if method == "ovtrack-teta":
            # match if buffer is not empty
            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter"
            )
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo

                if self.confused_features:
                    # fused features here
                    # appearance, bbox, cls
                    uncertainty_list = [[], [], []]
                    # getting the consistency of each object
                    key_feats_list = [embeds, bboxes[:,:-1], cls_embeds]
                    ref_feats_list = [memo_embeds, memo_bboxes[:,:-1], memo_cls_embeds]
                    for i in range(len(key_feats_list)):
                        key_feat = key_feats_list[i]
                        ref_feat = ref_feats_list[i]
                        consistency, consistency_diag = self.loss_cyc.get_pair_consistency(
                            [key_feat, ref_feat])
                        uncertainty_list[i].append(consistency_diag)
                    key_assoc_conf, key_bbox_conf, key_cls_conf = [
                        torch.cat(tensor_list).view(-1, 1).detach() for tensor_list in uncertainty_list]

                    matching_indices = None
                    if self.vis and hasattr(self, 'filename2ann'):
                        # only used to visualizing the restuls
                        ann = self.filename2ann[kwargs['filename']]
                        gt_bbox_list = [an['bbox'] for an in ann]
                        det_bbox_list = bboxes[:,:-1].cpu().tolist()

                        def compute_iou(box1, box2):
                            """
                            计算两个边界框之间的IoU。
                            box1 和 box2 均为格式 (x1, y1, x2, y2)。
                            """
                            # 计算交集
                            x_left = max(box1[0], box2[0])
                            y_top = max(box1[1], box2[1])
                            x_right = min(box1[2], box2[2])
                            y_bottom = min(box1[3], box2[3])

                            if x_right < x_left or y_bottom < y_top:
                                return 0.0  # 没有交集

                            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                            # 计算并集
                            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = box1_area + box2_area - intersection_area

                            # 计算IoU
                            iou = intersection_area / union_area
                            return iou

                        def get_matching_indices(gt_bbox_list, det_bbox_list, iou_threshold=0.5):
                            """
                            找出det_bbox_list中与gt_bbox_list的IoU大于iou_threshold的框子的索引位置。
                            返回：
                            - matching_det_indices: 检测框的匹配索引列表
                            - matching_gt_indices: 对应的真值框索引列表
                            """
                            matching_det_indices = []
                            matching_gt_indices = []

                            for det_index, det_box in enumerate(det_bbox_list):
                                max_iou = 0
                                max_gt_index = -1

                                # 找到与当前检测框IoU最大的真值框
                                for gt_index, gt_box in enumerate(gt_bbox_list):
                                    iou = compute_iou(det_box, gt_box)
                                    if iou > max_iou:
                                        max_iou = iou
                                        max_gt_index = gt_index

                                        # 如果最大IoU超过阈值，则记录这对匹配
                                if max_iou > iou_threshold:
                                    matching_det_indices.append(det_index)
                                    matching_gt_indices.append(max_gt_index)

                            return matching_det_indices, matching_gt_indices

                        matching_indices, matching_gt_indices = get_matching_indices(gt_bbox_list, det_bbox_list, iou_threshold=0.5)


                        # inter_app = [(det_bbox_list[index], key_assoc_conf[index].cpu().item()) for index in matching_indices if key_assoc_conf[index].item() != 1]
                        # inter_loc = [(det_bbox_list[index], key_bbox_conf[index].cpu().item()) for index in matching_indices if key_bbox_conf[index].item() != 1]
                        # inter_sem = [(det_bbox_list[index], key_cls_conf[index].cpu().item()) for index in matching_indices if key_cls_conf[index].item() != 1]

                        confused_key_feats, intra_loc_list, intra_sem_list = self.fusion_head(embeds, bboxes[:,:-1], cls_embeds, key_assoc_conf, key_bbox_conf, key_cls_conf, matching_indices)

                        # intra_loc = [(det_bbox_list[index], intra_loc_list[i]) for i, index in enumerate(matching_indices) if intra_loc_list[i] != 0]
                        # intra_sem = [(det_bbox_list[index], intra_sem_list[i]) for i, index in enumerate(matching_indices) if intra_sem_list[i] != 0]
                        total_dict = [{
                            'filename': kwargs['filename'],
                            'det_bbox': det_bbox_list[index],
                            'intra_loc': intra_loc_list[i],
                            'intra_sem': intra_sem_list[i],
                            'inter_app': key_assoc_conf[index].cpu().item(),
                            'inter_loc': key_bbox_conf[index].cpu().item(),
                            'inter_sem': key_cls_conf[index].cpu().item()
                        } for i, index in enumerate(matching_indices)]

                        for i, index in enumerate(matching_gt_indices):
                            total_dict[i]['gt_bbox'] = ann[index]['bbox']
                            total_dict[i]['category'] = ann[index]['category']
                            total_dict[i]['category_id'] = ann[index]['category_id']
                            total_dict[i]['track_id'] = ann[index]['track_id']

                        # torch.save(total_dict, os.path.join('/data1/clark/dataset/openDomain/TAO/tao/vis_data/tao_validation_conf_vis_with_track_id', kwargs['filename'].replace('/', '_').replace('.jpg', '.pth')))






                    else:
                        confused_key_feats = self.fusion_head(embeds, bboxes[:,:-1], cls_embeds, key_assoc_conf, key_bbox_conf, key_cls_conf)


                    embeds = confused_key_feats


                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )

                    exps = torch.exp(sims)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    cls_sims = cal_similarity(
                        cls_embeds,
                        memo_cls_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = cls_sims > self.contrastive_thr
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0

            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        elif method == "ovtrack-tao":

            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="intra"
            )
            # match if buffer is not empty
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo
                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    exps = torch.exp(sims) * cat_same.to(sims.device)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cos_scores *= cat_same.to(cos_scores.device)
                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0
                        m = self.momentum_obj_score
                        bboxes[i, -1] = (
                                m * bboxes[i, -1] + (1 - m) * memo_bboxes[memo_ind, -1]
                        )
            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        else:
            raise NotImplementedError

        if self.vis and hasattr(self, 'filename2ann') and 'matching_indices' in locals():
            for i, index in enumerate(matching_indices):
                pred_category_id = labels[index].cpu().item()
                pred_track_id = ids[index].cpu().item()
                total_dict[i]['pred_category_id'] = pred_category_id
                total_dict[i]['pred_track_id'] = pred_track_id
            torch.save(total_dict, os.path.join(
                '/data1/clark/dataset/openDomain/TAO/tao/vis_data/tao_validation_conf_vis_with_track_id',
                kwargs['filename'].replace('/', '_').replace('.jpg', '.pth')))

        return bboxes, labels, ids

    def remove_distractor(
            self,
            bboxes,
            labels,
            track_feats,
            cls_feats,
            object_score_thr=0.5,
            distractor_nms_thr=0.3,
            softmax_feats=None,
            nms="inter",
    ):

        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        if bboxes.shape[1] == 6:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-2], bboxes[:, :-2])
            else:
                raise NotImplementedError
        else:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
            elif nms == "intra":
                cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
                ious *= cat_same.to(ious.device)
            else:
                raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]
        cls_embeds = cls_feats[valid_inds]
        if softmax_feats is not None:
            softmax_feats = softmax_feats[valid_inds]

        return bboxes, labels, embeds, cls_embeds, softmax_feats



@TRACKERS.register_module()
class OVTrackerSlack(object):
    def __init__(
            self,
            init_score_thr=0.0001,
            obj_score_thr=0.0001,
            match_score_thr=0.5,
            memo_frames=10,
            momentum_embed=0.8,
            momentum_obj_score=0.5,
            distractor_nms_thr=0.3,
            distractor_score_thr=0.5,
            match_metric="bisoftmax",
            match_with_cosine=True,
            contrastive_thr=0.5,
            confused_features=False,
            vis=True,
    ):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ["bisoftmax", "cosine"]
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine
        self.contrastive_thr = contrastive_thr

        self.confused_features = confused_features
        self.vis = vis

        self.reset()

        if self.vis and os.path.exists('/data1/clark/dataset/openDomain/TAO/tao/annotations/tao_validation_ours_v1_filename2ann.pth'):
            self.filename2ann = torch.load('/data1/clark/dataset/openDomain/TAO/tao/annotations/tao_validation_ours_v1_filename2ann.pth')

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v["ids"])
        return list(set(valid_ids))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, cls_embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
                ids[tracklet_inds],
                bboxes[tracklet_inds],
                embeds[tracklet_inds],
                cls_embeds[tracklet_inds],
                labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]["bboxes"].append(bbox)
                self.tracklets[id]["labels"].append(label)
                self.tracklets[id]["embeds"] = (
                                                       1 - self.momentum_embed
                                               ) * self.tracklets[id]["embeds"] + self.momentum_embed * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["frame_ids"].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    cls_embeds=cls_embed,
                    frame_ids=[frame_id],
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["frame_ids"][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        memo_cls_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v["bboxes"][-1][None, :])
            memo_labels.append(v["labels"][-1].view(1, 1))
            memo_embeds.append(v["embeds"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
        )

    def set_fusion_head(self, fusion_head, loss_cyc):
        self.fusion_head = fusion_head
        self.loss_cyc = loss_cyc
    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_new_objs, dtype=torch.long
        )
        self.num_tracklets += num_new_objs
        return ids

    def match(
            self,
            bboxes,
            labels,
            embeds,
            cls_embeds,
            frame_id,
            temperature=-1,
            method="ovtrack-teta",
            **kwargs
    ):
        """

        Args:
            bboxes:
            labels:
            track_feats: if use transformer method, the track_feats will be the encoder feats
            cls_feats: if use transformer method, the cls_feats will be the decoder feats
            frame_id:
            temperature:
            method: 'TETer'| 'oracle' | 'appearance' | 'contrastive'
            **kwargs:

        Returns:

        """

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        if method == "ovtrack-teta":
            # match if buffer is not empty
            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter"
            )
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo

                if self.confused_features:
                    # fused features here
                    # appearance, bbox, cls
                    uncertainty_list = [[], [], []]
                    # getting the consistency of each object
                    key_feats_list = [embeds, bboxes[:,:-1], cls_embeds]
                    ref_feats_list = [memo_embeds, memo_bboxes[:,:-1], memo_cls_embeds]
                    for i in range(len(key_feats_list)):
                        key_feat = key_feats_list[i]
                        ref_feat = ref_feats_list[i]
                        consistency, consistency_diag = self.loss_cyc.get_pair_consistency(
                            [key_feat, ref_feat])
                        uncertainty_list[i].append(consistency_diag)
                    key_assoc_conf, key_bbox_conf, key_cls_conf = [
                        torch.cat(tensor_list).view(-1, 1).detach() for tensor_list in uncertainty_list]

                    matching_indices = None
                    if self.vis and hasattr(self, 'filename2ann'):
                        # only used to visualizing the restuls
                        ann = self.filename2ann[kwargs['filename']]
                        gt_bbox_list = [an['bbox'] for an in ann]
                        det_bbox_list = bboxes[:,:-1].cpu().tolist()

                        def compute_iou(box1, box2):
                            """
                            计算两个边界框之间的IoU。
                            box1 和 box2 均为格式 (x1, y1, x2, y2)。
                            """
                            # 计算交集
                            x_left = max(box1[0], box2[0])
                            y_top = max(box1[1], box2[1])
                            x_right = min(box1[2], box2[2])
                            y_bottom = min(box1[3], box2[3])

                            if x_right < x_left or y_bottom < y_top:
                                return 0.0  # 没有交集

                            intersection_area = (x_right - x_left) * (y_bottom - y_top)

                            # 计算并集
                            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = box1_area + box2_area - intersection_area

                            # 计算IoU
                            iou = intersection_area / union_area
                            return iou

                        def get_matching_indices(gt_bbox_list, det_bbox_list, iou_threshold=0.5):
                            """
                            找出det_bbox_list中与gt_bbox_list的IoU大于iou_threshold的框子的索引位置。
                            返回：
                            - matching_det_indices: 检测框的匹配索引列表
                            - matching_gt_indices: 对应的真值框索引列表
                            """
                            matching_det_indices = []
                            matching_gt_indices = []

                            for det_index, det_box in enumerate(det_bbox_list):
                                max_iou = 0
                                max_gt_index = -1

                                # 找到与当前检测框IoU最大的真值框
                                for gt_index, gt_box in enumerate(gt_bbox_list):
                                    iou = compute_iou(det_box, gt_box)
                                    if iou > max_iou:
                                        max_iou = iou
                                        max_gt_index = gt_index

                                        # 如果最大IoU超过阈值，则记录这对匹配
                                if max_iou > iou_threshold:
                                    matching_det_indices.append(det_index)
                                    matching_gt_indices.append(max_gt_index)

                            return matching_det_indices, matching_gt_indices

                        matching_indices, matching_gt_indices = get_matching_indices(gt_bbox_list, det_bbox_list, iou_threshold=0.5)


                        # inter_app = [(det_bbox_list[index], key_assoc_conf[index].cpu().item()) for index in matching_indices if key_assoc_conf[index].item() != 1]
                        # inter_loc = [(det_bbox_list[index], key_bbox_conf[index].cpu().item()) for index in matching_indices if key_bbox_conf[index].item() != 1]
                        # inter_sem = [(det_bbox_list[index], key_cls_conf[index].cpu().item()) for index in matching_indices if key_cls_conf[index].item() != 1]

                        confused_key_feats, intra_loc_list, intra_sem_list = self.fusion_head(embeds, bboxes[:,:-1], cls_embeds, key_assoc_conf, key_bbox_conf, key_cls_conf, matching_indices)

                        # intra_loc = [(det_bbox_list[index], intra_loc_list[i]) for i, index in enumerate(matching_indices) if intra_loc_list[i] != 0]
                        # intra_sem = [(det_bbox_list[index], intra_sem_list[i]) for i, index in enumerate(matching_indices) if intra_sem_list[i] != 0]
                        total_dict = [{
                            'filename': kwargs['filename'],
                            'det_bbox': det_bbox_list[index],
                            'intra_loc': intra_loc_list[i],
                            'intra_sem': intra_sem_list[i],
                            'inter_app': key_assoc_conf[index].cpu().item(),
                            'inter_loc': key_bbox_conf[index].cpu().item(),
                            'inter_sem': key_cls_conf[index].cpu().item()
                        } for i, index in enumerate(matching_indices)]

                        for i, index in enumerate(matching_gt_indices):
                            total_dict[i]['gt_bbox'] = ann[index]['bbox']
                            total_dict[i]['category'] = ann[index]['category']
                            total_dict[i]['category_id'] = ann[index]['category_id']
                            total_dict[i]['track_id'] = ann[index]['track_id']

                        # torch.save(total_dict, os.path.join('/data1/clark/dataset/openDomain/TAO/tao/vis_data/tao_validation_conf_vis_with_track_id', kwargs['filename'].replace('/', '_').replace('.jpg', '.pth')))






                    else:
                        confused_key_feats = self.fusion_head(embeds, bboxes[:,:-1], cls_embeds, key_assoc_conf, key_bbox_conf, key_cls_conf)


                    embeds = confused_key_feats


                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )

                    exps = torch.exp(sims)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    cls_sims = cal_similarity(
                        cls_embeds,
                        memo_cls_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = cls_sims > self.contrastive_thr
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0

            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        elif method == "ovtrack-tao":

            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="intra"
            )
            # match if buffer is not empty
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo
                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    exps = torch.exp(sims) * cat_same.to(sims.device)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cos_scores *= cat_same.to(cos_scores.device)
                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0
                        m = self.momentum_obj_score
                        bboxes[i, -1] = (
                                m * bboxes[i, -1] + (1 - m) * memo_bboxes[memo_ind, -1]
                        )
            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        else:
            raise NotImplementedError

        if self.vis and hasattr(self, 'filename2ann') and 'matching_indices' in locals():
            for i, index in enumerate(matching_indices):
                pred_category_id = labels[index].cpu().item()
                pred_track_id = ids[index].cpu().item()
                total_dict[i]['pred_category_id'] = pred_category_id
                total_dict[i]['pred_track_id'] = pred_track_id
            torch.save(total_dict, os.path.join(
                '/data1/clark/dataset/openDomain/TAO/tao/vis_data/tao_validation_conf_vis_with_track_id',
                kwargs['filename'].replace('/', '_').replace('.jpg', '.pth')))

        return bboxes, labels, ids

    def remove_distractor(
            self,
            bboxes,
            labels,
            track_feats,
            cls_feats,
            object_score_thr=0.5,
            distractor_nms_thr=0.3,
            softmax_feats=None,
            nms="inter",
    ):

        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        if bboxes.shape[1] == 6:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-2], bboxes[:, :-2])
            else:
                raise NotImplementedError
        else:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
            elif nms == "intra":
                cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
                ious *= cat_same.to(ious.device)
            else:
                raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]
        cls_embeds = cls_feats[valid_inds]
        if softmax_feats is not None:
            softmax_feats = softmax_feats[valid_inds]

        return bboxes, labels, embeds, cls_embeds, softmax_feats



@TRACKERS.register_module()
class OVTrackerUncertaintyWithKalman(SortTracker):
    def __init__(
            self,
            init_score_thr=0.0001,
            obj_score_thr=0.0001,
            match_score_thr=0.5,
            memo_frames=10,
            momentum_embed=0.8,
            momentum_obj_score=0.5,
            distractor_nms_thr=0.3,
            distractor_score_thr=0.5,
            match_metric="bisoftmax",
            match_with_cosine=True,
            contrastive_thr=0.5,
            confused_features=False,
            motion_weight=None,
            init_cfg=None,
            kf=None,
            **kwargs
    ):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ["bisoftmax", "cosine"]
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine
        self.contrastive_thr = contrastive_thr

        self.confused_features = confused_features

        self.motion_weight = motion_weight
        self.kf = build_motion(kf)

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))
        # self.pred_tracks = defaultdict(partial(defaultdict, list))
        # self.gt_tracks = defaultdict(partial(defaultdict, list))

    @property
    def valid_ids(self):
        valid_ids = []
        for k, v in self.gt_tracks.items():
            valid_ids.extend(v["ids"])
        return list(set(valid_ids))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, labels, embeds, cls_embeds, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
                ids[tracklet_inds],
                bboxes[tracklet_inds],
                embeds[tracklet_inds],
                cls_embeds[tracklet_inds],
                labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]["bboxes"].append(bbox)
                self.tracklets[id]["labels"].append(label)
                self.tracklets[id]["embeds"] = (
                                                       1 - self.momentum_embed
                                               ) * self.tracklets[id]["embeds"] + self.momentum_embed * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["frame_ids"].append(frame_id)

                bbox = bbox_xyxy_to_cxcyah(self.tracklets[id].bboxes[-1][:-1].unsqueeze(dim=0))  # size = (1, 4)
                assert bbox.ndim == 2 and bbox.shape[0] == 1
                bbox = bbox.squeeze(0).cpu().numpy()
                self.tracklets[id].mean, self.tracklets[id].covariance = self.kf.update(
                    self.tracklets[id].mean, self.tracklets[id].covariance, bbox)

            else:
                self.tracklets[id] = Dict(
                    bboxes=[bbox],
                    labels=[label],
                    embeds=embed,
                    cls_embeds=cls_embed,
                    frame_ids=[frame_id],
                )
                # adding the kf code here
                bbox = bbox_xyxy_to_cxcyah(self.tracklets[id]['bboxes'][-1][:-1].unsqueeze(dim=0))  # size = (1, 4)
                assert bbox.ndim == 2 and bbox.shape[0] == 1
                bbox = bbox.squeeze(0).cpu().numpy()
                self.tracklets[id].mean, self.tracklets[id].covariance = self.kf.initiate(
                    bbox)


        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["frame_ids"][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_embeds = []
        memo_cls_embeds = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_bboxes.append(v["bboxes"][-1][None, :])
            memo_labels.append(v["labels"][-1].view(1, 1))
            memo_embeds.append(v["embeds"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
        )

    def set_fusion_head(self, fusion_head, loss_cyc):
        self.fusion_head = fusion_head
        self.loss_cyc = loss_cyc
    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_new_objs, dtype=torch.long
        )
        self.num_tracklets += num_new_objs
        return ids

    def match(
            self,
            bboxes,
            labels,
            embeds,
            cls_embeds,
            frame_id,
            temperature=-1,
            method="ovtrack-teta",
            **kwargs
    ):
        """

        Args:
            bboxes:
            labels:
            track_feats: if use transformer method, the track_feats will be the encoder feats
            cls_feats: if use transformer method, the cls_feats will be the decoder feats
            frame_id:
            temperature:
            method: 'TETer'| 'oracle' | 'appearance' | 'contrastive'
            **kwargs:

        Returns:

        """
        # setting the self.kf from OVTrack

        if not hasattr(self, 'kf'):
            raise Exception("NO KF ERROR")


        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        if method == "ovtrack-teta":
            # match if buffer is not empty
            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="inter"
            )
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo

                if self.confused_features:
                    # fused features here
                    # appearance, bbox, cls
                    uncertainty_list = [[], [], []]
                    # getting the consistency of each object
                    key_feats_list = [embeds, bboxes[:,:-1], cls_embeds]
                    ref_feats_list = [memo_embeds, memo_bboxes[:,:-1], memo_cls_embeds]
                    for i in range(len(key_feats_list)):
                        key_feat = key_feats_list[i]
                        ref_feat = ref_feats_list[i]
                        consistency, consistency_diag = self.loss_cyc.get_pair_consistency(
                            [key_feat, ref_feat])
                        uncertainty_list[i].append(consistency_diag)
                    key_assoc_conf, key_bbox_conf, key_cls_conf = [
                        torch.cat(tensor_list).view(-1, 1).detach() for tensor_list in uncertainty_list]

                    confused_key_feats = self.fusion_head(embeds, bboxes[:,:-1], cls_embeds, key_assoc_conf, key_bbox_conf, key_cls_conf)
                    embeds = confused_key_feats

                # recording the
                if self.match_metric == "bisoftmax":
                    pred_bbox = torch.zeros((len(self.tracklets), 4), device=bboxes.device)
                    for i, (track_id, track) in enumerate(self.tracklets.items()):
                        predict_mean, _ = self.kf.predict(track['mean'], track['covariance'])
                        pred_bbox[i] = torch.tensor(predict_mean[:4], dtype=torch.float32, device=bboxes.device)

                    pred_bbox_xyxy = bbox_cxcyah_to_xyxy(pred_bbox)
                    iou_dists = bbox_overlaps(bboxes[:, :4], pred_bbox_xyxy)



                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )

                    exps = torch.exp(sims)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    # used to cast to [0, 1]
                    # cos_scores = (1.0 + cos_scores) / 2

                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                    scores = (1 - self.motion_weight) * scores + self.motion_weight * iou_dists


                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")

                    cls_sims = cal_similarity(
                        cls_embeds,
                        memo_cls_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = cls_sims > self.contrastive_thr
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)

                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0

            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        elif method == "ovtrack-tao":

            bboxes, labels, embeds, cls_embeds, _ = self.remove_distractor(
                bboxes, labels, track_feats=embeds, cls_feats=cls_embeds, nms="intra"
            )
            # match if buffer is not empty
            if bboxes.size(0) > 0 and not self.empty:
                (
                    memo_bboxes,
                    memo_labels,
                    memo_embeds,
                    memo_cls_embeds,
                    memo_ids,
                ) = self.memo
                if self.match_metric == "bisoftmax":
                    sims = cal_similarity(
                        embeds,
                        memo_embeds,
                        method="dot_product",
                        temperature=temperature,
                    )
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    exps = torch.exp(sims) * cat_same.to(sims.device)
                    d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
                    t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cos_scores *= cat_same.to(cos_scores.device)
                    scores = (d2t_scores + t2d_scores) / 2
                    if self.match_with_cosine:
                        scores = (scores + cos_scores) / 2

                elif self.match_metric == "cosine":
                    cos_scores = cal_similarity(embeds, memo_embeds, method="cosine")
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    scores = cos_scores * cat_same.float().to(cos_scores.device)
                else:
                    raise NotImplementedError()

                num_objs = bboxes.size(0)
                ids = torch.full((num_objs,), -1, dtype=torch.long)
                for i in range(num_objs):
                    if bboxes[i, -1] < self.obj_score_thr:
                        continue
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    if conf > self.match_score_thr:
                        ids[i] = memo_ids[memo_ind]
                        scores[:i, memo_ind] = 0
                        scores[i + 1 :, memo_ind] = 0
                        m = self.momentum_obj_score
                        bboxes[i, -1] = (
                                m * bboxes[i, -1] + (1 - m) * memo_bboxes[memo_ind, -1]
                        )
            else:
                ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            # init tracklets
            ids = self.init_tracklets(ids, bboxes[:, -1])
            self.update_memo(ids, bboxes, labels, embeds, cls_embeds, frame_id)
        else:
            raise NotImplementedError

        return bboxes, labels, ids

    def remove_distractor(
            self,
            bboxes,
            labels,
            track_feats,
            cls_feats,
            object_score_thr=0.5,
            distractor_nms_thr=0.3,
            softmax_feats=None,
            nms="inter",
    ):

        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        if bboxes.shape[1] == 6:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-2], bboxes[:, :-2])
            else:
                raise NotImplementedError
        else:
            if nms == "inter":
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
            elif nms == "intra":
                cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
                ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
                ious *= cat_same.to(ious.device)
            else:
                raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]
        cls_embeds = cls_feats[valid_inds]
        if softmax_feats is not None:
            softmax_feats = softmax_feats[valid_inds]

        return bboxes, labels, embeds, cls_embeds, softmax_feats

def bbox_xyxy_to_cxcyah(bboxes):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, ratio, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    return xyah

def bbox_cxcyah_to_xyxy(bboxes):
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    return torch.cat(x1y1x2y2, dim=-1)