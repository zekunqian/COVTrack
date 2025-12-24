import copy
import warnings

from ..losses.multipos_cross_entropy_loss import MultiPosCrossEntropyLossUncertainty

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import clip
import os.path as osp
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
from tqdm import tqdm
from mmdet.core import multiclass_nms
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import DBSCAN
import numpy as np
from .class_name import *
from mmcv.runner import get_dist_info


@HEADS.register_module()
class OVTrackRoIHead(StandardRoIHead):
    def __init__(
        self,
        track_roi_extractor=None,
        track_head=None,
        track_train_cfg=None,
        cem_roi_extractor=None,
        cem_train_cfg=None,
        cem_head=None,
        finetune_track=False,
        kd_weight=256,
        fixed_lambda=None,
        prompt_path=None,
        fix_bg=False,
        ensemble=True,
        custom_classes=False,
        dynamic_rcnn_thre=True,
        only_validation_categories=False,
        only_test_categories=False,
        use_special_prompt=False,
        use_special_text_prompt=False,
        use_special_image_prompt=False,
        use_special_prompt_only_on_novel=False,
        prompt_word_list=["complete", "incomplete"],
        prompt_group_list=None,
        prompt_group_mean_way=True,
        self_train=False,
        only_self_train=False,
        self_train_rcnn=None,
        init_track_head_by_bbox_head=False,
        debug=False,
        bbox_counter = -1,
        cluster_num=-1,
        cluster_method='kmeans', # or dbscan
        two_stage_inference=False,
        simple_concate=True,
        prob_alpha=1.0,
        record_probs_info=False,
        after_softmax = False,
        prob_thres = 0.0,
        prob_dynamic = False,
        prob_dynamic_ratio = 0.1,
        spatial_loss=False,
        only_spatial_loss=False,
        top_k_spatial=10,
        iou_thres=0.95,
        spatial_learning_rate=0.01,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


        if track_head is not None:
            self.init_track_head(track_roi_extractor, track_head)

        if track_train_cfg is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_assigner_sampler()

        if cem_head is not None:
            self.init_cem_head(cem_roi_extractor, cem_head)
        else:
            self.cem_head = None

        if cem_train_cfg is not None:
            self.cem_train_cfg = cem_train_cfg
            self.init_cem_assigner_sampler()
        self.finetune_track = finetune_track

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.fix_bg = fix_bg
        self.custom_classes = custom_classes
        self.dynamic_rcnn_thre = dynamic_rcnn_thre
        if custom_classes:
            self.fixed_lambda = 0.3

        print("fixed_lambda", fixed_lambda)
        print("prompt path", prompt_path)
        self.text_features_for_classes = []
        self.ensemble = ensemble
        print("ensemble:{}".format(self.ensemble))

        if custom_classes:
            self.CLASSES = text_input
            self.num_classes = len(text_input)
        else:
            self.num_classes = self.bbox_head.num_classes
            if self.num_classes == 8:
                self.CLASSES = BDD_CLASSES
            elif self.num_classes == 1203:
                self.CLASSES = LVIS_CLASSES
            elif self.num_classes == 1053:
                self.CLASSES = OVTB_CLASSES
            else:
                print("For custom classes, please set custom_classes=True")

        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = ""

        print("load:", save_path)
        if osp.exists(save_path) and not custom_classes:
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).squeeze()[
                    : self.bbox_head.num_classes
                ]
                self.text_features_for_classes = self.text_features_for_classes.to(
                    device
                )

            else:
                self.text_features_for_classes = (
                    torch.load(save_path).to(device).squeeze()
                )
                print(self.text_features_for_classes.shape)
        else:
            print("Custom target classes: ", self.CLASSES)
            clip_model, self.preprocess = clip.load("ViT-B/32", device)
            clip_model.eval()
            for child in clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            if os.path.exists(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')):
                clip_path = os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')
                self.text_features_for_classes = torch.load(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth'), device)
                print(f'Loading clip features from {clip_path}')
            else:
                for template in tqdm(template_list):
                    print(template)
                    text_features_for_classes = torch.cat(
                        [
                            clip_model.encode_text(
                                clip.tokenize(template.format(c)).to(device)
                            ).detach()
                            for c in self.CLASSES
                        ]
                    )
                    self.text_features_for_classes.append(
                        F.normalize(text_features_for_classes, dim=-1)
                    )

                self.text_features_for_classes = torch.stack(
                    self.text_features_for_classes
                ).mean(dim=0)

                self.text_features_for_classes = self.text_features_for_classes.float()
                self.text_features_for_classes = F.normalize(
                    self.text_features_for_classes, dim=-1
                )

        # TODO minimize the size of the matrix
        self.only_validation_categories = only_validation_categories
        self.only_test_categories = only_test_categories
        self.use_special_prompt = use_special_prompt
        self.use_special_text_prompt = use_special_text_prompt
        self.use_special_image_prompt = use_special_image_prompt
        self.use_special_prompt_only_on_novel = use_special_prompt_only_on_novel
        self.prompt_group_mean_way = prompt_group_mean_way
        self.self_train = self_train
        self.only_self_train = only_self_train
        self.self_train_rcnn = self_train_rcnn
        self.debug = debug
        self.bbox_counter = bbox_counter
        self.bbox_head_shared_conv_state_dict = self.bbox_head.shared_convs.state_dict()
        self.bbox_head_shread_fc_state_dict = self.bbox_head.shared_fcs.state_dict()
        # print('getting bbox head state dict.....')
        # self.track_head_conv_state_dict = self.track_head.convs.state_dict()
        # self.track_head_fc_state_dict = self.track_head.fcs.state_dict()
        self.init_track_head_by_bbox_head = init_track_head_by_bbox_head
        self.init_track_head_tag = False
        # only used to debug
        self.mean_positive_prob = []
        self.debug_list = []
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method
        self.two_stage_inference = two_stage_inference
        self.simple_concat = simple_concate
        self.prob_alpha = prob_alpha
        self.record_prob_info = record_probs_info
        self.after_softmax = after_softmax
        self.prob_thres = prob_thres
        self.prob_dynamic = prob_dynamic
        self.prob_dynamic_ratio = prob_dynamic_ratio
        self.result_save_path = ''
        self.spatial_loss = spatial_loss
        self.only_spatial_loss = only_spatial_loss
        self.top_k_spatial = top_k_spatial
        self.iou_thres = iou_thres
        self.spatial_learning_rate = spatial_learning_rate
        print("prob_alpha:{}".format(self.prob_alpha))
        print("special_prompt:{}".format(use_special_prompt))
        print("prob_dynamic:{}".format(prob_dynamic))
        print("prob_dynamic_ratio:{}".format(prob_dynamic_ratio))
        print("prob_static_thres:{}".format(prob_thres))
        print("image prompt:{}".format(use_special_image_prompt))
        print("text prompt:{}".format(use_special_text_prompt))
        print("spatial loss:{}".format(spatial_loss))
        print("top k spatial:{}".format(top_k_spatial))
        print("top k iou thres:{}".format(iou_thres))
        print("spatial learning rate:{}".format(spatial_learning_rate))
        # self.empty_fc = nn.Linear(1,1)

        # judge whether we use validation or test dataset
        assert not (only_validation_categories == True and only_test_categories == True), f"Error category combination type {only_validation_categories}, {only_test_categories}"

        if only_validation_categories:
            indices = [col - 1 for col in tao_validation_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(validation_novel_index, device=device)
        if only_test_categories:
            indices = [col - 1 for col in tao_test_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(test_novel_index, device=device)


        if self.use_special_prompt:
            # assert len(prompt_word_list) == 2, f"Error: no matching format of {prompt_word_list}"
            # clip_model, self.preprocess = clip.load("ViT-B/32", device)
            # clip_model.eval()
            # for child in clip_model.children():
            #     for param in child.parameters():
            #         param.requires_grad = False
            # self.special_prompt = clip_model.encode_text(
            #     clip.tokenize(prompt_word_list).to(device)
            # ).float().detach()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

            self.prompt_group_list = prompt_group_list
            if self.prompt_group_list is not None:
                assert len(prompt_group_list) % 2 == 0, f"Error: no pair of prompt format of {prompt_group_list}"
                clip_model, self.preprocess = clip.load("ViT-B/32", device)
                clip_model.eval()
                for child in clip_model.children():
                    for param in child.parameters():
                        param.requires_grad = False
                self.special_prompt = clip_model.encode_text(
                    clip.tokenize(prompt_group_list).to(device)
                ).float().detach()
                self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1).view(-1, 2,
                                                                                                          512).permute( 0, 2, 1)

            # self.special_prompt = torch.load('/data1/clark/models/detpro/special_prompt/complete_incomplete.pth', map_location=device).float()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

        print(self.text_features_for_classes.shape)

        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1, 512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
        self.projection = nn.Linear(1024, 512)
        # if self.ensemble:
        self.projection_for_image = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.projection_for_image.weight)
        nn.init.constant_(self.projection_for_image.bias, 0)

        if self.num_classes == 1230:
            self.base_label_ids = torch.tensor(lvis05_base, device=device)
            self.novel_label_ids = torch.tensor(lvis05_novel, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 1203:

            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 8:
            self.base_label_ids = torch.tensor(bdd_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(bdd_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_roi_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get("sampler", None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self
            )
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_track_head_by_bbox_head_method(self):
        self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
        self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
        print("Loaded track head convs and fcs from bbox head in init_track_head_by_bbox_head_method")
        self.init_track_head_tag = True

    def init_cem_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.cem_train_cfg.get("assigner", None):
            self.cem_roi_assigner = build_assigner(self.cem_train_cfg.assigner)
            self.cem_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.cem_share_assigner = True

        if self.cem_train_cfg.get("sampler", None):
            self.cem_roi_sampler = build_sampler(
                self.cem_train_cfg.sampler, context=self
            )
            self.cem_share_sampler = False
        else:
            self.cem_roi_sampler = self.bbox_sampler
            self.cem_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "track_head") and self.track_head is not None

    @property
    def with_cem(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "cem_head") and self.cem_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_cem_head(self, cem_roi_extractor, cem_head):
        """Initialize ``track_head``"""
        if cem_roi_extractor is not None:
            self.cem_roi_extractor = build_roi_extractor(cem_roi_extractor)
            self.cem_share_extractor = False
        else:
            self.cem_share_extractor = True
            self.cem_roi_extractor = self.bbox_roi_extractor
        self.cem_head = build_head(cem_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()
        if self.with_cem:
            self.cem_head.init_weights()
            if not self.cem_share_extractor:
                self.cem_roi_extractor.init_weights()

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_match_indices,
        ref_x,
        ref_img_metas,
        ref_proposals,
        ref_gt_bboxes,
        ref_gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        *args,
        **kwargs
    ):
        # if not self.finetune_track:
        #     losses = super().forward_train(
        #         x,
        #         img_metas,
        #         proposal_list,
        #         gt_bboxes,
        #         gt_labels,
        #         gt_bboxes_ignore,
        #         gt_masks,
        #         *args,
        #         **kwargs
        #     )
        # else:
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Init track shared conv head!")
            self.init_track_head_tag = True

        losses = {}

        key_gt_match_indices_correct = gt_match_indices[0].cpu().tolist()
        ref_gt_match_indices_correct = kwargs['ref_gt_match_indices'][0].cpu().tolist()

        key_gt_label = gt_labels[0].cpu().tolist()
        ref_gt_label = ref_gt_labels[0].cpu().tolist()

        key_match_label = [ref_gt_label[index] == key_gt_label[i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
        ref_match_label = [key_gt_label[index] == ref_gt_label[i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]



        num_imgs = len(img_metas)

        if self.self_train:
            # TODO finishing self train code here
            # score_thr is dynamic one
            pre_nms = None
            pre_max_per_img = None
            if self.self_train_rcnn.nms != self.test_cfg.nms:
                pre_nms = copy.deepcopy(self.test_cfg.nms)
                self.test_cfg.nms = self.self_train_rcnn.nms
            if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
                pre_max_per_img = self.test_cfg.max_per_img
                self.test_cfg.max_per_img = self.self_train_rcnn.max_per_img

            key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x, img_metas, proposal_list, False)
            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            ref_det_bboxes, ref_det_labels, ref_cem_feats, ref_track_feats = self.self_train_forward(ref_x, ref_img_metas, ref_proposals, False)
            num_det_per_img = tuple(det.shape[0] for det in ref_det_bboxes)
            ref_track_feats = ref_track_feats.split(num_det_per_img, 0)

            if self.debug:
                # only use to debug
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                if ref_gt_bboxes_ignore is None:
                    ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    key_positive_prob = (torch.sum(assign_result.max_overlaps > 0.5) / len(key_det_bboxes[i])).item()
                    key_sampling_results.append(key_positive_prob)
                    self.mean_positive_prob.append(key_positive_prob)

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[i],
                        ref_gt_bboxes[i],
                        ref_gt_bboxes_ignore[i],
                        ref_gt_labels[i],
                    )
                    ref_positive_prob = (torch.sum(ref_assign_result.max_overlaps > 0.5) / len(ref_det_bboxes[i])).item()
                    ref_sampling_results.append(ref_positive_prob)
                    self.mean_positive_prob.append(ref_positive_prob)
                print(f"{sum(key_sampling_results)/len(key_sampling_results)}, {sum(ref_sampling_results)/len(ref_sampling_results)}, {sum(self.mean_positive_prob)/len(self.mean_positive_prob)}")

            # calculating cycAss loss image by image
            loss_cyc = 0.0
            for i in range(len(img_metas)):
                features_list = [key_track_feats[i], ref_track_feats[i]]
                loss_cyc += self.track_head.self_loss(features_list)['cyc_loss']
            losses.update(dict(cyc_loss=loss_cyc/len(img_metas)))

            if pre_nms is not None:
                self.test_cfg.nms = pre_nms
            if pre_max_per_img is not None:
                self.test_cfg.max_per_img = pre_max_per_img

            if self.only_self_train:
                return losses


        if self.with_track or self.with_cem:
            if self.debug:
                with torch.no_grad():
                    key_det_bboxes, key_det_labels, [key_text_region_features, key_image_region_features,
                                             key_score_matrix, key_det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    key_det_inds = [det_ind//296 for det_ind in key_det_inds]
                    ref_det_bboxes, ref_det_labels, [ref_text_region_features, ref_image_region_features,
                                                     ref_score_matrix, ref_det_inds] = self.simple_test_bboxes(
                        ref_x, ref_img_metas, ref_proposals, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    ref_det_inds = [det_ind//296 for det_ind in ref_det_inds]

                    before_add_gt_as_proposals = self.track_roi_sampler.add_gt_as_proposals
                    before_pos_iou_thr = self.track_roi_assigner.pos_iou_thr
                    self.track_roi_sampler.add_gt_as_proposals = False
                    self.track_roi_assigner.pos_iou_thr = 0.5

                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    if ref_gt_bboxes_ignore is None:
                        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[0][:,:4], gt_bboxes[0], gt_bboxes_ignore[0], gt_labels[0]
                    )
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        key_det_bboxes[0][:,:4],
                        gt_bboxes[0],
                        gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in x],
                    )

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_bboxes_ignore[0],
                        ref_gt_labels[0],
                    )
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in ref_x],
                    )
                    self.track_roi_sampler.add_gt_as_proposals = before_add_gt_as_proposals
                    self.track_roi_assigner.pos_iou_thr = before_pos_iou_thr

                    # save tmp result to analyse
                    # bbox, labels, region_embedding, cls_score
                    key_save_bbox = sampling_result.pos_bboxes.cpu()
                    key_save_labels = sampling_result.pos_gt_labels.cpu()
                    key_save_image_embedding = key_image_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_text_embedding = key_text_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_score_matrix = key_score_matrix[0][key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_list = [key_save_score_matrix, key_save_text_embedding, key_save_image_embedding, key_save_bbox, key_save_labels]
                    self.debug_list.append(key_save_list)

                    ref_save_bbox = ref_sampling_result.pos_bboxes.cpu()
                    ref_save_labels = ref_sampling_result.pos_gt_labels.cpu()
                    ref_save_image_embedding = ref_image_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_text_embedding = ref_text_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_score_matrix = ref_score_matrix[0][ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_list = [ref_save_score_matrix, ref_save_text_embedding, ref_save_image_embedding, ref_save_bbox, ref_save_labels]
                    self.debug_list.append(ref_save_list)
                    if len(self.debug_list) > 50:
                        torch.save(self.debug_list, '/home/clark/test/ovtrack_test/cluster_analyse/original_list.pth')
                        print('hello world')





            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]

            if self.with_track:
                key_feats = self._track_forward(x, key_bboxes)
                ref_bboxes = [res.bboxes for res in ref_sampling_results]
                ref_feats = self._track_forward(ref_x, ref_bboxes)

                match_feats = self.track_head.match(
                    key_feats, ref_feats, key_sampling_results, ref_sampling_results
                )
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results
                )
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                losses.update(loss_track)

        return losses



    def forward_train_self(
            self,
            x_list,
            img_metas_list,
            proposal_list_for_x,
            *args,
            **kwargs
    ):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in forward_train_self")
            self.init_track_head_tag = True


        losses = {}

        num_imgs = len(img_metas_list[0])

        self_test_cfg = copy.deepcopy(self.test_cfg)
        # pre_nms = None
        # pre_max_per_img = None
        pre_dynamic_rcnn_thre = None
        # pre_score_thr = None
        if self.self_train_rcnn.nms != self.test_cfg.nms:
            # pre_nms = copy.deepcopy(self.test_cfg.nms)
            self_test_cfg.nms = self.self_train_rcnn.nms
        if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
            # pre_max_per_img = self.test_cfg.max_per_img
            self_test_cfg.max_per_img = self.self_train_rcnn.max_per_img
        if self.self_train_rcnn.dynamic_rcnn_thre != self.dynamic_rcnn_thre:
            pre_dynamic_rcnn_thre = self.dynamic_rcnn_thre
            self.dynamic_rcnn_thre = self.self_train_rcnn.dynamic_rcnn_thre
        if self.self_train_rcnn.score_thr != self.test_cfg.score_thr:
            # pre_score_thr = self.test_cfg.score_thr
            self_test_cfg.score_thr = self.self_train_rcnn.score_thr

        # det_bboxes_list, det_labels_list, cem_feats_list, track_feats_list = [], [], [], []
        key_feats_list = [[] for _ in range(num_imgs)]
        cluster_feats_list = [[] for _ in range(num_imgs)]

        spatial_xyxy_list = [[] for _ in range(num_imgs)]
        spatial_key_feats_list = [[] for _ in range(num_imgs)]

        # spatial_key_feats_list = [[] for _ in range(num_imgs)]
        # spatial_xyxy_list = [[] for _ in range(num_imgs)]

        #debug
        # e, i, x_list, img_metas_list, proposal_list_for_x = torch.load('/home/clark/test/ovtrack_error/self_train_forward_1705485087_5129771')

        for i in range(len(x_list)):
            outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False, test_config=self_test_cfg, cluster_num=self.cluster_num)
            #empty prediction
            while len(outputs) == 3:
                # print(f'No enough box in {img_metas_list[i][0]["filename"]}')
                tmp_self_cfg = copy.deepcopy(self_test_cfg)
                tmp_self_cfg.score_thr=self_test_cfg.score_thr/10
                outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False,
                                                  test_cfg=tmp_self_cfg)
                # return dict(cyc_loss=0*self.empty_fc(torch.tensor([[1.0]], device=key_det_bboxes[0].device)).sum()/1.0)
            if self.cluster_num != -1:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats, cls_score = outputs
            else:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = outputs
            # key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], False)
        # except Exception as e:
        #     torch.save([e, i, x_list, img_metas_list, proposal_list_for_x], f'/home/clark/test/ovtrack_error/self_train_forward_{str(time.time()).replace(".","_")}')
        #     # print(f'Error:{e}\ni:{i}\nx_list:{x_list[i]}\nimg_metas_list:{img_metas_list[i]}\nproposal_list_for_x:{proposal_list_for_x[i]} in self_train_forward')
        #     print(f'No enough box in {img_metas_list[i]["filename"]}')
        #     return dict(cyc_loss=x_list[0][0].sum()*0)

            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            for j in range(num_imgs):
                if key_track_feats[j].size(0) != 0:
                    key_feats_list[j].append(key_track_feats[j])
                if self.cluster_num != -1:
                    cluster_feats_list[j].append(cls_score[j])
                    if self.spatial_loss:
                        spatial_xyxy_list[j].append(key_det_bboxes[j])
        if self.bbox_counter != -1:
            for i in range(len(key_det_bboxes)):
                self.bbox_counter += key_det_bboxes[i].shape[0]
            print(self.bbox_counter)

        loss_cyc = 0.0
        loss_spatial = 0.0
        for i in range(num_imgs):
            # making cluster
            try:
                if self.cluster_num != -1:
                    num_det_per_img = tuple(cluster_feat.shape[0] for cluster_feat in cluster_feats_list[i])
                    cluster_feats = torch.cat(cluster_feats_list[i], dim=0)
                    X = cluster_feats.cpu()
                    if self.cluster_method == 'kmeans':
                        k_means_cluster_labesl = self.k_means_cluster(X, n_clusters=self.cluster_num)
                    elif self.cluster_method == 'dbscan':
                        k_means_cluster_labesl = self.dbscan_cluster(X)
                    else:
                        raise Exception(f'Unknown cluster type: {self.cluster_method}')
                    # max_cluster_index = max(k_means_cluster_labesl)
                    max_frequency_cluster_index = np.argmax(np.bincount(k_means_cluster_labesl))
                    cluster_list = torch.tensor(k_means_cluster_labesl, device=cluster_feats.device).split(num_det_per_img, 0)
                    if self.spatial_loss:
                        spatial_xyxy_list[i] = [
                            spatial_xyxy_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in
                            range(len(spatial_xyxy_list[i]))]
                        spatial_key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]
                    key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]


                if len(key_feats_list[i]) > 1:
                    loss_cyc += self.track_head.self_loss(key_feats_list[i])['cyc_loss']
                    if self.spatial_loss:
                        loss_spatial += self.track_head.self_spatial_loss(spatial_key_feats_list[i], spatial_xyxy_list[i], self.top_k_spatial, self.iou_thres, self.spatial_learning_rate)
            except Exception as e:
                # torch.save([e, i, key_feats_list], f'/home/clark/test/ovtrack_error/cyc_loss_{str(time.time()).replace(".","_")}')
                # print(f'All empty det Error:{e}\ni:{i}\nkey_feats_list:{key_feats_list[i]} in traverse cyc loss')
                # print(e, key_track_feats[0])
                if not self.spatial_loss:
                    return dict(cyc_loss=0.0*key_track_feats[0].sum())
                else:
                    if not self.only_spatial_loss:
                        return dict(cyc_loss=0.0 * key_track_feats[0].sum(), spatial_loss=0.0 * key_track_feats[0].sum())
                    else:
                        return dict(spatial_loss=0.0 * key_track_feats[0].sum())

                # for det_bbox in key_det_bboxes:
                #     if det_bbox.size(0) != 0:
                #         return dict(cyc_loss=det_bbox.sum() * 0)
                # return dict(cyc_loss=x_list[0][0].sum()*0)
        if type(loss_cyc) == float:
            loss_cyc = 0.0*key_track_feats[0].sum()
        if type(loss_spatial) == float:
            loss_spatial = 0.0*key_track_feats[0].sum()


        if not self.spatial_loss:
            losses.update(dict(cyc_loss=loss_cyc/num_imgs))
        else:
            if not self.only_spatial_loss:
                losses.update(dict(cyc_loss=loss_cyc/num_imgs, spatial_loss=loss_spatial/num_imgs))
            else:
                losses.update(dict(spatial_loss=loss_spatial/num_imgs))



        # if pre_nms is not None:
        #     self.test_cfg.nms = pre_nms
        # if pre_max_per_img is not None:
        #     self.test_cfg.max_per_img = pre_max_per_img
        if pre_dynamic_rcnn_thre is not None:
            self.dynamic_rcnn_thre = pre_dynamic_rcnn_thre
        # if pre_score_thr is not None:
        #     self.test_cfg.score_thr = pre_score_thr



        return losses

    def k_means_cluster(self, X, n_clusters=10, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels_pred = kmeans.labels_
        return labels_pred

    def dbscan_cluster(self, X, eps=0.04, min_samples=3):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = clustering.labels_
        return cluster_labels

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[: self.track_roi_extractor.num_inputs], rois
        )
        track_feats = self.track_head(track_feats) # 203, 256, 7, 7 -> 203, 256
        return track_feats

    def _cem_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        cem_feats = self.cem_roi_extractor(x[: self.cem_roi_extractor.num_inputs], rois)
        cem_feats = self.cem_head(cem_feats)

        return cem_feats

    def _clip_cem_forward(self, x, bboxes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return region_embeddings

    def simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in simple_test")
            self.init_track_head_tag = True

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None
        if self.record_prob_info:
            if det_bboxes.shape[1] == 5: # no prompt
                track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
            else:
                track_bboxes = det_bboxes[:, :-2] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
        else:
            track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                img_metas[0]["scale_factor"]
            ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def self_train_forward(self, x, img_metas, proposal_list, rescale, test_config=None, cluster_num=-1, **kwargs):


        with torch.no_grad():

            if cluster_num != -1:
                if test_config is None:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                         score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True
                    )
                else:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                         score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale, return_features=True, return_inds=True
                    )
            else:
                if test_config is None:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale
                    )
                else:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale
                    )


        all_zero_boox_tag = True
        for det_bbox in det_bboxes:
            if det_bbox.size(0) != 0:
                all_zero_boox_tag = False
                break
        if all_zero_boox_tag:
            return det_bboxes, det_labels, None

            # if det_bboxes[0].size(0) == 0:
            # return det_bboxes[0], det_labels[0], None


        track_feats = self._track_forward(x, det_bboxes)
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, det_bboxes)
        else:
            cem_feats = None

        if cluster_num != -1:
            det_inds = [ind//(len(self.validation_index)-1) for ind in det_inds]
            score_matrix = [score_matrix[i][det_inds[i]] for i in range(len(score_matrix))]
            return det_bboxes, det_labels, cem_feats, track_feats, score_matrix
        else:
            return det_bboxes, det_labels, cem_feats, track_feats


    def simple_test_with_fixed_dets(
        self, x, det_bboxes, det_labels, img_metas, **kwargs
    ):

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        # #TODO localization using (0,1,2,3)
        # # if finest_scale == 0.1(layer_tag==True) means only use layer(3) as classification
        # layer_tag = False
        # if self.bbox_roi_extractor.finest_scale != 56:
        #     layer_tag = True
        #     self.bbox_roi_extractor.finest_scale=56
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        # bbox_pred = self.bbox_head(region_embeddings)
        # bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        #
        # #TODO classification using (3)
        # if layer_tag:
        #     self.bbox_roi_extractor.finest_scale=0.1
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)

        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

        # return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, return_features=False, return_inds=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            return_features: used to return region features and matching score matrix
            return_inds: if return the selected indices

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # st1 = time.time()
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        rois = bbox2roi(proposals)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(
                bg_class_embedding, p=2, dim=1
            )
            text_features = torch.cat(
                [self.text_features_for_classes, bg_class_embedding], dim=0
            )
        else:
            text_features = self.text_features_for_classes

        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)

        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(
                region_embeddings_image, p=2, dim=1
            )
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)

        if self.use_special_prompt:
            assert self.use_special_text_prompt == True or self.use_special_image_prompt == True, "no matching special prompt types"

            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            text_complete_result = region_embeddings @ self.special_prompt.float()
            text_complete_result = text_complete_result / 0.007
            text_complete_result = text_complete_result.softmax(dim=-1).permute(1, 0, 2)

            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            img_complete_result = region_embeddings_image @ self.special_prompt.float()
            img_complete_result = img_complete_result / 0.007
            img_complete_result = img_complete_result.softmax(dim=-1).permute(1, 0, 2)

            if self.prompt_group_mean_way:
                text_complete_result = text_complete_result.mean(dim=1)[:, 0].view(-1, 1)
                img_complete_result = img_complete_result.mean(dim=1)[:, 0].view(-1, 1)

            if self.use_special_text_prompt and self.use_special_image_prompt:
                complete_prob = (text_complete_result + img_complete_result)/2
            elif self.use_special_text_prompt:
                complete_prob = text_complete_result
                self.prob_thres = 0.005 # 0.13 0.15
                # print(f'only text reset the text prompt prob')
            elif self.use_special_image_prompt:
                complete_prob = img_complete_result
                self.prob_thres = 0.45 # 0.38
                # print(f'only img reset the img prompt prob')
            # complete_prob = complete_prob[:, 0].view(-1, 1)



            # old one
            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            # text_complete_result = text_complete_result / 0.007
            # text_complete_result = text_complete_result.softmax(dim=-1)
            #
            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            # img_complete_result = img_complete_result / 0.007
            # img_complete_result = img_complete_result.softmax(dim=-1)
            #
            # # complete_prob = (text_complete_result + img_complete_result) / 2
            # if self.use_special_text_prompt and self.use_special_image_prompt:
            #     complete_prob = (text_complete_result + img_complete_result)/2
            # elif self.use_special_text_prompt:
            #     complete_prob = text_complete_result
            # elif self.use_special_image_prompt:
            #     complete_prob = img_complete_result
            # complete_prob = complete_prob[:, 0].view(-1, 1)

        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = (
                    cls_score_image ** (1 - self.fixed_lambda)
                    * cls_score_text ** self.fixed_lambda
                )
            else:
                if self.only_validation_categories or self.only_test_categories:

                    cls_score = torch.where(
                        self.validation_index,
                        # self.novel_index, #TODO only test minimize list
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                    )
                else:
                    cls_score = torch.where(
                        self.novel_index,
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                        )


        else:
            cls_score = cls_score_text


        # TODO ONLY TEST SMALL DATASET HERE
        # indices = [col - 1 for col in tao_validation_used_cate_ids] + [cls_score.shape[1] - 1]
        # cls_score = cls_score[:, indices]
        # cls_score = torch.index_select(cls_score, 1, torch.tensor(indices, device=cls_score.device))


        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if self.use_special_prompt:
            complete_prob = complete_prob.split(num_proposals_per_img, 0)

            # cls_score = torch.where(
            #     self.validation_index,
            #     # self.novel_index, #TODO only test minimize list
            #     cls_score_image ** (1 - a) * cls_score_text ** a,
            #     cls_score_text ** (1 - a) * cls_score_image ** a,
            #     )

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        if self.dynamic_rcnn_thre:
            rcnn_test_cfg.score_thr = (1 / len(text_features)) * 1.001
            # print(rcnn_test_cfg.score_thr)



        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_inds = []
        if self.use_special_prompt:
            if self.two_stage_inference:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i],
                            cls_score[i]
                        )
                    else:
                        # cls_score_new = cls_score[i]*complete_prob[i]
                        cls_score_new = cls_score[i]

                        # first stage
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        det_index_row = det_index // (cls_score_new.shape[1] - 1)
                        det_label = det_label
                        base_index = self.validation_index[det_label] == False
                        one_stage_det_bbox = det_bbox[base_index]
                        one_stage_det_label = det_label[base_index]

                        # two stage
                        # simple concate way
                        if self.simple_concat:
                            two_stage_det_bbox = det_bbox[~base_index]
                            # two_stage_det_label = det_label[~base_index]

                            cls_score_new = cls_score_new[det_index_row][~base_index]*complete_prob[i][det_index_row][~base_index]
                            scores = F.softmax(cls_score_new, dim=-1)
                            new_two_stage_det_bbox, new_two_stage_det_labels = multiclass_nms(
                                two_stage_det_bbox[:,:-1], scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img
                            )

                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)
                        else:
                            indices_to_keep = torch.tensor(list(set(range(len(rois[i]))) - set(det_index_row[base_index].tolist())))
                            cls_score_new = cls_score[i]*complete_prob[i]
                            new_two_stage_det_bbox, new_two_stage_det_labels = self.bbox_head.get_bboxes(
                                rois[i][indices_to_keep],
                                # cls_score[i]*complete_prob[i],
                                cls_score_new[indices_to_keep],
                                bbox_pred[i][indices_to_keep],
                                img_shapes[i],
                                scale_factors[i],
                                rescale=rescale,
                                cfg=rcnn_test_cfg,
                            )
                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)

                        det_bboxes.append(det_bbox)
                        det_labels.append(det_label)




                    #     else:
                    #         det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                    #             rois[i],
                    #             # cls_score[i]*complete_prob[i],
                    #             cls_score_new,
                    #             bbox_pred[i],
                    #             img_shapes[i],
                    #             scale_factors[i],
                    #             rescale=rescale,
                    #             cfg=rcnn_test_cfg,
                    #             return_inds=return_inds
                    #         )
                    #
                    #     det_inds.append(det_index)
                    #
                    # det_bboxes.append(det_bbox)
                    # det_labels.append(det_label)
                    # if self.two_stage_inference:
                    #     for i in range(len(det_labels)):
                    #         # det_index = (det_inds[i] // (cls_score_new.shape[1]-1))
                    #         det_bbox = det_bboxes[i]
                    #         det_label = det_labels[i]
                    #         cls_score_new_norm = F.softmax(cls_score_new, dim=-1)
                    #         cls_score_ori = [cls_score_new_norm[i] for i in det_index.tolist()]
                    #         cls_score_ori_zip = [round(cls_score_ori[i][det_label[i]].item(),7) for i in range(len(det_label))]
                    #         det_cls_score_zip = [round(item.item(), 7) for item in det_bbox[:,-1]]
                    #
                    #
                    #
                    #
                    #         print('hello world')

            else:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i]*self.prob_alpha,
                            cls_score[i]
                        )
                    else:
                        if not self.after_softmax:
                            cls_score_new = cls_score[i]*complete_prob[i]*self.prob_alpha
                        else:
                            cls_score_new = cls_score[i]
                            self.bbox_head.after_softmax = self.after_softmax
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.prob_thres = self.prob_thres
                            self.bbox_head.prob_dynamic = self.prob_dynamic
                            self.bbox_head.prob_dynamic_ratio = self.prob_dynamic_ratio
                    if not return_inds and not self.record_prob_info:
                        det_bbox, det_label = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                        )
                    else:
                        if self.record_prob_info:
                            self.bbox_head.result_save_path = self.result_save_path
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.record_prob_info = self.record_prob_info
                            self.bbox_head.filename = img_metas[i]['ori_filename']
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        if self.record_prob_info:

                            # record the prob
                            det_index_row = det_index // (cls_score_new.shape[1] - 1)
                            probs = complete_prob[i][det_index_row]
                            # det_inds.append(det_index_row)
                            det_bbox = torch.cat([det_bbox[:, :4], probs, det_bbox[:, -1].reshape(-1, 1)], dim=-1)
                        else:
                            det_inds.append(det_index)

                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)

        else:
            for i in range(len(proposals)):
                if not return_inds:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        )
                else:
                    det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        return_inds=return_inds
                    )
                    det_inds.append(det_index)

                det_bboxes.append(det_bbox)
                det_labels.append(det_label)

        if self.only_validation_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if self.only_test_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_test_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if not return_features or self.record_prob_info:
            return det_bboxes, det_labels
        else:
            assert return_features is True and return_inds is True, 'return features and inds do not match!'
            return det_bboxes, det_labels, [region_embeddings, region_embeddings_image, cls_score, det_inds]




@HEADS.register_module()
class FeatureFusionModule(nn.Module):
    def __init__(self, feature_dim=512, max_fusion_ratio=0.5, if_detach_asso_feat=True, add_res=True,
                 no_cls_feats=False, no_loc_feats=False,
                 no_uc=False, no_ul=False, no_ua=False,
                 no_self_cls=False, no_self_loc=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_fusion_ratio = max_fusion_ratio
        self.if_detach_asso_feat = if_detach_asso_feat
        # 辅助特征转换层
        self.bbox_transform = nn.Linear(4, feature_dim)
        self.cls_transform = nn.Linear(512, feature_dim)
        self.add_res = add_res

        #balation study
        self.no_cls_feats = no_cls_feats
        self.no_loc_feats = no_loc_feats

        self.no_uc = no_uc
        self.no_ul = no_ul
        self.no_ua = no_ua

        self.no_self_cls = no_self_cls
        self.no_self_loc = no_self_loc


        # 辅助信息门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # 只为bbox和cls生成门控值
            nn.Sigmoid()  # 使用sigmoid确保门控值在[0,1]之间
        )

        # 残差融合层
        self.residual_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # self.norm = nn.LayerNorm(feature_dim)
        self._init_weights()

    def _init_weights(self):
        """
        初始化模型的权重：
        - 对于有ReLU的线性层使用kaiming初始化
        - 对于输出层使用xavier初始化
        - 对于bias使用0初始化
        """
        # bbox和cls转换层初始化
        nn.init.kaiming_normal_(self.bbox_transform.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.bbox_transform.bias, 0)

        nn.init.kaiming_normal_(self.cls_transform.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.cls_transform.bias, 0)

        # 门控网络初始化
        for name, module in self.gate_network.named_children():
            if isinstance(module, nn.Linear):
                if 'weight' in module._parameters:
                    if name == '0':  # 第一层，后面接ReLU
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    else:  # 输出层，接sigmoid
                        nn.init.xavier_normal_(module.weight, gain=1.0)
                if 'bias' in module._parameters:
                    nn.init.constant_(module.bias, 0)

        # 残差层初始化
        for name, module in self.residual_layer.named_children():
            if isinstance(module, nn.Linear):
                if 'weight' in module._parameters:
                    if name == '0':  # 第一层，后面接ReLU
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    else:  # 输出层
                        # 使用较小的初始化范围，因为是残差连接
                        nn.init.xavier_normal_(module.weight, gain=0.01)
                if 'bias' in module._parameters:
                    nn.init.constant_(module.bias, 0)
    def calculate_adaptive_fusion_ratio(self, assoc_conf):
        """
        计算自适应融合比例：
        1. 当assoc_conf接近1时，fusion_ratio接近0
        2. 当assoc_conf接近0时，fusion_ratio接近max_fusion_ratio
        3. 使用平滑的非线性函数确保过渡自然
        """
        # 使用1-assoc_conf作为基础，确保conf高时ratio低
        # base_ratio = 1 - assoc_conf

        # 使用sigmoid函数使转换更平滑
        # 将[0,1]映射到[0, max_fusion_ratio]
        # adaptive_ratio = self.max_fusion_ratio * (1 - torch.exp(-2 * base_ratio)) / (1 + torch.exp(-2 * base_ratio))
        adaptive_ratio = self.max_fusion_ratio * ((1 - assoc_conf) ** 2)

        return adaptive_ratio

    def forward(self, assoc_feat, bbox_feat, cls_feat, assoc_conf, bbox_conf, cls_conf, return_index_list = None):
        """
        Args:
            assoc_feat: association特征 [B, 512] (主特征)
            bbox_feat: bbox特征 [B, 512] (辅助特征)
            cls_feat: classification特征 [B, 512] (辅助特征)
            assoc_conf: association置信度 [B, 1]
            bbox_conf: bbox置信度 [B, 1]
            cls_conf: classification置信度 [B, 1]
        Returns:
            enhanced_feat: 增强后的association特征 [B, 512]
        """

        batch_size = assoc_feat.size(0)

        # abalation study
        if self.no_cls_feats:
            cls_feat *= 0
        if self.no_loc_feats:
            bbox_feat *= 0




        if self.no_uc:
            cls_conf.fill_(1.0)
        if self.no_ul:
            bbox_conf.fill_(1.0)
        if self.no_ua:
            assoc_conf.fill_(1.0)


        # 1. 转换辅助特征
        bbox_feat = self.bbox_transform(bbox_feat)
        cls_feat = self.cls_transform(cls_feat)
        assert assoc_feat.shape[-1] == bbox_feat.shape[-1] == cls_feat.shape[-1], f"Unmatched confusion shape as : asso: {assoc_feat.shape}, bbox: {bbox_feat.shape}, cls:{cls_feat.shape}"

        # 2. 计算辅助特征的门控值
        concat_features = torch.cat([assoc_feat.detach() if self.if_detach_asso_feat else assoc_feat, bbox_feat, cls_feat], dim=-1)
        gate_values = self.gate_network(concat_features)  # [B, 2]
        # if self.no_self_loc:
        #     gate_values[:, 0] = gate_values[:, 0] * 0 + 1
        # if self.no_self_cls:
        #     gate_values[:, 1] = gate_values[:, 1] * 0 + 1
        if self.no_self_loc:
            new_col = torch.ones_like(gate_values[:, 0])
            gate_values = torch.cat([new_col.unsqueeze(1), gate_values[:, 1:]], dim=1)
        if self.no_self_cls:
            new_col = torch.ones_like(gate_values[:, 1])
            gate_values = torch.cat([gate_values[:, :1], new_col.unsqueeze(1)], dim=1)

        if return_index_list is not None:
            intra_loc_list = [gate_values[index][0].cpu().item() for index in return_index_list]
            intra_sem_list = [gate_values[index][1].cpu().item() for index in return_index_list]
        # 3. 结合置信度调整门控值
        confidence_weights = torch.stack([bbox_conf, cls_conf], dim=1).squeeze(-1)  # [B, 2]
        gate_values = gate_values * confidence_weights * 0.5  # 使用0.5进一步降低辅助特征的影响

        # 4. 计算辅助特征的残差
        gated_bbox = bbox_feat * gate_values[:, 0:1]
        gated_cls = cls_feat * gate_values[:, 1:2]

        # 5. 残差方式融合特征
        # 将主特征和门控后的辅助特征拼接
        fusion_input = torch.cat([
            assoc_feat.detach() if self.if_detach_asso_feat else assoc_feat,
            gated_bbox,
            gated_cls
        ], dim=-1)

        # 计算残差
        residual = self.residual_layer(fusion_input)

        # 计算自适应融合比例
        adaptive_ratio = self.calculate_adaptive_fusion_ratio(assoc_conf)

        # 使用自适应比例进行特征融合
        if self.add_res:
            enhanced_feat = assoc_feat + residual * adaptive_ratio
        else:
            # print("no res (only used to debug)")
            enhanced_feat = residual * adaptive_ratio

        # 7. 归一化
        # enhanced_feat = self.norm(enhanced_feat)

        if return_index_list is not None:
            return enhanced_feat, intra_loc_list, intra_sem_list
        else:
            return enhanced_feat



@HEADS.register_module()
class OVTrackRoIHeadUncertainty(StandardRoIHead):
    def __init__(
            self,
            track_roi_extractor=None,
            track_head=None,
            track_train_cfg=None,
            cem_roi_extractor=None,
            cem_train_cfg=None,
            cem_head=None,
            finetune_track=False,
            kd_weight=256,
            fixed_lambda=None,
            prompt_path=None,
            fix_bg=False,
            ensemble=True,
            custom_classes=False,
            dynamic_rcnn_thre=True,
            only_validation_categories=False,
            only_test_categories=False,
            use_special_prompt=False,
            use_special_text_prompt=False,
            use_special_image_prompt=False,
            use_special_prompt_only_on_novel=False,
            prompt_word_list=["complete", "incomplete"],
            prompt_group_list=None,
            prompt_group_mean_way=True,
            self_train=False,
            only_self_train=False,
            self_train_rcnn=None,
            init_track_head_by_bbox_head=False,
            debug=False,
            bbox_counter = -1,
            cluster_num=-1,
            cluster_method='kmeans', # or dbscan
            two_stage_inference=False,
            simple_concate=True,
            prob_alpha=1.0,
            record_probs_info=False,
            after_softmax = False,
            prob_thres = 0.0,
            prob_dynamic = False,
            prob_dynamic_ratio = 0.1,
            spatial_loss=False,
            only_spatial_loss=False,
            top_k_spatial=10,
            iou_thres=0.95,
            spatial_learning_rate=0.01,
            use_cls_static_ratio=False,
            use_motion_static_ratio=False,
            motion_static_ratio=0.0,
            use_cyc_loss=False,
            cyc_loss_ratio=1.0,
            cyc_loss_start_iteration=999999,
            cls_static_ratio=0,
            feature_fusion_head=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)


        if track_head is not None:
            self.init_track_head(track_roi_extractor, track_head)

        if track_train_cfg is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_assigner_sampler()

        if cem_head is not None:
            self.init_cem_head(cem_roi_extractor, cem_head)
        else:
            self.cem_head = None

        if feature_fusion_head is not None:
            self.init_fusion_head(feature_fusion_head)

        if cem_train_cfg is not None:
            self.cem_train_cfg = cem_train_cfg
            self.init_cem_assigner_sampler()
        self.finetune_track = finetune_track

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.fix_bg = fix_bg
        self.custom_classes = custom_classes
        self.dynamic_rcnn_thre = dynamic_rcnn_thre
        self.iteration_counter = 0
        if custom_classes:
            self.fixed_lambda = 0.3

        print("fixed_lambda", fixed_lambda)
        print("prompt path", prompt_path)
        self.text_features_for_classes = []
        self.ensemble = ensemble
        print("ensemble:{}".format(self.ensemble))

        if custom_classes:
            self.CLASSES = text_input
            self.num_classes = len(text_input)
        else:
            self.num_classes = self.bbox_head.num_classes
            if self.num_classes == 8:
                self.CLASSES = BDD_CLASSES
            elif self.num_classes == 1203:
                self.CLASSES = LVIS_CLASSES
            elif self.num_classes == 1053:
                self.CLASSES = OVTB_CLASSES
            else:
                print("For custom classes, please set custom_classes=True")

        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = ""

        print("load:", save_path + f" exists: {osp.exists(save_path)}" '\n' + f"custom_classes: {custom_classes}")
        if osp.exists(save_path) and not custom_classes:
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).squeeze()[
                                                 : self.bbox_head.num_classes
                                                 ]
                self.text_features_for_classes = self.text_features_for_classes.to(
                    device
                )

            else:
                self.text_features_for_classes = (
                    torch.load(save_path).to(device).squeeze()
                )
                print(self.text_features_for_classes.shape)
        else:
            print("Custom target classes: ", self.CLASSES)
            clip_model, self.preprocess = clip.load("ViT-B/32", device)
            clip_model.eval()
            for child in clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            if os.path.exists(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')):
                clip_path = os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')
                self.text_features_for_classes = torch.load(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth'), device)
                print(f'Loading clip features from {clip_path}')
            else:
                for template in tqdm(template_list):
                    print(template)
                    text_features_for_classes = torch.cat(
                        [
                            clip_model.encode_text(
                                clip.tokenize(template.format(c)).to(device)
                            ).detach()
                            for c in self.CLASSES
                        ]
                    )
                    self.text_features_for_classes.append(
                        F.normalize(text_features_for_classes, dim=-1)
                    )

                self.text_features_for_classes = torch.stack(
                    self.text_features_for_classes
                ).mean(dim=0)

                self.text_features_for_classes = self.text_features_for_classes.float()
                self.text_features_for_classes = F.normalize(
                    self.text_features_for_classes, dim=-1
                )

        # TODO minimize the size of the matrix
        self.only_validation_categories = only_validation_categories
        self.only_test_categories = only_test_categories
        self.use_special_prompt = use_special_prompt
        self.use_special_text_prompt = use_special_text_prompt
        self.use_special_image_prompt = use_special_image_prompt
        self.use_special_prompt_only_on_novel = use_special_prompt_only_on_novel
        self.prompt_group_mean_way = prompt_group_mean_way
        self.self_train = self_train
        self.only_self_train = only_self_train
        self.self_train_rcnn = self_train_rcnn
        self.debug = debug
        self.bbox_counter = bbox_counter
        self.bbox_head_shared_conv_state_dict = self.bbox_head.shared_convs.state_dict()
        self.bbox_head_shread_fc_state_dict = self.bbox_head.shared_fcs.state_dict()
        # print('getting bbox head state dict.....')
        # self.track_head_conv_state_dict = self.track_head.convs.state_dict()
        # self.track_head_fc_state_dict = self.track_head.fcs.state_dict()
        self.init_track_head_by_bbox_head = init_track_head_by_bbox_head
        self.init_track_head_tag = False
        # only used to debug
        self.mean_positive_prob = []
        self.debug_list = []
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method
        self.two_stage_inference = two_stage_inference
        self.simple_concat = simple_concate
        self.prob_alpha = prob_alpha
        self.record_prob_info = record_probs_info
        self.after_softmax = after_softmax
        self.prob_thres = prob_thres
        self.prob_dynamic = prob_dynamic
        self.prob_dynamic_ratio = prob_dynamic_ratio
        self.result_save_path = ''
        self.spatial_loss = spatial_loss
        self.only_spatial_loss = only_spatial_loss
        self.top_k_spatial = top_k_spatial
        self.iou_thres = iou_thres
        self.spatial_learning_rate = spatial_learning_rate
        self.cls_static_ratio = cls_static_ratio
        self.use_cls_static_ratio = use_cls_static_ratio
        self.use_motion_static_ratio = use_motion_static_ratio
        self.motion_static_ratio = motion_static_ratio
        self.use_cyc_loss = use_cyc_loss
        self.cyc_loss_ratio = cyc_loss_ratio
        self.cyc_loss_start_iteration = cyc_loss_start_iteration
        print("prob_alpha:{}".format(self.prob_alpha))
        print("special_prompt:{}".format(use_special_prompt))
        print("prob_dynamic:{}".format(prob_dynamic))
        print("prob_dynamic_ratio:{}".format(prob_dynamic_ratio))
        print("prob_static_thres:{}".format(prob_thres))
        print("image prompt:{}".format(use_special_image_prompt))
        print("text prompt:{}".format(use_special_text_prompt))
        print("spatial loss:{}".format(spatial_loss))
        print("top k spatial:{}".format(top_k_spatial))
        print("top k iou thres:{}".format(iou_thres))
        print("spatial learning rate:{}".format(spatial_learning_rate))
        # self.empty_fc = nn.Linear(1,1)

        # adding motional fc layer
        # if self.use_motion_static_ratio:
        #     self.bbox_fc = nn.Linear(4, self.track_head.fc_embed.out_features).to(device)
        #     nn.init.xavier_uniform_(self.bbox_fc.weight)
        #     nn.init.constant_(self.bbox_fc.bias, 0)

        # judge whether we use validation or test dataset
        assert not (only_validation_categories == True and only_test_categories == True), f"Error category combination type {only_validation_categories}, {only_test_categories}"

        if only_validation_categories:
            indices = [col - 1 for col in tao_validation_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(validation_novel_index, device=device)
        if only_test_categories:
            indices = [col - 1 for col in tao_test_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(test_novel_index, device=device)


        if self.use_special_prompt:
            # assert len(prompt_word_list) == 2, f"Error: no matching format of {prompt_word_list}"
            # clip_model, self.preprocess = clip.load("ViT-B/32", device)
            # clip_model.eval()
            # for child in clip_model.children():
            #     for param in child.parameters():
            #         param.requires_grad = False
            # self.special_prompt = clip_model.encode_text(
            #     clip.tokenize(prompt_word_list).to(device)
            # ).float().detach()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

            self.prompt_group_list = prompt_group_list
            if self.prompt_group_list is not None:
                assert len(prompt_group_list) % 2 == 0, f"Error: no pair of prompt format of {prompt_group_list}"
                clip_model, self.preprocess = clip.load("ViT-B/32", device)
                clip_model.eval()
                for child in clip_model.children():
                    for param in child.parameters():
                        param.requires_grad = False
                self.special_prompt = clip_model.encode_text(
                    clip.tokenize(prompt_group_list).to(device)
                ).float().detach()
                self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1).view(-1, 2,
                                                                                                          512).permute( 0, 2, 1)

            # self.special_prompt = torch.load('/data1/clark/models/detpro/special_prompt/complete_incomplete.pth', map_location=device).float()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

        if  self.num_classes == 1053:
            self.text_features_for_classes = self.text_features_for_classes.float()
            self.text_features_for_classes = F.normalize(
                self.text_features_for_classes, dim=-1
            )
            print(f"OVTB normalized the text features")

        print(f"text features shape is {self.text_features_for_classes.shape}")

        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1, 512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
        self.projection = nn.Linear(1024, 512)
        # if self.ensemble:
        self.projection_for_image = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.projection_for_image.weight)
        nn.init.constant_(self.projection_for_image.bias, 0)

        if self.num_classes == 1230:
            self.base_label_ids = torch.tensor(lvis05_base, device=device)
            self.novel_label_ids = torch.tensor(lvis05_novel, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 1203:

            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()
        elif self.num_classes == 1053:
            self.base_label_ids = torch.tensor(ovtb_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(ovtb_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 8:
            self.base_label_ids = torch.tensor(bdd_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(bdd_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

    def print_self_filter_variables(self, type_list = [int, float, bool]):
        for name, value in self.__dict__.items():
            if type(value) in type_list:
                print(f"{name}: {value}")
    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_roi_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get("sampler", None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self
            )
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_track_head_by_bbox_head_method(self):
        self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
        self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
        print("Loaded track head convs and fcs from bbox head in init_track_head_by_bbox_head_method")
        self.init_track_head_tag = True

    def init_cem_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.cem_train_cfg.get("assigner", None):
            self.cem_roi_assigner = build_assigner(self.cem_train_cfg.assigner)
            self.cem_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.cem_share_assigner = True

        if self.cem_train_cfg.get("sampler", None):
            self.cem_roi_sampler = build_sampler(
                self.cem_train_cfg.sampler, context=self
            )
            self.cem_share_sampler = False
        else:
            self.cem_roi_sampler = self.bbox_sampler
            self.cem_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "track_head") and self.track_head is not None

    @property
    def with_cem(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "cem_head") and self.cem_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)
    def init_fusion_head(self, track_head):
        """Initialize ``track_head``"""
        self.fusion_head = build_head(track_head)

    def init_cem_head(self, cem_roi_extractor, cem_head):
        """Initialize ``track_head``"""
        if cem_roi_extractor is not None:
            self.cem_roi_extractor = build_roi_extractor(cem_roi_extractor)
            self.cem_share_extractor = False
        else:
            self.cem_share_extractor = True
            self.cem_roi_extractor = self.bbox_roi_extractor
        self.cem_head = build_head(cem_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()
        if self.with_cem:
            self.cem_head.init_weights()
            if not self.cem_share_extractor:
                self.cem_roi_extractor.init_weights()

    def forward_train(
            self,
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_x,
            ref_img_metas,
            ref_proposals,
            ref_gt_bboxes,
            ref_gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            ref_gt_bboxes_ignore=None,
            *args,
            **kwargs
    ):
        # if not self.finetune_track:
        #     losses = super().forward_train(
        #         x,
        #         img_metas,
        #         proposal_list,
        #         gt_bboxes,
        #         gt_labels,
        #         gt_bboxes_ignore,
        #         gt_masks,
        #         *args,
        #         **kwargs
        #     )
        # else:
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Init track shared conv head!")
            self.init_track_head_tag = True

        losses = {}

        num_imgs = len(img_metas)

        if self.self_train:
            # TODO finishing self train code here
            # score_thr is dynamic one
            pre_nms = None
            pre_max_per_img = None
            if self.self_train_rcnn.nms != self.test_cfg.nms:
                pre_nms = copy.deepcopy(self.test_cfg.nms)
                self.test_cfg.nms = self.self_train_rcnn.nms
            if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
                pre_max_per_img = self.test_cfg.max_per_img
                self.test_cfg.max_per_img = self.self_train_rcnn.max_per_img

            key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x, img_metas, proposal_list, False)
            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            ref_det_bboxes, ref_det_labels, ref_cem_feats, ref_track_feats = self.self_train_forward(ref_x, ref_img_metas, ref_proposals, False)
            num_det_per_img = tuple(det.shape[0] for det in ref_det_bboxes)
            ref_track_feats = ref_track_feats.split(num_det_per_img, 0)

            if self.debug:
                # only use to debug
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                if ref_gt_bboxes_ignore is None:
                    ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    key_positive_prob = (torch.sum(assign_result.max_overlaps > 0.5) / len(key_det_bboxes[i])).item()
                    key_sampling_results.append(key_positive_prob)
                    self.mean_positive_prob.append(key_positive_prob)

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[i],
                        ref_gt_bboxes[i],
                        ref_gt_bboxes_ignore[i],
                        ref_gt_labels[i],
                    )
                    ref_positive_prob = (torch.sum(ref_assign_result.max_overlaps > 0.5) / len(ref_det_bboxes[i])).item()
                    ref_sampling_results.append(ref_positive_prob)
                    self.mean_positive_prob.append(ref_positive_prob)
                print(f"{sum(key_sampling_results)/len(key_sampling_results)}, {sum(ref_sampling_results)/len(ref_sampling_results)}, {sum(self.mean_positive_prob)/len(self.mean_positive_prob)}")

            # calculating cycAss loss image by image
            loss_cyc = 0.0
            for i in range(len(img_metas)):
                features_list = [key_track_feats[i], ref_track_feats[i]]
                loss_cyc += self.track_head.self_loss(features_list)['cyc_loss']
            losses.update(dict(cyc_loss=loss_cyc/len(img_metas)))

            if pre_nms is not None:
                self.test_cfg.nms = pre_nms
            if pre_max_per_img is not None:
                self.test_cfg.max_per_img = pre_max_per_img

            if self.only_self_train:
                return losses


        if self.with_track or self.with_cem:
            if self.debug:
                with torch.no_grad():
                    key_det_bboxes, key_det_labels, [key_text_region_features, key_image_region_features,
                                                     key_score_matrix, key_det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    key_det_inds = [det_ind//296 for det_ind in key_det_inds]
                    ref_det_bboxes, ref_det_labels, [ref_text_region_features, ref_image_region_features,
                                                     ref_score_matrix, ref_det_inds] = self.simple_test_bboxes(
                        ref_x, ref_img_metas, ref_proposals, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    ref_det_inds = [det_ind//296 for det_ind in ref_det_inds]

                    before_add_gt_as_proposals = self.track_roi_sampler.add_gt_as_proposals
                    before_pos_iou_thr = self.track_roi_assigner.pos_iou_thr
                    self.track_roi_sampler.add_gt_as_proposals = False
                    self.track_roi_assigner.pos_iou_thr = 0.5

                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    if ref_gt_bboxes_ignore is None:
                        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[0][:,:4], gt_bboxes[0], gt_bboxes_ignore[0], gt_labels[0]
                    )
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        key_det_bboxes[0][:,:4],
                        gt_bboxes[0],
                        gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in x],
                    )

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_bboxes_ignore[0],
                        ref_gt_labels[0],
                    )
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in ref_x],
                    )
                    self.track_roi_sampler.add_gt_as_proposals = before_add_gt_as_proposals
                    self.track_roi_assigner.pos_iou_thr = before_pos_iou_thr

                    # save tmp result to analyse
                    # bbox, labels, region_embedding, cls_score
                    key_save_bbox = sampling_result.pos_bboxes.cpu()
                    key_save_labels = sampling_result.pos_gt_labels.cpu()
                    key_save_image_embedding = key_image_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_text_embedding = key_text_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_score_matrix = key_score_matrix[0][key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_list = [key_save_score_matrix, key_save_text_embedding, key_save_image_embedding, key_save_bbox, key_save_labels]
                    self.debug_list.append(key_save_list)

                    ref_save_bbox = ref_sampling_result.pos_bboxes.cpu()
                    ref_save_labels = ref_sampling_result.pos_gt_labels.cpu()
                    ref_save_image_embedding = ref_image_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_text_embedding = ref_text_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_score_matrix = ref_score_matrix[0][ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_list = [ref_save_score_matrix, ref_save_text_embedding, ref_save_image_embedding, ref_save_bbox, ref_save_labels]
                    self.debug_list.append(ref_save_list)
                    if len(self.debug_list) > 50:
                        torch.save(self.debug_list, '/home/clark/test/ovtrack_test/cluster_analyse/original_list.pth')
                        print('hello world')





            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]

            if self.with_track:
                key_feats = self._track_forward(x, key_bboxes)
                ref_bboxes = [res.bboxes for res in ref_sampling_results]
                ref_feats = self._track_forward(ref_x, ref_bboxes)

                match_feats = self.track_head.match(
                    key_feats, ref_feats, key_sampling_results, ref_sampling_results
                )
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results
                )
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                losses.update(loss_track)

        return losses

    def get_classification_head_features(self, x, rois):
        # text head
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings_text = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)

        # image head
        _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
        region_embeddings_image = self.projection_for_image(region_embeddings_image)
        region_embeddings_image = torch.nn.functional.normalize( region_embeddings_image, p=2, dim=1)

        return region_embeddings_text, region_embeddings_image


    def uncertainty_forward_train(
            self,
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_x,
            ref_img_metas,
            ref_proposals,
            ref_gt_bboxes,
            ref_gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            ref_gt_bboxes_ignore=None,
            *args,
            **kwargs
    ):
        self.iteration_counter += 1
        losses = {}

        num_imgs = len(img_metas)

        key_gt_match_indices_correct = gt_match_indices[0].cpu().tolist()
        ref_gt_match_indices_correct = kwargs['ref_gt_match_indices'][0].cpu().tolist()

        key_gt_label = gt_labels[0].cpu().tolist()
        ref_gt_label = ref_gt_labels[0].cpu().tolist()

        try:
            key_match_label = [ref_gt_label[index] == key_gt_label[i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
            ref_match_label = [key_gt_label[index] == ref_gt_label[i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]
        except Exception as e:
            print()


        # rank, world_size = get_dist_info()
        # content_line = list()
        # for img_meta in img_metas:
        #     content_line.append(f"{img_meta['filename']} ")
        # content_line.append("-1")
        # for img_meta in ref_img_metas:
        #     content_line.append(f"{img_meta['filename']} ")
        #
        # print(f"[{rank}/{world_size}] [5] train forward before" + " " + " ".join(content_line) )


        if self.self_train:
            # TODO finishing self train code here
            # score_thr is dynamic one
            pre_nms = None
            pre_max_per_img = None
            if self.self_train_rcnn.nms != self.test_cfg.nms:
                pre_nms = copy.deepcopy(self.test_cfg.nms)
                self.test_cfg.nms = self.self_train_rcnn.nms
            if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
                pre_max_per_img = self.test_cfg.max_per_img
                self.test_cfg.max_per_img = self.self_train_rcnn.max_per_img

            key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x, img_metas, proposal_list, False)
            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            ref_det_bboxes, ref_det_labels, ref_cem_feats, ref_track_feats = self.self_train_forward(ref_x, ref_img_metas, ref_proposals, False)
            num_det_per_img = tuple(det.shape[0] for det in ref_det_bboxes)
            ref_track_feats = ref_track_feats.split(num_det_per_img, 0)

            if self.debug:
                # only use to debug
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                if ref_gt_bboxes_ignore is None:
                    ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    key_positive_prob = (torch.sum(assign_result.max_overlaps > 0.5) / len(key_det_bboxes[i])).item()
                    key_sampling_results.append(key_positive_prob)
                    self.mean_positive_prob.append(key_positive_prob)

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[i],
                        ref_gt_bboxes[i],
                        ref_gt_bboxes_ignore[i],
                        ref_gt_labels[i],
                    )
                    ref_positive_prob = (torch.sum(ref_assign_result.max_overlaps > 0.5) / len(ref_det_bboxes[i])).item()
                    ref_sampling_results.append(ref_positive_prob)
                    self.mean_positive_prob.append(ref_positive_prob)
                print(f"{sum(key_sampling_results)/len(key_sampling_results)}, {sum(ref_sampling_results)/len(ref_sampling_results)}, {sum(self.mean_positive_prob)/len(self.mean_positive_prob)}")

            # calculating cycAss loss image by image
            loss_cyc = 0.0
            for i in range(len(img_metas)):
                features_list = [key_track_feats[i], ref_track_feats[i]]
                loss_cyc += self.track_head.self_loss(features_list)['cyc_loss']
            losses.update(dict(cyc_loss=loss_cyc/len(img_metas)))

            if pre_nms is not None:
                self.test_cfg.nms = pre_nms
            if pre_max_per_img is not None:
                self.test_cfg.max_per_img = pre_max_per_img

            if self.only_self_train:
                return losses


        if self.with_track or self.with_cem:
            if self.debug:
                with torch.no_grad():
                    key_det_bboxes, key_det_labels, [key_text_region_features, key_image_region_features,
                                                     key_score_matrix, key_det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    key_det_inds = [det_ind//296 for det_ind in key_det_inds]
                    ref_det_bboxes, ref_det_labels, [ref_text_region_features, ref_image_region_features,
                                                     ref_score_matrix, ref_det_inds] = self.simple_test_bboxes(
                        ref_x, ref_img_metas, ref_proposals, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    ref_det_inds = [det_ind//296 for det_ind in ref_det_inds]

                    before_add_gt_as_proposals = self.track_roi_sampler.add_gt_as_proposals
                    before_pos_iou_thr = self.track_roi_assigner.pos_iou_thr
                    self.track_roi_sampler.add_gt_as_proposals = False
                    self.track_roi_assigner.pos_iou_thr = 0.5

                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    if ref_gt_bboxes_ignore is None:
                        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[0][:,:4], gt_bboxes[0], gt_bboxes_ignore[0], gt_labels[0]
                    )
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        key_det_bboxes[0][:,:4],
                        gt_bboxes[0],
                        gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in x],
                    )

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_bboxes_ignore[0],
                        ref_gt_labels[0],
                    )
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in ref_x],
                    )
                    self.track_roi_sampler.add_gt_as_proposals = before_add_gt_as_proposals
                    self.track_roi_assigner.pos_iou_thr = before_pos_iou_thr

                    # save tmp result to analyse
                    # bbox, labels, region_embedding, cls_score
                    key_save_bbox = sampling_result.pos_bboxes.cpu()
                    key_save_labels = sampling_result.pos_gt_labels.cpu()
                    key_save_image_embedding = key_image_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_text_embedding = key_text_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_score_matrix = key_score_matrix[0][key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_list = [key_save_score_matrix, key_save_text_embedding, key_save_image_embedding, key_save_bbox, key_save_labels]
                    self.debug_list.append(key_save_list)

                    ref_save_bbox = ref_sampling_result.pos_bboxes.cpu()
                    ref_save_labels = ref_sampling_result.pos_gt_labels.cpu()
                    ref_save_image_embedding = ref_image_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_text_embedding = ref_text_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_score_matrix = ref_score_matrix[0][ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_list = [ref_save_score_matrix, ref_save_text_embedding, ref_save_image_embedding, ref_save_bbox, ref_save_labels]
                    self.debug_list.append(ref_save_list)
                    if len(self.debug_list) > 50:
                        torch.save(self.debug_list, '/home/clark/test/ovtrack_test/cluster_analyse/original_list.pth')
                        print('hello world')





            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]
            ref_bboxes = [res.bboxes for res in ref_sampling_results]

            key_bboxes_cat = torch.cat(key_bboxes, dim = 0)
            ref_bboxes_cat = torch.cat(ref_bboxes, dim = 0)

            # key_bboxes_feats = self.bbox_fc(key_bboxes_cat)
            # ref_bboxes_feats = self.bbox_fc(ref_bboxes_cat)

            key_feats = self._track_forward(x, key_bboxes)
            ref_feats = self._track_forward(ref_x, ref_bboxes)

            # ori loss to aux:
            ori_key_feats = key_feats
            ori_ref_feats = ref_feats




            #getting classification features
            key_proposal_num_list = [bbox.shape[0] for bbox in key_bboxes]
            ref_proposal_num_list = [bbox.shape[0] for bbox in ref_bboxes]
            key_rois = bbox2roi(key_bboxes)
            key_text_head_feats, key_img_head_feats = self.get_classification_head_features(x, key_rois)
            ref_rois = bbox2roi(ref_bboxes)
            ref_text_head_feats, ref_img_head_feats = self.get_classification_head_features(ref_x, ref_rois)

            key_cls_feats = (key_text_head_feats + key_img_head_feats)/2
            ref_cls_feats = (ref_text_head_feats + ref_img_head_feats)/2




            # fusing the features
            # deprecated way
            if self.use_cls_static_ratio and self.cls_static_ratio != 0:
                key_feats += key_cls_feats*self.cls_static_ratio
                ref_feats += ref_cls_feats*self.cls_static_ratio
            ## lasted fusion module
            # [[key_assoc], [key_bbox], [key_cls], [ref_assoc], [ref_bbox], [ref_cls]]
            uncertainty_list = [[], [], [], [], [], []]
            consistency_list = [[], []]
            if (self.use_cls_static_ratio and self.use_motion_static_ratio) or self.use_cyc_loss:
                # getting the consistency of each object
                key_feats_list = [key_feats, key_bboxes_cat, key_cls_feats]
                ref_feats_list = [ref_feats, ref_bboxes_cat, ref_cls_feats]
                for i in range(len(key_feats_list)):
                    key_feats_split_list = torch.split(key_feats_list[i], key_proposal_num_list, dim=0)
                    ref_feats_split_list = torch.split(ref_feats_list[i], ref_proposal_num_list, dim=0)
                    for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
                        consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
                            [key_feat, ref_feat])
                        consistency_reverse, consistency_diag_reverse = self.track_head.loss_cyc.get_pair_consistency(
                            [ref_feat, key_feat])
                        uncertainty_list[i].append(consistency_diag)
                        uncertainty_list[3 + i].append(consistency_diag_reverse)
                        if i == 0:
                            consistency_list[0].append(consistency)
                            consistency_list[1].append(consistency_reverse)
                if self.use_cls_static_ratio and self.use_motion_static_ratio:
                    if self.iteration_counter >= self.cyc_loss_start_iteration:
                        key_assoc_conf, key_bbox_conf, key_cls_conf, ref_assoc_conf, ref_bbox_conf, ref_cls_conf = [torch.cat(tensor_list).view(-1, 1).detach() for tensor_list in uncertainty_list]
                        # keep the original loss
                        match_feats = self.track_head.match(
                            ori_key_feats, ori_ref_feats, key_sampling_results, ref_sampling_results
                        )
                        asso_targets = self.track_head.get_track_targets(
                            gt_match_indices, key_sampling_results, ref_sampling_results
                        )

                        if isinstance(self.track_head.loss_track, MultiPosCrossEntropyLossUncertainty):
                            # generating the uncertainty consistency
                            updated_weights = []
                            # getting the element num of each frame
                            key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim=0)
                            ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim=0)
                            for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
                                consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
                                    [key_feat.detach(), ref_feat.detach()])
                                updated_weights.append(consistency_diag)
                            # asso_targets[1] = updated_weights
                            asso_targets = (asso_targets[0], updated_weights)
                            ori_loss_track = self.track_head.loss(*match_feats, *asso_targets)
                        else:
                            ori_loss_track = self.track_head.loss(*match_feats, *asso_targets)
                        ori_loss_track['ori_loss_track'] = ori_loss_track.pop('loss_track')
                        ori_loss_track['ori_loss_track_aux'] = ori_loss_track.pop('loss_track_aux')
                        losses.update(ori_loss_track)
                    else:
                        key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim=0)
                        ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim=0)
                        # for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
                        #     consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
                        #         [key_feat, ref_feat])
                        # generating the uncertainty list
                        key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)
                        key_bbox_conf = torch.zeros(key_feats.shape[0], 1, device=key_feats.device)
                        key_cls_conf = torch.zeros(key_feats.shape[0], 1, device=key_feats.device)
                        # key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5
                        # key_bbox_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5
                        # key_cls_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5

                        ref_assoc_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)
                        ref_bbox_conf = torch.zeros(ref_feats.shape[0], 1, device=key_feats.device)
                        ref_cls_conf = torch.zeros(ref_feats.shape[0], 1, device=key_feats.device)
                        # ref_assoc_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5
                        # ref_bbox_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5
                        # ref_cls_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5

                    key_feats = self.fusion_head(key_feats, key_bboxes_cat, key_cls_feats, key_assoc_conf, key_bbox_conf, key_cls_conf)
                    ref_feats = self.fusion_head(ref_feats, ref_bboxes_cat, ref_cls_feats, ref_assoc_conf, ref_bbox_conf, ref_cls_conf)






            # #
            # # only used to debug: check the classification resutls with the ground truth: passed √
            # #
            # input_one = x[0].new_ones(1)
            # bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            # bg_class_embedding = torch.nn.functional.normalize(
            #     bg_class_embedding, p=2, dim=1
            # )
            # text_features = torch.cat(
            #     [self.text_features_for_classes, bg_class_embedding], dim=0
            # )
            # cls_score_text = key_text_head_feats @ text_features.T
            # cls_score_text = cls_score_text / 0.007
            # cls_score_text = cls_score_text.softmax(dim=1)
            #
            # cls_score_image = key_img_head_feats @ text_features.T
            # cls_score_image = cls_score_image / 0.007
            # cls_score_image[:, -1] = -1e11
            # cls_score_image = cls_score_image.softmax(dim=1)
            #
            # a = 1 / 3
            #
            # cls_score = torch.where(
            #     self.novel_index,
            #     cls_score_image ** (1 - a) * cls_score_text ** a,
            #     cls_score_text ** (1 - a) * cls_score_image ** a,
            # )
            # max_probs, max_indices = torch.max(cls_score, dim=1)






            match_feats = self.track_head.match(
                key_feats, ref_feats, key_sampling_results, ref_sampling_results
            )
            asso_targets = self.track_head.get_track_targets(
                gt_match_indices, key_sampling_results, ref_sampling_results
            )

            if self.use_cyc_loss:
                # adding the gt consistency loss
                consistency_gt_list = create_diagonal_masks(asso_targets[0])
                consistency_matrix_list = consistency_list[0]

                ref_asso_targets = [matrix.T for matrix in asso_targets[0]]
                ref_consistency_gt_list = create_diagonal_masks(ref_asso_targets)
                ref_consistency_matrix_list = consistency_list[1]

                consistency_loss = 0.0
                for i in range(len(consistency_gt_list)):
                    consistency_gt = consistency_gt_list[i]
                    consistency_matrix = consistency_matrix_list[i]
                    consis_loss = nn.MSELoss()(consistency_matrix, consistency_gt.float())
                    consistency_loss += consis_loss
                    # ref part
                    ref_consistency_gt = ref_consistency_gt_list[i]
                    ref_consistency_matrix = ref_consistency_matrix_list[i]
                    ref_consis_loss = nn.MSELoss()(ref_consistency_matrix, ref_consistency_gt.float())
                    consistency_loss += ref_consis_loss

                consistency_loss /= (len(consistency_gt_list) * 2)

                losses.update({'loss_cyc' : consistency_loss * self.cyc_loss_ratio})

            if isinstance(self.track_head.loss_track, MultiPosCrossEntropyLossUncertainty):
                # generating the uncertainty consistency
                updated_weights = []
                # getting the element num of each frame
                key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim = 0)
                ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim = 0)
                for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
                    consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency([key_feat.detach(), ref_feat.detach()])
                    updated_weights.append(consistency_diag)
                # asso_targets[1] = updated_weights
                asso_targets = (asso_targets[0], updated_weights)
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                # consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency([key_embeds, ref_embeds])

                pass
            else:
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

            losses.update(loss_track)

            # if self.use_cyc_loss and self.iteration_counter >= self.cyc_loss_start_iteration:
            #     feats_list = [key_feats, ref_feats]
            #     loss_cyc = self.track_head.loss_cyc(feats_list)
            #     losses.update({"loss_cyc": loss_cyc})


            # print(f"[{rank}/{world_size}] [6] train forward return loss" + " " + " ".join(content_line))
            # with open('/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_after/tao_train_dataset/debug_seq_tao_base/debug/no_problem_list.txt', 'a') as f:
            #     f.write(" ".join(content_line) + "\n")

        return losses



    def forward_train_self(
            self,
            x_list,
            img_metas_list,
            proposal_list_for_x,
            *args,
            **kwargs
    ):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in forward_train_self")
            self.init_track_head_tag = True


        losses = {}

        num_imgs = len(img_metas_list[0])

        self_test_cfg = copy.deepcopy(self.test_cfg)
        # pre_nms = None
        # pre_max_per_img = None
        pre_dynamic_rcnn_thre = None
        # pre_score_thr = None
        if self.self_train_rcnn.nms != self.test_cfg.nms:
            # pre_nms = copy.deepcopy(self.test_cfg.nms)
            self_test_cfg.nms = self.self_train_rcnn.nms
        if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
            # pre_max_per_img = self.test_cfg.max_per_img
            self_test_cfg.max_per_img = self.self_train_rcnn.max_per_img
        if self.self_train_rcnn.dynamic_rcnn_thre != self.dynamic_rcnn_thre:
            pre_dynamic_rcnn_thre = self.dynamic_rcnn_thre
            self.dynamic_rcnn_thre = self.self_train_rcnn.dynamic_rcnn_thre
        if self.self_train_rcnn.score_thr != self.test_cfg.score_thr:
            # pre_score_thr = self.test_cfg.score_thr
            self_test_cfg.score_thr = self.self_train_rcnn.score_thr

        # det_bboxes_list, det_labels_list, cem_feats_list, track_feats_list = [], [], [], []
        key_feats_list = [[] for _ in range(num_imgs)]
        cluster_feats_list = [[] for _ in range(num_imgs)]

        spatial_xyxy_list = [[] for _ in range(num_imgs)]
        spatial_key_feats_list = [[] for _ in range(num_imgs)]

        # spatial_key_feats_list = [[] for _ in range(num_imgs)]
        # spatial_xyxy_list = [[] for _ in range(num_imgs)]

        #debug
        # e, i, x_list, img_metas_list, proposal_list_for_x = torch.load('/home/clark/test/ovtrack_error/self_train_forward_1705485087_5129771')

        for i in range(len(x_list)):
            outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False, test_config=self_test_cfg, cluster_num=self.cluster_num)
            #empty prediction
            while len(outputs) == 3:
                # print(f'No enough box in {img_metas_list[i][0]["filename"]}')
                tmp_self_cfg = copy.deepcopy(self_test_cfg)
                tmp_self_cfg.score_thr=self_test_cfg.score_thr/10
                outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False,
                                                  test_cfg=tmp_self_cfg)
                # return dict(cyc_loss=0*self.empty_fc(torch.tensor([[1.0]], device=key_det_bboxes[0].device)).sum()/1.0)
            if self.cluster_num != -1:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats, cls_score = outputs
            else:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = outputs
            # key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], False)
            # except Exception as e:
            #     torch.save([e, i, x_list, img_metas_list, proposal_list_for_x], f'/home/clark/test/ovtrack_error/self_train_forward_{str(time.time()).replace(".","_")}')
            #     # print(f'Error:{e}\ni:{i}\nx_list:{x_list[i]}\nimg_metas_list:{img_metas_list[i]}\nproposal_list_for_x:{proposal_list_for_x[i]} in self_train_forward')
            #     print(f'No enough box in {img_metas_list[i]["filename"]}')
            #     return dict(cyc_loss=x_list[0][0].sum()*0)

            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            for j in range(num_imgs):
                if key_track_feats[j].size(0) != 0:
                    key_feats_list[j].append(key_track_feats[j])
                if self.cluster_num != -1:
                    cluster_feats_list[j].append(cls_score[j])
                    if self.spatial_loss:
                        spatial_xyxy_list[j].append(key_det_bboxes[j])
        if self.bbox_counter != -1:
            for i in range(len(key_det_bboxes)):
                self.bbox_counter += key_det_bboxes[i].shape[0]
            print(self.bbox_counter)

        loss_cyc = 0.0
        loss_spatial = 0.0
        for i in range(num_imgs):
            # making cluster
            try:
                if self.cluster_num != -1:
                    num_det_per_img = tuple(cluster_feat.shape[0] for cluster_feat in cluster_feats_list[i])
                    cluster_feats = torch.cat(cluster_feats_list[i], dim=0)

                    # 2. 检查数据有效性
                    if torch.isnan(cluster_feats).any() or torch.isinf(cluster_feats).any():
                        raise ValueError("Invalid values in cluster_feats")

                        # 3. 确保数据点数量足够
                    if cluster_feats.shape[0] < self.cluster_num:
                        raise ValueError(
                            f"Not enough data points ({cluster_feats.shape[0]}) for {self.cluster_num} clusters")

                    X = cluster_feats.cpu()
                    if self.cluster_method == 'kmeans':
                        k_means_cluster_labesl = self.k_means_cluster(X, n_clusters=self.cluster_num)
                    elif self.cluster_method == 'dbscan':
                        k_means_cluster_labesl = self.dbscan_cluster(X)
                    else:
                        raise Exception(f'Unknown cluster type: {self.cluster_method}')
                    # max_cluster_index = max(k_means_cluster_labesl)
                    max_frequency_cluster_index = np.argmax(np.bincount(k_means_cluster_labesl))
                    cluster_list = torch.tensor(k_means_cluster_labesl, device=cluster_feats.device).split(num_det_per_img, 0)
                    if self.spatial_loss:
                        spatial_xyxy_list[i] = [
                            spatial_xyxy_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in
                            range(len(spatial_xyxy_list[i]))]
                        spatial_key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]
                    key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]


                if len(key_feats_list[i]) > 1:
                    loss_cyc += self.track_head.self_loss(key_feats_list[i])['cyc_loss']
                    if self.spatial_loss:
                        loss_spatial += self.track_head.self_spatial_loss(spatial_key_feats_list[i], spatial_xyxy_list[i], self.top_k_spatial, self.iou_thres, self.spatial_learning_rate)
            except Exception as e:
                # torch.save([e, i, key_feats_list], f'/home/clark/test/ovtrack_error/cyc_loss_{str(time.time()).replace(".","_")}')
                # print(f'All empty det Error:{e}\ni:{i}\nkey_feats_list:{key_feats_list[i]} in traverse cyc loss')
                # print(e, key_track_feats[0])
                if not self.spatial_loss:
                    return dict(cyc_loss=0.0*key_track_feats[0].sum())
                else:
                    if not self.only_spatial_loss:
                        return dict(cyc_loss=0.0 * key_track_feats[0].sum(), spatial_loss=0.0 * key_track_feats[0].sum())
                    else:
                        return dict(spatial_loss=0.0 * key_track_feats[0].sum())

                # for det_bbox in key_det_bboxes:
                #     if det_bbox.size(0) != 0:
                #         return dict(cyc_loss=det_bbox.sum() * 0)
                # return dict(cyc_loss=x_list[0][0].sum()*0)
        if type(loss_cyc) == float:
            loss_cyc = 0.0*key_track_feats[0].sum()
        if type(loss_spatial) == float:
            loss_spatial = 0.0*key_track_feats[0].sum()


        if not self.spatial_loss:
            losses.update(dict(cyc_loss=loss_cyc/num_imgs))
        else:
            if not self.only_spatial_loss:
                losses.update(dict(cyc_loss=loss_cyc/num_imgs, spatial_loss=loss_spatial/num_imgs))
            else:
                losses.update(dict(spatial_loss=loss_spatial/num_imgs))



        # if pre_nms is not None:
        #     self.test_cfg.nms = pre_nms
        # if pre_max_per_img is not None:
        #     self.test_cfg.max_per_img = pre_max_per_img
        if pre_dynamic_rcnn_thre is not None:
            self.dynamic_rcnn_thre = pre_dynamic_rcnn_thre
        # if pre_score_thr is not None:
        #     self.test_cfg.score_thr = pre_score_thr



        return losses

    def k_means_cluster(self, X, n_clusters=10, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels_pred = kmeans.labels_
        return labels_pred

    def dbscan_cluster(self, X, eps=0.04, min_samples=3):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = clustering.labels_
        return cluster_labels

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[: self.track_roi_extractor.num_inputs], rois
        )
        track_feats = self.track_head(track_feats) # 203, 256, 7, 7 -> 203, 256
        return track_feats

    def _cem_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        cem_feats = self.cem_roi_extractor(x[: self.cem_roi_extractor.num_inputs], rois)
        cem_feats = self.cem_head(cem_feats)

        return cem_feats

    def _clip_cem_forward(self, x, bboxes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return region_embeddings

    def simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in simple_test")
            self.init_track_head_tag = True

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None
        if self.record_prob_info:
            if det_bboxes.shape[1] == 5: # no prompt
                track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
            else:
                track_bboxes = det_bboxes[:, :-2] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
        else:
            track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                img_metas[0]["scale_factor"]
            ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats




    def uncertainty_simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in simple_test")
            self.init_track_head_tag = True

        try:
            det_bboxes, det_labels, (text_head_feats, img_head_feats, cls_score, det_inds) = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True
            )
        except Exception as e:
            torch.save([e, x, img_metas, proposal_list, self.test_cfg, rescale, True, True, f'det_bboxes, det_labels, (text_head_feats, img_head_feats, cls_score, det_inds) = self.simple_test_bboxes( x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True )'], f'/home/clark/test/ratio_out_of_memory_error.pth')
            print(f'Error:{e}\nx:{x}\nimg_metas:{img_metas}\nproposal_list:{proposal_list} in uncertainty_simple_test')
            import sys
            sys.exit(0)

        text_head_feats = text_head_feats[det_inds[0]//(cls_score[0].shape[1]-1)]
        img_head_feats = img_head_feats[det_inds[0]//(cls_score[0].shape[1]-1)]
        cls_feats = (text_head_feats + img_head_feats) / 2


        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None
        if self.record_prob_info:
            if det_bboxes.shape[1] == 5: # no prompt
                track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
            else:
                track_bboxes = det_bboxes[:, :-2] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
        else:
            track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                img_metas[0]["scale_factor"]
            ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        if self.use_cls_static_ratio and self.use_motion_static_ratio:
            key_bboxes_cat = det_bboxes[:, :-1]
            key_feats = track_feats
            key_cls_feats = cls_feats
            # key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # key_bbox_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # key_cls_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # track_feats = self.fusion_head(key_feats, key_bboxes_cat, key_cls_feats, key_assoc_conf, key_bbox_conf,
            #                              key_cls_conf)

            # if self.use_cls_static_ratio:
            #     track_feats += cls_feats * self.cls_static_ratio

            return det_bboxes, det_labels, key_cls_feats, track_feats
        else:
            return det_bboxes, det_labels, cem_feats, track_feats


    def self_train_forward(self, x, img_metas, proposal_list, rescale, test_config=None, cluster_num=-1, **kwargs):


        with torch.no_grad():

            if cluster_num != -1:
                if test_config is None:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                             score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True
                    )
                else:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                             score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale, return_features=True, return_inds=True
                    )
            else:
                if test_config is None:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale
                    )
                else:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale
                    )


        all_zero_boox_tag = True
        for det_bbox in det_bboxes:
            if det_bbox.size(0) != 0:
                all_zero_boox_tag = False
                break
        if all_zero_boox_tag:
            return det_bboxes, det_labels, None

            # if det_bboxes[0].size(0) == 0:
            # return det_bboxes[0], det_labels[0], None


        track_feats = self._track_forward(x, det_bboxes)
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, det_bboxes)
        else:
            cem_feats = None

        if cluster_num != -1:
            det_inds = [ind//(len(self.validation_index)-1) for ind in det_inds]
            score_matrix = [score_matrix[i][det_inds[i]] for i in range(len(score_matrix))]
            return det_bboxes, det_labels, cem_feats, track_feats, score_matrix
        else:
            return det_bboxes, det_labels, cem_feats, track_feats


    def simple_test_with_fixed_dets(
            self, x, det_bboxes, det_labels, img_metas, **kwargs
    ):

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        # #TODO localization using (0,1,2,3)
        # # if finest_scale == 0.1(layer_tag==True) means only use layer(3) as classification
        # layer_tag = False
        # if self.bbox_roi_extractor.finest_scale != 56:
        #     layer_tag = True
        #     self.bbox_roi_extractor.finest_scale=56
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        # bbox_pred = self.bbox_head(region_embeddings)
        # bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        #
        # #TODO classification using (3)
        # if layer_tag:
        #     self.bbox_roi_extractor.finest_scale=0.1
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)

        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

        # return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, return_features=False, return_inds=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            return_features: used to return region features and matching score matrix
            return_inds: if return the selected indices

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # st1 = time.time()
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        rois = bbox2roi(proposals)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(
                bg_class_embedding, p=2, dim=1
            )
            text_features = torch.cat(
                [self.text_features_for_classes, bg_class_embedding], dim=0
            )
        else:
            text_features = self.text_features_for_classes

        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)

        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(
                region_embeddings_image, p=2, dim=1
            )
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)

        if self.use_special_prompt:
            assert self.use_special_text_prompt == True or self.use_special_image_prompt == True, "no matching special prompt types"

            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            text_complete_result = region_embeddings @ self.special_prompt.float()
            text_complete_result = text_complete_result / 0.007
            text_complete_result = text_complete_result.softmax(dim=-1).permute(1, 0, 2)

            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            img_complete_result = region_embeddings_image @ self.special_prompt.float()
            img_complete_result = img_complete_result / 0.007
            img_complete_result = img_complete_result.softmax(dim=-1).permute(1, 0, 2)

            if self.prompt_group_mean_way:
                text_complete_result = text_complete_result.mean(dim=1)[:, 0].view(-1, 1)
                img_complete_result = img_complete_result.mean(dim=1)[:, 0].view(-1, 1)

            if self.use_special_text_prompt and self.use_special_image_prompt:
                complete_prob = (text_complete_result + img_complete_result)/2
            elif self.use_special_text_prompt:
                complete_prob = text_complete_result
                self.prob_thres = 0.005 # 0.13 0.15
                # print(f'only text reset the text prompt prob')
            elif self.use_special_image_prompt:
                complete_prob = img_complete_result
                self.prob_thres = 0.45 # 0.38
                # print(f'only img reset the img prompt prob')
            # complete_prob = complete_prob[:, 0].view(-1, 1)



            # old one
            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            # text_complete_result = text_complete_result / 0.007
            # text_complete_result = text_complete_result.softmax(dim=-1)
            #
            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            # img_complete_result = img_complete_result / 0.007
            # img_complete_result = img_complete_result.softmax(dim=-1)
            #
            # # complete_prob = (text_complete_result + img_complete_result) / 2
            # if self.use_special_text_prompt and self.use_special_image_prompt:
            #     complete_prob = (text_complete_result + img_complete_result)/2
            # elif self.use_special_text_prompt:
            #     complete_prob = text_complete_result
            # elif self.use_special_image_prompt:
            #     complete_prob = img_complete_result
            # complete_prob = complete_prob[:, 0].view(-1, 1)

        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = (
                        cls_score_image ** (1 - self.fixed_lambda)
                        * cls_score_text ** self.fixed_lambda
                )
            else:
                if self.only_validation_categories or self.only_test_categories:

                    cls_score = torch.where(
                        self.validation_index,
                        # self.novel_index, #TODO only test minimize list
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                        )
                else:
                    cls_score = torch.where(
                        self.novel_index,
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                        )


        else:
            cls_score = cls_score_text


        # TODO ONLY TEST SMALL DATASET HERE
        # indices = [col - 1 for col in tao_validation_used_cate_ids] + [cls_score.shape[1] - 1]
        # cls_score = cls_score[:, indices]
        # cls_score = torch.index_select(cls_score, 1, torch.tensor(indices, device=cls_score.device))


        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if self.use_special_prompt:
            complete_prob = complete_prob.split(num_proposals_per_img, 0)

            # cls_score = torch.where(
            #     self.validation_index,
            #     # self.novel_index, #TODO only test minimize list
            #     cls_score_image ** (1 - a) * cls_score_text ** a,
            #     cls_score_text ** (1 - a) * cls_score_image ** a,
            #     )

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        if self.dynamic_rcnn_thre:
            rcnn_test_cfg.score_thr = (1 / len(text_features)) * 1.001
            # print(rcnn_test_cfg.score_thr)



        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_inds = []
        if self.use_special_prompt:
            if self.two_stage_inference:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i],
                            cls_score[i]
                        )
                    else:
                        # cls_score_new = cls_score[i]*complete_prob[i]
                        cls_score_new = cls_score[i]

                        # first stage
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        det_index_row = det_index // (cls_score_new.shape[1] - 1)
                        det_label = det_label
                        base_index = self.validation_index[det_label] == False
                        one_stage_det_bbox = det_bbox[base_index]
                        one_stage_det_label = det_label[base_index]

                        # two stage
                        # simple concate way
                        if self.simple_concat:
                            two_stage_det_bbox = det_bbox[~base_index]
                            # two_stage_det_label = det_label[~base_index]

                            cls_score_new = cls_score_new[det_index_row][~base_index]*complete_prob[i][det_index_row][~base_index]
                            scores = F.softmax(cls_score_new, dim=-1)
                            new_two_stage_det_bbox, new_two_stage_det_labels = multiclass_nms(
                                two_stage_det_bbox[:,:-1], scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img
                            )

                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)
                        else:
                            indices_to_keep = torch.tensor(list(set(range(len(rois[i]))) - set(det_index_row[base_index].tolist())))
                            cls_score_new = cls_score[i]*complete_prob[i]
                            new_two_stage_det_bbox, new_two_stage_det_labels = self.bbox_head.get_bboxes(
                                rois[i][indices_to_keep],
                                # cls_score[i]*complete_prob[i],
                                cls_score_new[indices_to_keep],
                                bbox_pred[i][indices_to_keep],
                                img_shapes[i],
                                scale_factors[i],
                                rescale=rescale,
                                cfg=rcnn_test_cfg,
                            )
                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)

                        det_bboxes.append(det_bbox)
                        det_labels.append(det_label)




                    #     else:
                    #         det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                    #             rois[i],
                    #             # cls_score[i]*complete_prob[i],
                    #             cls_score_new,
                    #             bbox_pred[i],
                    #             img_shapes[i],
                    #             scale_factors[i],
                    #             rescale=rescale,
                    #             cfg=rcnn_test_cfg,
                    #             return_inds=return_inds
                    #         )
                    #
                    #     det_inds.append(det_index)
                    #
                    # det_bboxes.append(det_bbox)
                    # det_labels.append(det_label)
                    # if self.two_stage_inference:
                    #     for i in range(len(det_labels)):
                    #         # det_index = (det_inds[i] // (cls_score_new.shape[1]-1))
                    #         det_bbox = det_bboxes[i]
                    #         det_label = det_labels[i]
                    #         cls_score_new_norm = F.softmax(cls_score_new, dim=-1)
                    #         cls_score_ori = [cls_score_new_norm[i] for i in det_index.tolist()]
                    #         cls_score_ori_zip = [round(cls_score_ori[i][det_label[i]].item(),7) for i in range(len(det_label))]
                    #         det_cls_score_zip = [round(item.item(), 7) for item in det_bbox[:,-1]]
                    #
                    #
                    #
                    #
                    #         print('hello world')

            else:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i]*self.prob_alpha,
                            cls_score[i]
                        )
                    else:
                        if not self.after_softmax:
                            cls_score_new = cls_score[i]*complete_prob[i]*self.prob_alpha
                        else:
                            cls_score_new = cls_score[i]
                            self.bbox_head.after_softmax = self.after_softmax
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.prob_thres = self.prob_thres
                            self.bbox_head.prob_dynamic = self.prob_dynamic
                            self.bbox_head.prob_dynamic_ratio = self.prob_dynamic_ratio
                    if not return_inds and not self.record_prob_info:
                        det_bbox, det_label = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                        )
                    else:
                        if self.record_prob_info:
                            self.bbox_head.result_save_path = self.result_save_path
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.record_prob_info = self.record_prob_info
                            self.bbox_head.filename = img_metas[i]['ori_filename']
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        if self.record_prob_info:

                            # record the prob
                            det_index_row = det_index // (cls_score_new.shape[1] - 1)
                            probs = complete_prob[i][det_index_row]
                            # det_inds.append(det_index_row)
                            det_bbox = torch.cat([det_bbox[:, :4], probs, det_bbox[:, -1].reshape(-1, 1)], dim=-1)
                        else:
                            det_inds.append(det_index)

                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)

        else:
            for i in range(len(proposals)):
                if not return_inds:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                    )
                else:
                    det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        return_inds=return_inds
                    )
                    det_inds.append(det_index)

                det_bboxes.append(det_bbox)
                det_labels.append(det_label)

        if self.only_validation_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if self.only_test_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_test_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if not return_features or self.record_prob_info:
            return det_bboxes, det_labels
        else:
            assert return_features is True and return_inds is True, 'return features and inds do not match!'
            return det_bboxes, det_labels, [region_embeddings, region_embeddings_image, cls_score, det_inds]





@HEADS.register_module()
class OVTrackRoISLAck(StandardRoIHead):
    def __init__(
            self,
            track_roi_extractor=None,
            track_head=None,
            cls_head=None,
            loc_head=None,
            track_train_cfg=None,
            cem_roi_extractor=None,
            cem_train_cfg=None,
            cem_head=None,
            finetune_track=False,
            kd_weight=256,
            fixed_lambda=None,
            prompt_path=None,
            fix_bg=False,
            ensemble=True,
            custom_classes=False,
            dynamic_rcnn_thre=True,
            only_validation_categories=False,
            only_test_categories=False,
            use_special_prompt=False,
            use_special_text_prompt=False,
            use_special_image_prompt=False,
            use_special_prompt_only_on_novel=False,
            prompt_word_list=["complete", "incomplete"],
            prompt_group_list=None,
            prompt_group_mean_way=True,
            self_train=False,
            only_self_train=False,
            self_train_rcnn=None,
            init_track_head_by_bbox_head=False,
            debug=False,
            bbox_counter = -1,
            cluster_num=-1,
            cluster_method='kmeans', # or dbscan
            two_stage_inference=False,
            simple_concate=True,
            prob_alpha=1.0,
            record_probs_info=False,
            after_softmax = False,
            prob_thres = 0.0,
            prob_dynamic = False,
            prob_dynamic_ratio = 0.1,
            spatial_loss=False,
            only_spatial_loss=False,
            top_k_spatial=10,
            iou_thres=0.95,
            spatial_learning_rate=0.01,
            use_cls_static_ratio=False,
            use_motion_static_ratio=False,
            motion_static_ratio=0.0,
            use_cyc_loss=False,
            cyc_loss_ratio=1.0,
            cyc_loss_start_iteration=999999,
            cls_static_ratio=0,
            feature_fusion_head=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if track_head is not None:
            self.init_track_head(track_roi_extractor, track_head)
        # init cls head
        self.cls_head = FiveLayerMLP(512)
        # init loc head
        self.loc_head = FiveLayerMLP(4)
        self.stog = FeatureInteractionModule()
        self.stog.to(device)

        if track_train_cfg is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_assigner_sampler()

        if cem_head is not None:
            self.init_cem_head(cem_roi_extractor, cem_head)
        else:
            self.cem_head = None

        if feature_fusion_head is not None:
            self.init_fusion_head(feature_fusion_head)

        if cem_train_cfg is not None:
            self.cem_train_cfg = cem_train_cfg
            self.init_cem_assigner_sampler()
        self.finetune_track = finetune_track

        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.fix_bg = fix_bg
        self.custom_classes = custom_classes
        self.dynamic_rcnn_thre = dynamic_rcnn_thre
        self.iteration_counter = 0
        if custom_classes:
            self.fixed_lambda = 0.3

        print("fixed_lambda", fixed_lambda)
        print("prompt path", prompt_path)
        self.text_features_for_classes = []
        self.ensemble = ensemble
        print("ensemble:{}".format(self.ensemble))

        if custom_classes:
            self.CLASSES = text_input
            self.num_classes = len(text_input)
        else:
            self.num_classes = self.bbox_head.num_classes
            if self.num_classes == 8:
                self.CLASSES = BDD_CLASSES
            elif self.num_classes == 1203:
                self.CLASSES = LVIS_CLASSES
            elif self.num_classes == 1053:
                self.CLASSES = OVTB_CLASSES
            else:
                print("For custom classes, please set custom_classes=True")

        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = ""

        print("load:", save_path + f" exists: {osp.exists(save_path)}" '\n' + f"custom_classes: {custom_classes}")
        if osp.exists(save_path) and not custom_classes:
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).squeeze()[
                                                 : self.bbox_head.num_classes
                                                 ]
                self.text_features_for_classes = self.text_features_for_classes.to(
                    device
                )

            else:
                self.text_features_for_classes = (
                    torch.load(save_path).to(device).squeeze()
                )
                print(self.text_features_for_classes.shape)
        else:
            print("Custom target classes: ", self.CLASSES)
            clip_model, self.preprocess = clip.load("ViT-B/32", device)
            clip_model.eval()
            for child in clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            if os.path.exists(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')):
                clip_path = os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth')
                self.text_features_for_classes = torch.load(os.path.join('/data1/clark/models/ovtrack', f'{len(self.CLASSES)}_512_clip_class.pth'), device)
                print(f'Loading clip features from {clip_path}')
            else:
                for template in tqdm(template_list):
                    print(template)
                    text_features_for_classes = torch.cat(
                        [
                            clip_model.encode_text(
                                clip.tokenize(template.format(c)).to(device)
                            ).detach()
                            for c in self.CLASSES
                        ]
                    )
                    self.text_features_for_classes.append(
                        F.normalize(text_features_for_classes, dim=-1)
                    )

                self.text_features_for_classes = torch.stack(
                    self.text_features_for_classes
                ).mean(dim=0)

                self.text_features_for_classes = self.text_features_for_classes.float()
                self.text_features_for_classes = F.normalize(
                    self.text_features_for_classes, dim=-1
                )

        # TODO minimize the size of the matrix
        self.only_validation_categories = only_validation_categories
        self.only_test_categories = only_test_categories
        self.use_special_prompt = use_special_prompt
        self.use_special_text_prompt = use_special_text_prompt
        self.use_special_image_prompt = use_special_image_prompt
        self.use_special_prompt_only_on_novel = use_special_prompt_only_on_novel
        self.prompt_group_mean_way = prompt_group_mean_way
        self.self_train = self_train
        self.only_self_train = only_self_train
        self.self_train_rcnn = self_train_rcnn
        self.debug = debug
        self.bbox_counter = bbox_counter
        self.bbox_head_shared_conv_state_dict = self.bbox_head.shared_convs.state_dict()
        self.bbox_head_shread_fc_state_dict = self.bbox_head.shared_fcs.state_dict()
        # print('getting bbox head state dict.....')
        # self.track_head_conv_state_dict = self.track_head.convs.state_dict()
        # self.track_head_fc_state_dict = self.track_head.fcs.state_dict()
        self.init_track_head_by_bbox_head = init_track_head_by_bbox_head
        self.init_track_head_tag = False
        # only used to debug
        self.mean_positive_prob = []
        self.debug_list = []
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method
        self.two_stage_inference = two_stage_inference
        self.simple_concat = simple_concate
        self.prob_alpha = prob_alpha
        self.record_prob_info = record_probs_info
        self.after_softmax = after_softmax
        self.prob_thres = prob_thres
        self.prob_dynamic = prob_dynamic
        self.prob_dynamic_ratio = prob_dynamic_ratio
        self.result_save_path = ''
        self.spatial_loss = spatial_loss
        self.only_spatial_loss = only_spatial_loss
        self.top_k_spatial = top_k_spatial
        self.iou_thres = iou_thres
        self.spatial_learning_rate = spatial_learning_rate
        self.cls_static_ratio = cls_static_ratio
        self.use_cls_static_ratio = use_cls_static_ratio
        self.use_motion_static_ratio = use_motion_static_ratio
        self.motion_static_ratio = motion_static_ratio
        self.use_cyc_loss = use_cyc_loss
        self.cyc_loss_ratio = cyc_loss_ratio
        self.cyc_loss_start_iteration = cyc_loss_start_iteration
        print("prob_alpha:{}".format(self.prob_alpha))
        print("special_prompt:{}".format(use_special_prompt))
        print("prob_dynamic:{}".format(prob_dynamic))
        print("prob_dynamic_ratio:{}".format(prob_dynamic_ratio))
        print("prob_static_thres:{}".format(prob_thres))
        print("image prompt:{}".format(use_special_image_prompt))
        print("text prompt:{}".format(use_special_text_prompt))
        print("spatial loss:{}".format(spatial_loss))
        print("top k spatial:{}".format(top_k_spatial))
        print("top k iou thres:{}".format(iou_thres))
        print("spatial learning rate:{}".format(spatial_learning_rate))
        # self.empty_fc = nn.Linear(1,1)

        # adding motional fc layer
        # if self.use_motion_static_ratio:
        #     self.bbox_fc = nn.Linear(4, self.track_head.fc_embed.out_features).to(device)
        #     nn.init.xavier_uniform_(self.bbox_fc.weight)
        #     nn.init.constant_(self.bbox_fc.bias, 0)

        # judge whether we use validation or test dataset
        assert not (only_validation_categories == True and only_test_categories == True), f"Error category combination type {only_validation_categories}, {only_test_categories}"

        if only_validation_categories:
            indices = [col - 1 for col in tao_validation_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(validation_novel_index, device=device)
        if only_test_categories:
            indices = [col - 1 for col in tao_test_used_cate_ids]
            self.text_features_for_classes = self.text_features_for_classes[indices,:]
            # self.validation_index = torch.tensor([True for i in range(len(indices))] + [False], device=device)
            self.validation_index = torch.tensor(test_novel_index, device=device)


        if self.use_special_prompt:
            # assert len(prompt_word_list) == 2, f"Error: no matching format of {prompt_word_list}"
            # clip_model, self.preprocess = clip.load("ViT-B/32", device)
            # clip_model.eval()
            # for child in clip_model.children():
            #     for param in child.parameters():
            #         param.requires_grad = False
            # self.special_prompt = clip_model.encode_text(
            #     clip.tokenize(prompt_word_list).to(device)
            # ).float().detach()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

            self.prompt_group_list = prompt_group_list
            if self.prompt_group_list is not None:
                assert len(prompt_group_list) % 2 == 0, f"Error: no pair of prompt format of {prompt_group_list}"
                clip_model, self.preprocess = clip.load("ViT-B/32", device)
                clip_model.eval()
                for child in clip_model.children():
                    for param in child.parameters():
                        param.requires_grad = False
                self.special_prompt = clip_model.encode_text(
                    clip.tokenize(prompt_group_list).to(device)
                ).float().detach()
                self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1).view(-1, 2,
                                                                                                          512).permute( 0, 2, 1)

            # self.special_prompt = torch.load('/data1/clark/models/detpro/special_prompt/complete_incomplete.pth', map_location=device).float()
            # self.special_prompt = torch.nn.functional.normalize(self.special_prompt, p=2, dim=1)

        if  self.num_classes == 1053:
            self.text_features_for_classes = self.text_features_for_classes.float()
            self.text_features_for_classes = F.normalize(
                self.text_features_for_classes, dim=-1
            )
            print(f"OVTB normalized the text features")

        print(f"text features shape is {self.text_features_for_classes.shape}")

        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1, 512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
        self.projection = nn.Linear(1024, 512)
        # if self.ensemble:
        self.projection_for_image = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.projection_for_image.weight)
        nn.init.constant_(self.projection_for_image.bias, 0)

        if self.num_classes == 1230:
            self.base_label_ids = torch.tensor(lvis05_base, device=device)
            self.novel_label_ids = torch.tensor(lvis05_novel, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 1203:

            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()
        elif self.num_classes == 1053:
            self.base_label_ids = torch.tensor(ovtb_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(ovtb_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

        elif self.num_classes == 8:
            self.base_label_ids = torch.tensor(bdd_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(bdd_novel_label_ids, device=device)
            self.novel_index = F.pad(
                torch.bincount(self.novel_label_ids),
                (0, self.bbox_head.num_classes - self.novel_label_ids.max()),
            ).bool()

    def print_self_filter_variables(self, type_list = [int, float, bool]):
        for name, value in self.__dict__.items():
            if type(value) in type_list:
                print(f"{name}: {value}")
    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_roi_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get("sampler", None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self
            )
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_track_head_by_bbox_head_method(self):
        self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
        self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
        print("Loaded track head convs and fcs from bbox head in init_track_head_by_bbox_head_method")
        self.init_track_head_tag = True

    def init_cem_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.cem_train_cfg.get("assigner", None):
            self.cem_roi_assigner = build_assigner(self.cem_train_cfg.assigner)
            self.cem_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.cem_share_assigner = True

        if self.cem_train_cfg.get("sampler", None):
            self.cem_roi_sampler = build_sampler(
                self.cem_train_cfg.sampler, context=self
            )
            self.cem_share_sampler = False
        else:
            self.cem_roi_sampler = self.bbox_sampler
            self.cem_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "track_head") and self.track_head is not None

    @property
    def with_cem(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "cem_head") and self.cem_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)
    def init_cls_head(self, track_roi_extractor, cls_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.cls_head = build_head(cls_head)
    def init_loc_head(self, track_roi_extractor, loc_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.loc_head = build_head(loc_head)
    def init_fusion_head(self, track_head):
        """Initialize ``track_head``"""
        self.fusion_head = build_head(track_head)

    def init_cem_head(self, cem_roi_extractor, cem_head):
        """Initialize ``track_head``"""
        if cem_roi_extractor is not None:
            self.cem_roi_extractor = build_roi_extractor(cem_roi_extractor)
            self.cem_share_extractor = False
        else:
            self.cem_share_extractor = True
            self.cem_roi_extractor = self.bbox_roi_extractor
        self.cem_head = build_head(cem_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()
        if self.with_cem:
            self.cem_head.init_weights()
            if not self.cem_share_extractor:
                self.cem_roi_extractor.init_weights()

    def forward_train(
            self,
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_x,
            ref_img_metas,
            ref_proposals,
            ref_gt_bboxes,
            ref_gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            ref_gt_bboxes_ignore=None,
            *args,
            **kwargs
    ):
        # if not self.finetune_track:
        #     losses = super().forward_train(
        #         x,
        #         img_metas,
        #         proposal_list,
        #         gt_bboxes,
        #         gt_labels,
        #         gt_bboxes_ignore,
        #         gt_masks,
        #         *args,
        #         **kwargs
        #     )
        # else:
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Init track shared conv head!")
            self.init_track_head_tag = True

        losses = {}

        num_imgs = len(img_metas)

        if self.self_train:
            # TODO finishing self train code here
            # score_thr is dynamic one
            pre_nms = None
            pre_max_per_img = None
            if self.self_train_rcnn.nms != self.test_cfg.nms:
                pre_nms = copy.deepcopy(self.test_cfg.nms)
                self.test_cfg.nms = self.self_train_rcnn.nms
            if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
                pre_max_per_img = self.test_cfg.max_per_img
                self.test_cfg.max_per_img = self.self_train_rcnn.max_per_img

            key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x, img_metas, proposal_list, False)
            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            ref_det_bboxes, ref_det_labels, ref_cem_feats, ref_track_feats = self.self_train_forward(ref_x, ref_img_metas, ref_proposals, False)
            num_det_per_img = tuple(det.shape[0] for det in ref_det_bboxes)
            ref_track_feats = ref_track_feats.split(num_det_per_img, 0)

            if self.debug:
                # only use to debug
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                if ref_gt_bboxes_ignore is None:
                    ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    key_positive_prob = (torch.sum(assign_result.max_overlaps > 0.5) / len(key_det_bboxes[i])).item()
                    key_sampling_results.append(key_positive_prob)
                    self.mean_positive_prob.append(key_positive_prob)

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[i],
                        ref_gt_bboxes[i],
                        ref_gt_bboxes_ignore[i],
                        ref_gt_labels[i],
                    )
                    ref_positive_prob = (torch.sum(ref_assign_result.max_overlaps > 0.5) / len(ref_det_bboxes[i])).item()
                    ref_sampling_results.append(ref_positive_prob)
                    self.mean_positive_prob.append(ref_positive_prob)
                print(f"{sum(key_sampling_results)/len(key_sampling_results)}, {sum(ref_sampling_results)/len(ref_sampling_results)}, {sum(self.mean_positive_prob)/len(self.mean_positive_prob)}")

            # calculating cycAss loss image by image
            loss_cyc = 0.0
            for i in range(len(img_metas)):
                features_list = [key_track_feats[i], ref_track_feats[i]]
                loss_cyc += self.track_head.self_loss(features_list)['cyc_loss']
            losses.update(dict(cyc_loss=loss_cyc/len(img_metas)))

            if pre_nms is not None:
                self.test_cfg.nms = pre_nms
            if pre_max_per_img is not None:
                self.test_cfg.max_per_img = pre_max_per_img

            if self.only_self_train:
                return losses


        if self.with_track or self.with_cem:
            if self.debug:
                with torch.no_grad():
                    key_det_bboxes, key_det_labels, [key_text_region_features, key_image_region_features,
                                                     key_score_matrix, key_det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    key_det_inds = [det_ind//296 for det_ind in key_det_inds]
                    ref_det_bboxes, ref_det_labels, [ref_text_region_features, ref_image_region_features,
                                                     ref_score_matrix, ref_det_inds] = self.simple_test_bboxes(
                        ref_x, ref_img_metas, ref_proposals, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    ref_det_inds = [det_ind//296 for det_ind in ref_det_inds]

                    before_add_gt_as_proposals = self.track_roi_sampler.add_gt_as_proposals
                    before_pos_iou_thr = self.track_roi_assigner.pos_iou_thr
                    self.track_roi_sampler.add_gt_as_proposals = False
                    self.track_roi_assigner.pos_iou_thr = 0.5

                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    if ref_gt_bboxes_ignore is None:
                        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[0][:,:4], gt_bboxes[0], gt_bboxes_ignore[0], gt_labels[0]
                    )
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        key_det_bboxes[0][:,:4],
                        gt_bboxes[0],
                        gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in x],
                    )

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_bboxes_ignore[0],
                        ref_gt_labels[0],
                    )
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in ref_x],
                    )
                    self.track_roi_sampler.add_gt_as_proposals = before_add_gt_as_proposals
                    self.track_roi_assigner.pos_iou_thr = before_pos_iou_thr

                    # save tmp result to analyse
                    # bbox, labels, region_embedding, cls_score
                    key_save_bbox = sampling_result.pos_bboxes.cpu()
                    key_save_labels = sampling_result.pos_gt_labels.cpu()
                    key_save_image_embedding = key_image_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_text_embedding = key_text_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_score_matrix = key_score_matrix[0][key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_list = [key_save_score_matrix, key_save_text_embedding, key_save_image_embedding, key_save_bbox, key_save_labels]
                    self.debug_list.append(key_save_list)

                    ref_save_bbox = ref_sampling_result.pos_bboxes.cpu()
                    ref_save_labels = ref_sampling_result.pos_gt_labels.cpu()
                    ref_save_image_embedding = ref_image_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_text_embedding = ref_text_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_score_matrix = ref_score_matrix[0][ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_list = [ref_save_score_matrix, ref_save_text_embedding, ref_save_image_embedding, ref_save_bbox, ref_save_labels]
                    self.debug_list.append(ref_save_list)
                    if len(self.debug_list) > 50:
                        torch.save(self.debug_list, '/home/clark/test/ovtrack_test/cluster_analyse/original_list.pth')
                        print('hello world')





            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]

            if self.with_track:
                key_feats = self._track_forward(x, key_bboxes)
                ref_bboxes = [res.bboxes for res in ref_sampling_results]
                ref_feats = self._track_forward(ref_x, ref_bboxes)

                match_feats = self.track_head.match(
                    key_feats, ref_feats, key_sampling_results, ref_sampling_results
                )
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results
                )
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                losses.update(loss_track)

        return losses

    def get_classification_head_features(self, x, rois):
        # text head
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings_text = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)

        # image head
        _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
        region_embeddings_image = self.projection_for_image(region_embeddings_image)
        region_embeddings_image = torch.nn.functional.normalize( region_embeddings_image, p=2, dim=1)

        return region_embeddings_text, region_embeddings_image


    def uncertainty_forward_train(
            self,
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_x,
            ref_img_metas,
            ref_proposals,
            ref_gt_bboxes,
            ref_gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            ref_gt_bboxes_ignore=None,
            *args,
            **kwargs
    ):
        self.iteration_counter += 1
        losses = {}

        num_imgs = len(img_metas)

        key_gt_match_indices_correct = gt_match_indices[0].cpu().tolist()
        ref_gt_match_indices_correct = kwargs['ref_gt_match_indices'][0].cpu().tolist()

        key_gt_label = gt_labels[0].cpu().tolist()
        ref_gt_label = ref_gt_labels[0].cpu().tolist()

        try:
            key_match_label = [ref_gt_label[index] == key_gt_label[i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
            ref_match_label = [key_gt_label[index] == ref_gt_label[i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]
        except Exception as e:
            print()


        # rank, world_size = get_dist_info()
        # content_line = list()
        # for img_meta in img_metas:
        #     content_line.append(f"{img_meta['filename']} ")
        # content_line.append("-1")
        # for img_meta in ref_img_metas:
        #     content_line.append(f"{img_meta['filename']} ")
        #
        # print(f"[{rank}/{world_size}] [5] train forward before" + " " + " ".join(content_line) )


        if self.self_train:
            # TODO finishing self train code here
            # score_thr is dynamic one
            pre_nms = None
            pre_max_per_img = None
            if self.self_train_rcnn.nms != self.test_cfg.nms:
                pre_nms = copy.deepcopy(self.test_cfg.nms)
                self.test_cfg.nms = self.self_train_rcnn.nms
            if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
                pre_max_per_img = self.test_cfg.max_per_img
                self.test_cfg.max_per_img = self.self_train_rcnn.max_per_img

            key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x, img_metas, proposal_list, False)
            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            ref_det_bboxes, ref_det_labels, ref_cem_feats, ref_track_feats = self.self_train_forward(ref_x, ref_img_metas, ref_proposals, False)
            num_det_per_img = tuple(det.shape[0] for det in ref_det_bboxes)
            ref_track_feats = ref_track_feats.split(num_det_per_img, 0)

            if self.debug:
                # only use to debug
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                if ref_gt_bboxes_ignore is None:
                    ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                key_sampling_results, ref_sampling_results = [], []
                for i in range(num_imgs):
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                    )
                    key_positive_prob = (torch.sum(assign_result.max_overlaps > 0.5) / len(key_det_bboxes[i])).item()
                    key_sampling_results.append(key_positive_prob)
                    self.mean_positive_prob.append(key_positive_prob)

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[i],
                        ref_gt_bboxes[i],
                        ref_gt_bboxes_ignore[i],
                        ref_gt_labels[i],
                    )
                    ref_positive_prob = (torch.sum(ref_assign_result.max_overlaps > 0.5) / len(ref_det_bboxes[i])).item()
                    ref_sampling_results.append(ref_positive_prob)
                    self.mean_positive_prob.append(ref_positive_prob)
                print(f"{sum(key_sampling_results)/len(key_sampling_results)}, {sum(ref_sampling_results)/len(ref_sampling_results)}, {sum(self.mean_positive_prob)/len(self.mean_positive_prob)}")

            # calculating cycAss loss image by image
            loss_cyc = 0.0
            for i in range(len(img_metas)):
                features_list = [key_track_feats[i], ref_track_feats[i]]
                loss_cyc += self.track_head.self_loss(features_list)['cyc_loss']
            losses.update(dict(cyc_loss=loss_cyc/len(img_metas)))

            if pre_nms is not None:
                self.test_cfg.nms = pre_nms
            if pre_max_per_img is not None:
                self.test_cfg.max_per_img = pre_max_per_img

            if self.only_self_train:
                return losses


        if self.with_track or self.with_cem:
            if self.debug:
                with torch.no_grad():
                    key_det_bboxes, key_det_labels, [key_text_region_features, key_image_region_features,
                                                     key_score_matrix, key_det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    key_det_inds = [det_ind//296 for det_ind in key_det_inds]
                    ref_det_bboxes, ref_det_labels, [ref_text_region_features, ref_image_region_features,
                                                     ref_score_matrix, ref_det_inds] = self.simple_test_bboxes(
                        ref_x, ref_img_metas, ref_proposals, self.test_cfg, rescale=False, return_features=True, return_inds=True
                    )
                    ref_det_inds = [det_ind//296 for det_ind in ref_det_inds]

                    before_add_gt_as_proposals = self.track_roi_sampler.add_gt_as_proposals
                    before_pos_iou_thr = self.track_roi_assigner.pos_iou_thr
                    self.track_roi_sampler.add_gt_as_proposals = False
                    self.track_roi_assigner.pos_iou_thr = 0.5

                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    if ref_gt_bboxes_ignore is None:
                        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    assign_result = self.track_roi_assigner.assign(
                        key_det_bboxes[0][:,:4], gt_bboxes[0], gt_bboxes_ignore[0], gt_labels[0]
                    )
                    sampling_result = self.track_roi_sampler.sample(
                        assign_result,
                        key_det_bboxes[0][:,:4],
                        gt_bboxes[0],
                        gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in x],
                    )

                    ref_assign_result = self.track_roi_assigner.assign(
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_bboxes_ignore[0],
                        ref_gt_labels[0],
                    )
                    ref_sampling_result = self.track_roi_sampler.sample(
                        ref_assign_result,
                        ref_det_bboxes[0][:,:4],
                        ref_gt_bboxes[0],
                        ref_gt_labels[0],
                        feats=[lvl_feat[0][None] for lvl_feat in ref_x],
                    )
                    self.track_roi_sampler.add_gt_as_proposals = before_add_gt_as_proposals
                    self.track_roi_assigner.pos_iou_thr = before_pos_iou_thr

                    # save tmp result to analyse
                    # bbox, labels, region_embedding, cls_score
                    key_save_bbox = sampling_result.pos_bboxes.cpu()
                    key_save_labels = sampling_result.pos_gt_labels.cpu()
                    key_save_image_embedding = key_image_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_text_embedding = key_text_region_features[key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_score_matrix = key_score_matrix[0][key_det_inds][sampling_result.pos_inds].cpu()
                    key_save_list = [key_save_score_matrix, key_save_text_embedding, key_save_image_embedding, key_save_bbox, key_save_labels]
                    self.debug_list.append(key_save_list)

                    ref_save_bbox = ref_sampling_result.pos_bboxes.cpu()
                    ref_save_labels = ref_sampling_result.pos_gt_labels.cpu()
                    ref_save_image_embedding = ref_image_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_text_embedding = ref_text_region_features[ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_score_matrix = ref_score_matrix[0][ref_det_inds][ref_sampling_result.pos_inds].cpu()
                    ref_save_list = [ref_save_score_matrix, ref_save_text_embedding, ref_save_image_embedding, ref_save_bbox, ref_save_labels]
                    self.debug_list.append(ref_save_list)
                    if len(self.debug_list) > 50:
                        torch.save(self.debug_list, '/home/clark/test/ovtrack_test/cluster_analyse/original_list.pth')
                        print('hello world')





            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]
            ref_bboxes = [res.bboxes for res in ref_sampling_results]

            key_bboxes_cat = torch.cat(key_bboxes, dim = 0)
            ref_bboxes_cat = torch.cat(ref_bboxes, dim = 0)

            key_norm_bboxes_cat = normalize_detections(key_bboxes_cat)
            ref_norm_bboxes_cat = normalize_detections(ref_bboxes_cat)

            key_bboxes_feats = self.loc_head(key_norm_bboxes_cat)
            ref_bboxes_feats = self.loc_head(ref_norm_bboxes_cat)
            # key_bboxes_feats = self.bbox_fc(key_bboxes_cat)
            # ref_bboxes_feats = self.bbox_fc(ref_bboxes_cat)


            key_feats = self._track_forward(x, key_bboxes)
            ref_feats = self._track_forward(ref_x, ref_bboxes)

            # ori loss to aux:
            ori_key_feats = key_feats
            ori_ref_feats = ref_feats




            #getting classification features
            key_proposal_num_list = [bbox.shape[0] for bbox in key_bboxes]
            ref_proposal_num_list = [bbox.shape[0] for bbox in ref_bboxes]
            key_rois = bbox2roi(key_bboxes)
            key_text_head_feats, key_img_head_feats = self.get_classification_head_features(x, key_rois)
            ref_rois = bbox2roi(ref_bboxes)
            ref_text_head_feats, ref_img_head_feats = self.get_classification_head_features(ref_x, ref_rois)

            key_cls_feats = (key_text_head_feats + key_img_head_feats)/2
            ref_cls_feats = (ref_text_head_feats + ref_img_head_feats)/2

            key_cls_feats = self.cls_head(key_cls_feats)
            ref_cls_feats = self.cls_head(ref_cls_feats)



            # # fusing the features
            # # deprecated way
            # if self.use_cls_static_ratio and self.cls_static_ratio != 0:
            #     key_feats += key_cls_feats*self.cls_static_ratio
            #     ref_feats += ref_cls_feats*self.cls_static_ratio
            # ## lasted fusion module
            # # [[key_assoc], [key_bbox], [key_cls], [ref_assoc], [ref_bbox], [ref_cls]]
            # uncertainty_list = [[], [], [], [], [], []]
            # consistency_list = [[], []]
            # if (self.use_cls_static_ratio and self.use_motion_static_ratio) or self.use_cyc_loss:
            #     # getting the consistency of each object
            #     key_feats_list = [key_feats, key_bboxes_cat, key_cls_feats]
            #     ref_feats_list = [ref_feats, ref_bboxes_cat, ref_cls_feats]
            #     for i in range(len(key_feats_list)):
            #         key_feats_split_list = torch.split(key_feats_list[i], key_proposal_num_list, dim=0)
            #         ref_feats_split_list = torch.split(ref_feats_list[i], ref_proposal_num_list, dim=0)
            #         for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
            #             consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
            #                 [key_feat, ref_feat])
            #             consistency_reverse, consistency_diag_reverse = self.track_head.loss_cyc.get_pair_consistency(
            #                 [ref_feat, key_feat])
            #             uncertainty_list[i].append(consistency_diag)
            #             uncertainty_list[3 + i].append(consistency_diag_reverse)
            #             if i == 0:
            #                 consistency_list[0].append(consistency)
            #                 consistency_list[1].append(consistency_reverse)
            #     if self.use_cls_static_ratio and self.use_motion_static_ratio:
            #         if self.iteration_counter >= self.cyc_loss_start_iteration:
            #             key_assoc_conf, key_bbox_conf, key_cls_conf, ref_assoc_conf, ref_bbox_conf, ref_cls_conf = [torch.cat(tensor_list).view(-1, 1).detach() for tensor_list in uncertainty_list]
            #             # keep the original loss
            #             match_feats = self.track_head.match(
            #                 ori_key_feats, ori_ref_feats, key_sampling_results, ref_sampling_results
            #             )
            #             asso_targets = self.track_head.get_track_targets(
            #                 gt_match_indices, key_sampling_results, ref_sampling_results
            #             )
            #
            #             if isinstance(self.track_head.loss_track, MultiPosCrossEntropyLossUncertainty):
            #                 # generating the uncertainty consistency
            #                 updated_weights = []
            #                 # getting the element num of each frame
            #                 key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim=0)
            #                 ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim=0)
            #                 for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
            #                     consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
            #                         [key_feat.detach(), ref_feat.detach()])
            #                     updated_weights.append(consistency_diag)
            #                 # asso_targets[1] = updated_weights
            #                 asso_targets = (asso_targets[0], updated_weights)
            #                 ori_loss_track = self.track_head.loss(*match_feats, *asso_targets)
            #             else:
            #                 ori_loss_track = self.track_head.loss(*match_feats, *asso_targets)
            #             ori_loss_track['ori_loss_track'] = ori_loss_track.pop('loss_track')
            #             ori_loss_track['ori_loss_track_aux'] = ori_loss_track.pop('loss_track_aux')
            #             losses.update(ori_loss_track)
            #         else:
            #             key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim=0)
            #             ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim=0)
            #             # for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
            #             #     consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency(
            #             #         [key_feat, ref_feat])
            #             # generating the uncertainty list
            #             key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)
            #             key_bbox_conf = torch.zeros(key_feats.shape[0], 1, device=key_feats.device)
            #             key_cls_conf = torch.zeros(key_feats.shape[0], 1, device=key_feats.device)
            #             # key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5
            #             # key_bbox_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5
            #             # key_cls_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device)*0.5
            #
            #             ref_assoc_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)
            #             ref_bbox_conf = torch.zeros(ref_feats.shape[0], 1, device=key_feats.device)
            #             ref_cls_conf = torch.zeros(ref_feats.shape[0], 1, device=key_feats.device)
            #             # ref_assoc_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5
            #             # ref_bbox_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5
            #             # ref_cls_conf = torch.ones(ref_feats.shape[0], 1, device=key_feats.device)*0.5
            #
            #         key_feats = self.fusion_head(key_feats, key_bboxes_cat, key_cls_feats, key_assoc_conf, key_bbox_conf, key_cls_conf)
            #         ref_feats = self.fusion_head(ref_feats, ref_bboxes_cat, ref_cls_feats, ref_assoc_conf, ref_bbox_conf, ref_cls_conf)
            #
            #
            #
            #
            #

            # #
            # # only used to debug: check the classification resutls with the ground truth: passed √
            # #
            # input_one = x[0].new_ones(1)
            # bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            # bg_class_embedding = torch.nn.functional.normalize(
            #     bg_class_embedding, p=2, dim=1
            # )
            # text_features = torch.cat(
            #     [self.text_features_for_classes, bg_class_embedding], dim=0
            # )
            # cls_score_text = key_text_head_feats @ text_features.T
            # cls_score_text = cls_score_text / 0.007
            # cls_score_text = cls_score_text.softmax(dim=1)
            #
            # cls_score_image = key_img_head_feats @ text_features.T
            # cls_score_image = cls_score_image / 0.007
            # cls_score_image[:, -1] = -1e11
            # cls_score_image = cls_score_image.softmax(dim=1)
            #
            # a = 1 / 3
            #
            # cls_score = torch.where(
            #     self.novel_index,
            #     cls_score_image ** (1 - a) * cls_score_text ** a,
            #     cls_score_text ** (1 - a) * cls_score_image ** a,
            # )
            # max_probs, max_indices = torch.max(cls_score, dim=1)



            key_feats = key_feats + key_cls_feats + key_bboxes_feats
            ref_feats = ref_feats + ref_cls_feats + ref_bboxes_feats

            key_feats_split = torch.split(key_feats, key_proposal_num_list, dim = 0)
            ref_feats_split = torch.split(ref_feats, ref_proposal_num_list, dim = 0)


            key_feats_new, ref_feats_new = [], []

            for i in range(len(key_feats_split)):
                key_feats_new_inner, ref_feats_new_inner = self.stog(key_feats_split[i].unsqueeze(0), ref_feats_split[i].unsqueeze(0))
                # key_feats_new.append(key_feats_new_inner.squeeze() if len(key_feats_new_inner.shape) == 3 else key_feats_new_inner)
                key_feats_new.append(key_feats_new_inner.squeeze(dim=0))
                ref_feats_new.append(ref_feats_new_inner.squeeze(dim=0))

            # try:
                # print(len(key_feats_new), len(ref_feats_new), key_feats_new[0].shape, ref_feats_new[0].shape)
            key_feats, ref_feats = torch.cat(key_feats_new, dim=0), torch.cat(ref_feats_new, dim=0)
            # except:
            #     print('error')
            #     print(key_feats_new)
            #     print(ref_feats_new)
            #     exit()


            match_feats = self.track_head.match(
                key_feats, ref_feats, key_sampling_results, ref_sampling_results
            )
            asso_targets = self.track_head.get_track_targets(
                gt_match_indices, key_sampling_results, ref_sampling_results
            )
            # loss_track = self.track_head.loss(*match_feats, *asso_targets)

            loss_track = 0
            for i in range(len(asso_targets[0])):
                loss_track += sinkhorn_loss(match_feats[0][i], asso_targets[0][i])

            losses.update({'track_loss': loss_track/len(asso_targets[0])})


            # if self.use_cyc_loss:
            #     # adding the gt consistency loss
            #     consistency_gt_list = create_diagonal_masks(asso_targets[0])
            #     consistency_matrix_list = consistency_list[0]
            #
            #     ref_asso_targets = [matrix.T for matrix in asso_targets[0]]
            #     ref_consistency_gt_list = create_diagonal_masks(ref_asso_targets)
            #     ref_consistency_matrix_list = consistency_list[1]
            #
            #     consistency_loss = 0.0
            #     for i in range(len(consistency_gt_list)):
            #         consistency_gt = consistency_gt_list[i]
            #         consistency_matrix = consistency_matrix_list[i]
            #         consis_loss = nn.MSELoss()(consistency_matrix, consistency_gt.float())
            #         consistency_loss += consis_loss
            #         # ref part
            #         ref_consistency_gt = ref_consistency_gt_list[i]
            #         ref_consistency_matrix = ref_consistency_matrix_list[i]
            #         ref_consis_loss = nn.MSELoss()(ref_consistency_matrix, ref_consistency_gt.float())
            #         consistency_loss += ref_consis_loss
            #
            #     consistency_loss /= (len(consistency_gt_list) * 2)
            #
            #     losses.update({'loss_cyc' : consistency_loss * self.cyc_loss_ratio})

            # if isinstance(self.track_head.loss_track, MultiPosCrossEntropyLossUncertainty):
            #     # generating the uncertainty consistency
            #     updated_weights = []
            #     # getting the element num of each frame
            #     key_feats_split_list = torch.split(key_feats, key_proposal_num_list, dim = 0)
            #     ref_feats_split_list = torch.split(ref_feats, ref_proposal_num_list, dim = 0)
            #     for key_feat, ref_feat in zip(key_feats_split_list, ref_feats_split_list):
            #         consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency([key_feat.detach(), ref_feat.detach()])
            #         updated_weights.append(consistency_diag)
            #     # asso_targets[1] = updated_weights
            #     asso_targets = (asso_targets[0], updated_weights)
            #     loss_track = self.track_head.loss(*match_feats, *asso_targets)
            #
            #     # consistency, consistency_diag = self.track_head.loss_cyc.get_pair_consistency([key_embeds, ref_embeds])
            #
            #     pass
            # else:


            # if self.use_cyc_loss and self.iteration_counter >= self.cyc_loss_start_iteration:
            #     feats_list = [key_feats, ref_feats]
            #     loss_cyc = self.track_head.loss_cyc(feats_list)
            #     losses.update({"loss_cyc": loss_cyc})


            # print(f"[{rank}/{world_size}] [6] train forward return loss" + " " + " ".join(content_line))
            # with open('/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_after/tao_train_dataset/debug_seq_tao_base/debug/no_problem_list.txt', 'a') as f:
            #     f.write(" ".join(content_line) + "\n")

        return losses



    def forward_train_self(
            self,
            x_list,
            img_metas_list,
            proposal_list_for_x,
            *args,
            **kwargs
    ):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in forward_train_self")
            self.init_track_head_tag = True


        losses = {}

        num_imgs = len(img_metas_list[0])

        self_test_cfg = copy.deepcopy(self.test_cfg)
        # pre_nms = None
        # pre_max_per_img = None
        pre_dynamic_rcnn_thre = None
        # pre_score_thr = None
        if self.self_train_rcnn.nms != self.test_cfg.nms:
            # pre_nms = copy.deepcopy(self.test_cfg.nms)
            self_test_cfg.nms = self.self_train_rcnn.nms
        if self.self_train_rcnn.max_per_img != self.test_cfg.max_per_img:
            # pre_max_per_img = self.test_cfg.max_per_img
            self_test_cfg.max_per_img = self.self_train_rcnn.max_per_img
        if self.self_train_rcnn.dynamic_rcnn_thre != self.dynamic_rcnn_thre:
            pre_dynamic_rcnn_thre = self.dynamic_rcnn_thre
            self.dynamic_rcnn_thre = self.self_train_rcnn.dynamic_rcnn_thre
        if self.self_train_rcnn.score_thr != self.test_cfg.score_thr:
            # pre_score_thr = self.test_cfg.score_thr
            self_test_cfg.score_thr = self.self_train_rcnn.score_thr

        # det_bboxes_list, det_labels_list, cem_feats_list, track_feats_list = [], [], [], []
        key_feats_list = [[] for _ in range(num_imgs)]
        cluster_feats_list = [[] for _ in range(num_imgs)]

        spatial_xyxy_list = [[] for _ in range(num_imgs)]
        spatial_key_feats_list = [[] for _ in range(num_imgs)]

        # spatial_key_feats_list = [[] for _ in range(num_imgs)]
        # spatial_xyxy_list = [[] for _ in range(num_imgs)]

        #debug
        # e, i, x_list, img_metas_list, proposal_list_for_x = torch.load('/home/clark/test/ovtrack_error/self_train_forward_1705485087_5129771')

        for i in range(len(x_list)):
            outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False, test_config=self_test_cfg, cluster_num=self.cluster_num)
            #empty prediction
            while len(outputs) == 3:
                # print(f'No enough box in {img_metas_list[i][0]["filename"]}')
                tmp_self_cfg = copy.deepcopy(self_test_cfg)
                tmp_self_cfg.score_thr=self_test_cfg.score_thr/10
                outputs = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], rescale=False,
                                                  test_cfg=tmp_self_cfg)
                # return dict(cyc_loss=0*self.empty_fc(torch.tensor([[1.0]], device=key_det_bboxes[0].device)).sum()/1.0)
            if self.cluster_num != -1:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats, cls_score = outputs
            else:
                key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = outputs
            # key_det_bboxes, key_det_labels, key_cem_feats, key_track_feats = self.self_train_forward(x_list[i], img_metas_list[i], proposal_list_for_x[i], False)
            # except Exception as e:
            #     torch.save([e, i, x_list, img_metas_list, proposal_list_for_x], f'/home/clark/test/ovtrack_error/self_train_forward_{str(time.time()).replace(".","_")}')
            #     # print(f'Error:{e}\ni:{i}\nx_list:{x_list[i]}\nimg_metas_list:{img_metas_list[i]}\nproposal_list_for_x:{proposal_list_for_x[i]} in self_train_forward')
            #     print(f'No enough box in {img_metas_list[i]["filename"]}')
            #     return dict(cyc_loss=x_list[0][0].sum()*0)

            num_det_per_img = tuple(det.shape[0] for det in key_det_bboxes)
            key_track_feats = key_track_feats.split(num_det_per_img, 0)
            for j in range(num_imgs):
                if key_track_feats[j].size(0) != 0:
                    key_feats_list[j].append(key_track_feats[j])
                if self.cluster_num != -1:
                    cluster_feats_list[j].append(cls_score[j])
                    if self.spatial_loss:
                        spatial_xyxy_list[j].append(key_det_bboxes[j])
        if self.bbox_counter != -1:
            for i in range(len(key_det_bboxes)):
                self.bbox_counter += key_det_bboxes[i].shape[0]
            print(self.bbox_counter)

        loss_cyc = 0.0
        loss_spatial = 0.0
        for i in range(num_imgs):
            # making cluster
            try:
                if self.cluster_num != -1:
                    num_det_per_img = tuple(cluster_feat.shape[0] for cluster_feat in cluster_feats_list[i])
                    cluster_feats = torch.cat(cluster_feats_list[i], dim=0)

                    # 2. 检查数据有效性
                    if torch.isnan(cluster_feats).any() or torch.isinf(cluster_feats).any():
                        raise ValueError("Invalid values in cluster_feats")

                        # 3. 确保数据点数量足够
                    if cluster_feats.shape[0] < self.cluster_num:
                        raise ValueError(
                            f"Not enough data points ({cluster_feats.shape[0]}) for {self.cluster_num} clusters")

                    X = cluster_feats.cpu()
                    if self.cluster_method == 'kmeans':
                        k_means_cluster_labesl = self.k_means_cluster(X, n_clusters=self.cluster_num)
                    elif self.cluster_method == 'dbscan':
                        k_means_cluster_labesl = self.dbscan_cluster(X)
                    else:
                        raise Exception(f'Unknown cluster type: {self.cluster_method}')
                    # max_cluster_index = max(k_means_cluster_labesl)
                    max_frequency_cluster_index = np.argmax(np.bincount(k_means_cluster_labesl))
                    cluster_list = torch.tensor(k_means_cluster_labesl, device=cluster_feats.device).split(num_det_per_img, 0)
                    if self.spatial_loss:
                        spatial_xyxy_list[i] = [
                            spatial_xyxy_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in
                            range(len(spatial_xyxy_list[i]))]
                        spatial_key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]
                    key_feats_list[i] = [key_feats_list[i][index][cluster_list[index] == max_frequency_cluster_index] for index in range(len(key_feats_list[i]))]


                if len(key_feats_list[i]) > 1:
                    loss_cyc += self.track_head.self_loss(key_feats_list[i])['cyc_loss']
                    if self.spatial_loss:
                        loss_spatial += self.track_head.self_spatial_loss(spatial_key_feats_list[i], spatial_xyxy_list[i], self.top_k_spatial, self.iou_thres, self.spatial_learning_rate)
            except Exception as e:
                # torch.save([e, i, key_feats_list], f'/home/clark/test/ovtrack_error/cyc_loss_{str(time.time()).replace(".","_")}')
                # print(f'All empty det Error:{e}\ni:{i}\nkey_feats_list:{key_feats_list[i]} in traverse cyc loss')
                # print(e, key_track_feats[0])
                if not self.spatial_loss:
                    return dict(cyc_loss=0.0*key_track_feats[0].sum())
                else:
                    if not self.only_spatial_loss:
                        return dict(cyc_loss=0.0 * key_track_feats[0].sum(), spatial_loss=0.0 * key_track_feats[0].sum())
                    else:
                        return dict(spatial_loss=0.0 * key_track_feats[0].sum())

                # for det_bbox in key_det_bboxes:
                #     if det_bbox.size(0) != 0:
                #         return dict(cyc_loss=det_bbox.sum() * 0)
                # return dict(cyc_loss=x_list[0][0].sum()*0)
        if type(loss_cyc) == float:
            loss_cyc = 0.0*key_track_feats[0].sum()
        if type(loss_spatial) == float:
            loss_spatial = 0.0*key_track_feats[0].sum()


        if not self.spatial_loss:
            losses.update(dict(cyc_loss=loss_cyc/num_imgs))
        else:
            if not self.only_spatial_loss:
                losses.update(dict(cyc_loss=loss_cyc/num_imgs, spatial_loss=loss_spatial/num_imgs))
            else:
                losses.update(dict(spatial_loss=loss_spatial/num_imgs))



        # if pre_nms is not None:
        #     self.test_cfg.nms = pre_nms
        # if pre_max_per_img is not None:
        #     self.test_cfg.max_per_img = pre_max_per_img
        if pre_dynamic_rcnn_thre is not None:
            self.dynamic_rcnn_thre = pre_dynamic_rcnn_thre
        # if pre_score_thr is not None:
        #     self.test_cfg.score_thr = pre_score_thr



        return losses

    def k_means_cluster(self, X, n_clusters=10, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        labels_pred = kmeans.labels_
        return labels_pred

    def dbscan_cluster(self, X, eps=0.04, min_samples=3):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = clustering.labels_
        return cluster_labels

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[: self.track_roi_extractor.num_inputs], rois
        )
        track_feats = self.track_head(track_feats) # 203, 256, 7, 7 -> 203, 256
        return track_feats

    def _cem_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        cem_feats = self.cem_roi_extractor(x[: self.cem_roi_extractor.num_inputs], rois)
        cem_feats = self.cem_head(cem_feats)

        return cem_feats

    def _clip_cem_forward(self, x, bboxes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(bboxes)
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return region_embeddings

    def simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in simple_test")
            self.init_track_head_tag = True

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None
        if self.record_prob_info:
            if det_bboxes.shape[1] == 5: # no prompt
                track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
            else:
                track_bboxes = det_bboxes[:, :-2] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
        else:
            track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                img_metas[0]["scale_factor"]
            ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats




    def uncertainty_simple_test(self, x, img_metas, proposal_list, rescale, **kwargs):
        if self.init_track_head_by_bbox_head and self.init_track_head_tag == False:
            self.track_head.convs.load_state_dict(self.bbox_head.shared_convs.state_dict())
            self.track_head.fcs.load_state_dict(self.bbox_head.shared_fcs.state_dict())
            print("Loaded track head convs and fcs from bbox head in simple_test")
            self.init_track_head_tag = True

        try:
            det_bboxes, det_labels, (text_head_feats, img_head_feats, cls_score, det_inds) = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True
            )
        except Exception as e:
            torch.save([e, x, img_metas, proposal_list, self.test_cfg, rescale, True, True, f'det_bboxes, det_labels, (text_head_feats, img_head_feats, cls_score, det_inds) = self.simple_test_bboxes( x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True )'], f'/home/clark/test/ratio_out_of_memory_error.pth')
            print(f'Error:{e}\nx:{x}\nimg_metas:{img_metas}\nproposal_list:{proposal_list} in uncertainty_simple_test')
            import sys
            sys.exit(0)

        text_head_feats = text_head_feats[det_inds[0]//(cls_score[0].shape[1]-1)]
        img_head_feats = img_head_feats[det_inds[0]//(cls_score[0].shape[1]-1)]
        cls_feats = (text_head_feats + img_head_feats) / 2


        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None
        if self.record_prob_info:
            if det_bboxes.shape[1] == 5: # no prompt
                track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
            else:
                track_bboxes = det_bboxes[:, :-2] * torch.tensor(
                    img_metas[0]["scale_factor"]
                ).to(det_bboxes.device)
        else:
            track_bboxes = det_bboxes[:, :-1] * torch.tensor(
                img_metas[0]["scale_factor"]
            ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        key_norm_bboxes_cat = normalize_detections(det_bboxes)
        key_bboxes_feats = self.loc_head(key_norm_bboxes_cat)

        key_cls_feats = self.cls_head(cls_feats)

        key_feats = track_feats + key_cls_feats + key_bboxes_feats

        key_feats_split = [key_feats]

        key_feats_new, ref_feats_new = [], []

        for i in range(len(key_feats_split)):
            key_feats_new_inner, ref_feats_new_inner = self.stog(key_feats_split[i].unsqueeze(0),
                                                                 key_feats_split[i].unsqueeze(0))
            # key_feats_new.append(key_feats_new_inner.squeeze() if len(key_feats_new_inner.shape) == 3 else key_feats_new_inner)
            key_feats_new.append(key_feats_new_inner.squeeze(dim=0))
            ref_feats_new.append(ref_feats_new_inner.squeeze(dim=0))

        key_feats, ref_feats = torch.cat(key_feats_new, dim=0), torch.cat(ref_feats_new, dim=0)

        track_feats = key_feats

        if self.use_cls_static_ratio and self.use_motion_static_ratio:
            key_bboxes_cat = det_bboxes[:, :-1]
            key_feats = track_feats
            key_cls_feats = cls_feats
            # key_assoc_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # key_bbox_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # key_cls_conf = torch.ones(key_feats.shape[0], 1, device=key_feats.device) * 0.5
            # track_feats = self.fusion_head(key_feats, key_bboxes_cat, key_cls_feats, key_assoc_conf, key_bbox_conf,
            #                              key_cls_conf)

            # if self.use_cls_static_ratio:
            #     track_feats += cls_feats * self.cls_static_ratio

            return det_bboxes, det_labels, key_cls_feats, track_feats
        else:
            return det_bboxes, det_labels, cem_feats, track_feats


    def self_train_forward(self, x, img_metas, proposal_list, rescale, test_config=None, cluster_num=-1, **kwargs):


        with torch.no_grad():

            if cluster_num != -1:
                if test_config is None:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                             score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale, return_features=True, return_inds=True
                    )
                else:
                    det_bboxes, det_labels, [text_region_features, image_region_features,
                                             score_matrix, det_inds] = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale, return_features=True, return_inds=True
                    )
            else:
                if test_config is None:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, self.test_cfg, rescale=rescale
                    )
                else:
                    det_bboxes, det_labels = self.simple_test_bboxes(
                        x, img_metas, proposal_list, test_config, rescale=rescale
                    )


        all_zero_boox_tag = True
        for det_bbox in det_bboxes:
            if det_bbox.size(0) != 0:
                all_zero_boox_tag = False
                break
        if all_zero_boox_tag:
            return det_bboxes, det_labels, None

            # if det_bboxes[0].size(0) == 0:
            # return det_bboxes[0], det_labels[0], None


        track_feats = self._track_forward(x, det_bboxes)
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, det_bboxes)
        else:
            cem_feats = None

        if cluster_num != -1:
            det_inds = [ind//(len(self.validation_index)-1) for ind in det_inds]
            score_matrix = [score_matrix[i][det_inds[i]] for i in range(len(score_matrix))]
            return det_bboxes, det_labels, cem_feats, track_feats, score_matrix
        else:
            return det_bboxes, det_labels, cem_feats, track_feats


    def simple_test_with_fixed_dets(
            self, x, det_bboxes, det_labels, img_metas, **kwargs
    ):

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)

        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        # #TODO localization using (0,1,2,3)
        # # if finest_scale == 0.1(layer_tag==True) means only use layer(3) as classification
        # layer_tag = False
        # if self.bbox_roi_extractor.finest_scale != 56:
        #     layer_tag = True
        #     self.bbox_roi_extractor.finest_scale=56
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        # bbox_pred = self.bbox_head(region_embeddings)
        # bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        #
        # #TODO classification using (3)
        # if layer_tag:
        #     self.bbox_roi_extractor.finest_scale=0.1
        # bbox_feats = self.bbox_roi_extractor(
        #     x[: self.bbox_roi_extractor.num_inputs], rois
        # )
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        # region_embeddings = self.bbox_head.forward_embedding(bbox_feats)

        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

        # return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, return_features=False, return_inds=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            return_features: used to return region features and matching score matrix
            return_inds: if return the selected indices

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # st1 = time.time()
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        rois = bbox2roi(proposals)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(
                bg_class_embedding, p=2, dim=1
            )
            text_features = torch.cat(
                [self.text_features_for_classes, bg_class_embedding], dim=0
            )
        else:
            text_features = self.text_features_for_classes

        cls_score_text = region_embeddings @ text_features.T
        cls_score_text = cls_score_text / 0.007
        cls_score_text = cls_score_text.softmax(dim=1)

        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(
                region_embeddings_image, p=2, dim=1
            )
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            cls_score_image[:, -1] = -1e11
            cls_score_image = cls_score_image.softmax(dim=1)

        if self.use_special_prompt:
            assert self.use_special_text_prompt == True or self.use_special_image_prompt == True, "no matching special prompt types"

            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            text_complete_result = region_embeddings @ self.special_prompt.float()
            text_complete_result = text_complete_result / 0.007
            text_complete_result = text_complete_result.softmax(dim=-1).permute(1, 0, 2)

            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            img_complete_result = region_embeddings_image @ self.special_prompt.float()
            img_complete_result = img_complete_result / 0.007
            img_complete_result = img_complete_result.softmax(dim=-1).permute(1, 0, 2)

            if self.prompt_group_mean_way:
                text_complete_result = text_complete_result.mean(dim=1)[:, 0].view(-1, 1)
                img_complete_result = img_complete_result.mean(dim=1)[:, 0].view(-1, 1)

            if self.use_special_text_prompt and self.use_special_image_prompt:
                complete_prob = (text_complete_result + img_complete_result)/2
            elif self.use_special_text_prompt:
                complete_prob = text_complete_result
                self.prob_thres = 0.005 # 0.13 0.15
                # print(f'only text reset the text prompt prob')
            elif self.use_special_image_prompt:
                complete_prob = img_complete_result
                self.prob_thres = 0.45 # 0.38
                # print(f'only img reset the img prompt prob')
            # complete_prob = complete_prob[:, 0].view(-1, 1)



            # old one
            # text_complete_result = region_embeddings @ self.special_prompt.float().T
            # text_complete_result = text_complete_result / 0.007
            # text_complete_result = text_complete_result.softmax(dim=-1)
            #
            # img_complete_result = region_embeddings_image @ self.special_prompt.float().T
            # img_complete_result = img_complete_result / 0.007
            # img_complete_result = img_complete_result.softmax(dim=-1)
            #
            # # complete_prob = (text_complete_result + img_complete_result) / 2
            # if self.use_special_text_prompt and self.use_special_image_prompt:
            #     complete_prob = (text_complete_result + img_complete_result)/2
            # elif self.use_special_text_prompt:
            #     complete_prob = text_complete_result
            # elif self.use_special_image_prompt:
            #     complete_prob = img_complete_result
            # complete_prob = complete_prob[:, 0].view(-1, 1)

        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = (
                        cls_score_image ** (1 - self.fixed_lambda)
                        * cls_score_text ** self.fixed_lambda
                )
            else:
                if self.only_validation_categories or self.only_test_categories:

                    cls_score = torch.where(
                        self.validation_index,
                        # self.novel_index, #TODO only test minimize list
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                        )
                else:
                    cls_score = torch.where(
                        self.novel_index,
                        cls_score_image ** (1 - a) * cls_score_text ** a,
                        cls_score_text ** (1 - a) * cls_score_image ** a,
                        )


        else:
            cls_score = cls_score_text


        # TODO ONLY TEST SMALL DATASET HERE
        # indices = [col - 1 for col in tao_validation_used_cate_ids] + [cls_score.shape[1] - 1]
        # cls_score = cls_score[:, indices]
        # cls_score = torch.index_select(cls_score, 1, torch.tensor(indices, device=cls_score.device))


        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if self.use_special_prompt:
            complete_prob = complete_prob.split(num_proposals_per_img, 0)

            # cls_score = torch.where(
            #     self.validation_index,
            #     # self.novel_index, #TODO only test minimize list
            #     cls_score_image ** (1 - a) * cls_score_text ** a,
            #     cls_score_text ** (1 - a) * cls_score_image ** a,
            #     )

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        if self.dynamic_rcnn_thre:
            rcnn_test_cfg.score_thr = (1 / len(text_features)) * 1.001
            # print(rcnn_test_cfg.score_thr)



        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_inds = []
        if self.use_special_prompt:
            if self.two_stage_inference:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i],
                            cls_score[i]
                        )
                    else:
                        # cls_score_new = cls_score[i]*complete_prob[i]
                        cls_score_new = cls_score[i]

                        # first stage
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        det_index_row = det_index // (cls_score_new.shape[1] - 1)
                        det_label = det_label
                        base_index = self.validation_index[det_label] == False
                        one_stage_det_bbox = det_bbox[base_index]
                        one_stage_det_label = det_label[base_index]

                        # two stage
                        # simple concate way
                        if self.simple_concat:
                            two_stage_det_bbox = det_bbox[~base_index]
                            # two_stage_det_label = det_label[~base_index]

                            cls_score_new = cls_score_new[det_index_row][~base_index]*complete_prob[i][det_index_row][~base_index]
                            scores = F.softmax(cls_score_new, dim=-1)
                            new_two_stage_det_bbox, new_two_stage_det_labels = multiclass_nms(
                                two_stage_det_bbox[:,:-1], scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img
                            )

                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)
                        else:
                            indices_to_keep = torch.tensor(list(set(range(len(rois[i]))) - set(det_index_row[base_index].tolist())))
                            cls_score_new = cls_score[i]*complete_prob[i]
                            new_two_stage_det_bbox, new_two_stage_det_labels = self.bbox_head.get_bboxes(
                                rois[i][indices_to_keep],
                                # cls_score[i]*complete_prob[i],
                                cls_score_new[indices_to_keep],
                                bbox_pred[i][indices_to_keep],
                                img_shapes[i],
                                scale_factors[i],
                                rescale=rescale,
                                cfg=rcnn_test_cfg,
                            )
                            det_bbox = torch.cat((one_stage_det_bbox, new_two_stage_det_bbox), dim=0)
                            det_label = torch.cat((one_stage_det_label, new_two_stage_det_labels), dim=0)

                        det_bboxes.append(det_bbox)
                        det_labels.append(det_label)




                    #     else:
                    #         det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                    #             rois[i],
                    #             # cls_score[i]*complete_prob[i],
                    #             cls_score_new,
                    #             bbox_pred[i],
                    #             img_shapes[i],
                    #             scale_factors[i],
                    #             rescale=rescale,
                    #             cfg=rcnn_test_cfg,
                    #             return_inds=return_inds
                    #         )
                    #
                    #     det_inds.append(det_index)
                    #
                    # det_bboxes.append(det_bbox)
                    # det_labels.append(det_label)
                    # if self.two_stage_inference:
                    #     for i in range(len(det_labels)):
                    #         # det_index = (det_inds[i] // (cls_score_new.shape[1]-1))
                    #         det_bbox = det_bboxes[i]
                    #         det_label = det_labels[i]
                    #         cls_score_new_norm = F.softmax(cls_score_new, dim=-1)
                    #         cls_score_ori = [cls_score_new_norm[i] for i in det_index.tolist()]
                    #         cls_score_ori_zip = [round(cls_score_ori[i][det_label[i]].item(),7) for i in range(len(det_label))]
                    #         det_cls_score_zip = [round(item.item(), 7) for item in det_bbox[:,-1]]
                    #
                    #
                    #
                    #
                    #         print('hello world')

            else:
                for i in range(len(proposals)):
                    if self.use_special_prompt_only_on_novel:
                        cls_score_new = torch.where(
                            self.validation_index,
                            cls_score[i]*complete_prob[i]*self.prob_alpha,
                            cls_score[i]
                        )
                    else:
                        if not self.after_softmax:
                            cls_score_new = cls_score[i]*complete_prob[i]*self.prob_alpha
                        else:
                            cls_score_new = cls_score[i]
                            self.bbox_head.after_softmax = self.after_softmax
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.prob_thres = self.prob_thres
                            self.bbox_head.prob_dynamic = self.prob_dynamic
                            self.bbox_head.prob_dynamic_ratio = self.prob_dynamic_ratio
                    if not return_inds and not self.record_prob_info:
                        det_bbox, det_label = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                        )
                    else:
                        if self.record_prob_info:
                            self.bbox_head.result_save_path = self.result_save_path
                            self.bbox_head.complete_prob = complete_prob[i]
                            self.bbox_head.record_prob_info = self.record_prob_info
                            self.bbox_head.filename = img_metas[i]['ori_filename']
                        det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                            rois[i],
                            # cls_score[i]*complete_prob[i],
                            cls_score_new,
                            bbox_pred[i],
                            img_shapes[i],
                            scale_factors[i],
                            rescale=rescale,
                            cfg=rcnn_test_cfg,
                            return_inds=True
                        )
                        if self.record_prob_info:

                            # record the prob
                            det_index_row = det_index // (cls_score_new.shape[1] - 1)
                            probs = complete_prob[i][det_index_row]
                            # det_inds.append(det_index_row)
                            det_bbox = torch.cat([det_bbox[:, :4], probs, det_bbox[:, -1].reshape(-1, 1)], dim=-1)
                        else:
                            det_inds.append(det_index)

                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)

        else:
            for i in range(len(proposals)):
                if not return_inds:
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                    )
                else:
                    det_bbox, det_label, det_index = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_pred[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg,
                        return_inds=return_inds
                    )
                    det_inds.append(det_index)

                det_bboxes.append(det_bbox)
                det_labels.append(det_label)

        if self.only_validation_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if self.only_test_categories:
            # recover the indices to the position
            # det_labels = [torch.tensor([tao_validation_used_cate_ids[index.item()+1-1]-1 for index in det_labels[0]], device=det_labels[0].device)]
            det_labels = [torch.tensor([tao_test_used_cate_ids[index.item()]-1 for index in det_labels[i]], device=det_labels[0].device) for i in range(len(det_labels))]

        if not return_features or self.record_prob_info:
            return det_bboxes, det_labels
        else:
            assert return_features is True and return_inds is True, 'return features and inds do not match!'
            return det_bboxes, det_labels, [region_embeddings, region_embeddings_image, cls_score, det_inds]




def create_diagonal_masks(assignment_matrices):
    """
    Args:
        assignment_matrices: list of tensors, 每个tensor形状为(n, m)，包含0和1

    Returns:
        list of tensors, 每个tensor形状为(n, n)，对角线根据assignment决定是0还是1
    """
    result = []

    for assign_mat in assignment_matrices:
        n = assign_mat.size(0)  # 直接从assignment matrix获取行数

        # 获取每一行是否有1的标志（如果有1则为True）
        has_assignment = (assign_mat.sum(dim=1) > 0)  # shape: (n,)

        # 创建对角矩阵
        diagonal_mask = torch.zeros((n, n), device=assign_mat.device, dtype=assign_mat.dtype)
        diagonal_mask.fill_diagonal_(1)

        # 使用外积将has_assignment转换为完整的mask
        diagonal_mask = diagonal_mask * has_assignment.unsqueeze(1)

        result.append(diagonal_mask)

    return result


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class FiveLayerMLP(nn.Module):
    def __init__(self, input_dim):
        super(FiveLayerMLP, self).__init__()

        # 定义层：输入层到第一层，第一层到第二层，第二层到第三层，第三层到第四层，最后一层输出
        self.layer1 = nn.Linear(input_dim, 256)  # 第一层
        self.layer2 = nn.Linear(256, 256)  # 第二层
        self.layer3 = nn.Linear(256, 256)  # 第三层
        self.layer4 = nn.Linear(256, 256)  # 第四层
        self.layer5 = nn.Linear(256, 256)  # 最后一层（输出层）

        # 定义 GroupNorm 层
        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.group_norm3 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.group_norm4 = nn.GroupNorm(num_groups=32, num_channels=256)

        # ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        x = self.group_norm1(x)
        x = self.relu(x)

        # 第二层
        x = self.layer2(x)
        x = self.group_norm2(x)
        x = self.relu(x)

        # 第三层
        x = self.layer3(x)
        x = self.group_norm3(x)
        x = self.relu(x)

        # 第四层
        x = self.layer4(x)
        x = self.group_norm4(x)
        x = self.relu(x)

        # 第五层（输出层，不使用激活函数）
        x = self.layer5(x)

        return x

def normalize_detections(detections, W=1333, H=800):
    """
    对检测结果进行归一化处理。

    参数：
    detections: tensor，形状为 (N, 4)，包含 xmin, ymin, xmax, ymax
    W: 图像的宽度（默认为 1333）
    H: 图像的高度（默认为 800）

    返回：
    normalized: tensor，归一化后的检测结果，形状为 (N, 4)
    """
    # 计算缩放因子 s
    s = 0.7 * max(H, W)

    # 解包检测结果
    xmin = detections[:, 0]
    ymin = detections[:, 1]
    xmax = detections[:, 2]
    ymax = detections[:, 3]

    # 计算归一化
    normalized = torch.stack((
        xmin / s - W / (2 * s),  # 修正归一化计算
        ymin / s - H / (2 * s),  # 修正归一化计算
        (xmax - xmin) / s,        # 宽度
        (ymax - ymin) / s         # 高度
    ), dim=1)

    return normalized


# STOG

class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads

        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, k=None, v=None, is_self_attn=True):
        if k is None: k = x
        if v is None: v = x

        B, N, C = x.shape

        # 如果是自注意力，直接使用x
        q = self.query_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous()
        out = out.view(B, N, C)
        out = self.out_proj(out)

        return out


class FeatureInteractionModule(nn.Module):
    def __init__(self, feat_dim=256, num_layers=4, num_heads=4):
        super().__init__()

        self.num_layers = num_layers
        self.self_attns = nn.ModuleList([
            MultiHeadAttention(feat_dim, num_heads) for _ in range(num_layers)
        ])
        self.cross_attns = nn.ModuleList([
            MultiHeadAttention(feat_dim, num_heads) for _ in range(num_layers)
        ])

        # MLP用于特征精炼
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, feat_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, key_feats, ref_feats):
        # 确保输入维度正确
        B, N1, C = key_feats.shape
        _, N2, _ = ref_feats.shape

        # 拷贝输入，避免原地修改
        key_output = key_feats.clone()
        ref_output = ref_feats.clone()

        for i in range(self.num_layers):
            # 自注意力
            key_self_attn = self.self_attns[i](key_output)
            ref_self_attn = self.self_attns[i](ref_output)

            # 交叉注意力
            key_cross_attn = self.cross_attns[i](key_output, ref_output, ref_output)
            ref_cross_attn = self.cross_attns[i](ref_output, key_output, key_output)

            # 特征融合：原始特征 + 注意力输出
            key_fused = torch.cat([key_output, key_cross_attn], dim=-1)
            ref_fused = torch.cat([ref_output, ref_cross_attn], dim=-1)

            # MLP精炼
            key_output = self.mlps[i](key_fused)
            ref_output = self.mlps[i](ref_fused)

        return key_output, ref_output


# def sinkhorn_loss(sim_matrix, gt_matrix):
#     """
#     Compute Sinkhorn loss for matching between two sets of objects
#
#     Args:
#     - sim_matrix (torch.Tensor): Similarity matrix of shape (n1, n2)
#     - gt_matrix (torch.Tensor): Ground truth assignment matrix of shape (n1, n2)
#
#     Returns:
#     - torch.Tensor: Scalar Sinkhorn loss value
#     """
#     # 对相似度矩阵进行softmax归一化
#     S_ij = F.softmax(sim_matrix, dim=-1)
#
#     # 计算Sinkhorn损失 - 参考公式 L_sinkhorn = -∑(T_ij * log(S_ij))
#     loss = -torch.sum(gt_matrix * torch.log(S_ij + 1e-10))
#
#     return loss


def sinkhorn_loss(sim_matrix, gt_matrix, epsilon=1e-10):
    """
    Compute Sinkhorn loss with improved numerical stability

    Args:
    - sim_matrix (torch.Tensor): Similarity matrix of shape (n1, n2)
    - gt_matrix (torch.Tensor): Ground truth assignment matrix of shape (n1, n2)
    - epsilon (float): Small constant to prevent log(0)

    Returns:
    - torch.Tensor: Scalar Sinkhorn loss value
    """
    # 检查输入矩阵的有效性
    if torch.isnan(sim_matrix).any() or torch.isnan(gt_matrix).any():
        raise ValueError("Input matrices contain NaN values")

    # 对相似度矩阵进行softmax归一化
    # 使用log-sum-exp技巧提高数值稳定性
    S_ij = F.softmax(sim_matrix, dim=-1)

    # 安全的对数计算
    safe_log = lambda x: torch.log(torch.clamp(x, min=epsilon))

    # 计算Sinkhorn损失 - 参考公式 L_sinkhorn = -∑(T_ij * log(S_ij))
    loss = -torch.sum(gt_matrix * safe_log(S_ij))

    # 检查损失是否为有效值
    if torch.isnan(loss):
        print("Warning: Loss computation resulted in NaN")
        # 可以选择返回一个默认值或引发异常
        return torch.tensor(0.0, device=sim_matrix.device)

    return loss

# def sinkhorn_loss(sim_matrix, gt_matrix, num_iterations=100):
#     """
#     Compute Sinkhorn loss with explicit iteration control
#
#     Args:
#     - sim_matrix (torch.Tensor): Similarity matrix of shape (n1, n2)
#     - gt_matrix (torch.Tensor): Ground truth assignment matrix of shape (n1, n2)
#     - num_iterations (int): Number of Sinkhorn iterations
#     - tau (float): Temperature parameter for scaling
#
#     Returns:
#     - torch.Tensor: Scalar Sinkhorn loss value
#     """
#
#     # 初始化双重变量
#     log_alpha = torch.zeros_like(sim_matrix)
#     log_beta = torch.zeros_like(sim_matrix)
#
#     # Sinkhorn迭代
#     for _ in range(num_iterations):
#         # 行归一化
#         log_alpha = sim_matrix - log_beta.logsumexp(dim=1, keepdim=True)
#
#         # 列归一化
#         log_beta = sim_matrix - log_alpha.logsumexp(dim=0, keepdim=True)
#
#     # 计算最终的软分配矩阵
#     S_ij = torch.exp(log_alpha + log_beta + sim_matrix)
#
#     # 计算损失
#     loss = -torch.sum(gt_matrix * torch.log(S_ij + 1e-10))
#
#     return loss