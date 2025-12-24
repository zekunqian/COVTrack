import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL.Image
import torch
from mmdet.core import bbox2result
from mmdet.models import TwoStageDetector
from PIL import Image

from ovtrack.core import imshow_tracks, restore_result, track2result
from ..builder import MODELS, build_tracker, build_motion


@MODELS.register_module()
class OVTrack(TwoStageDetector):
    def __init__(
        self,
        tracker=None,
        use_kf_motion=False,
        freeze_detector=False,
        save_dets=False,
        pickle_path=None,
        use_fixed_dets=False,
        method="ovtrack-teta",
        uncertainty_ovtrack=False,
        slack_ovtrack=False,
        *args,
        **kwargs
    ):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker
        # self.motion_cfg = motion
        self.use_kf_motion = use_kf_motion
        self.method = method
        print(self.method)
        self.freeze_detector = freeze_detector
        self.save_dets = save_dets
        self.use_fixed_dets = use_fixed_dets
        self.pickle_path = pickle_path
        self.uncertainty_ovtrack = uncertainty_ovtrack
        self.slack_ovtrack = slack_ovtrack
        if save_dets:
            assert pickle_path is not None, "please indicate pickle path"
            assert (
                use_fixed_dets == False
            ), "use_fixed_dets and save_dets should not be used in the same time"
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_head,
        ]

        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def prepare_cfg(self, kwargs):
        if kwargs.get("train_cfg", False):
            if kwargs["train_cfg"].get("embed", None):
                kwargs["roi_head"]["track_train_cfg"] = kwargs["train_cfg"].get(
                    "embed", None
                )

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def init_motion(self):
        self.motion = build_motion(self.motion_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img) # 2, 3, 704, 1184 -> [2-256-176-296, 2-512-88-148, 2-1024-44-74, 2-2048-22-37]
        if self.with_neck:
            x = self.neck(x) # [2,256,(176-296, 88-148, 44-74, 22-37, 11-19)]
            return x

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_match_indices,
        ref_img,
        ref_img_metas,
        ref_gt_bboxes,
        ref_gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        **kwargs
    ):

        x = self.extract_feat(img)# 2, 3, 800, 1344 -> 2, 3, 704, 1184
        ref_x = self.extract_feat(ref_img) # 2,3,736,1248 ->  2, 3, 640, 1088

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)

        if self.freeze_detector:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)

        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        if self.uncertainty_ovtrack:
            roi_losses = self.roi_head.uncertainty_forward_train(
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
                gt_bboxes_ignore,
                gt_masks,
                ref_gt_bboxes_ignore,
                **kwargs
            )
        else:
            roi_losses = self.roi_head.forward_train(
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
                gt_bboxes_ignore,
                gt_masks,
                ref_gt_bboxes_ignore,
                **kwargs
            )
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):

        assert self.roi_head.with_track, "Track head must be implemented."
        img_name = img_metas[0]["filename"]
        if img_name is not None:
            pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")

        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id == 0:
            self.init_tracker()
        if self.roi_head.custom_classes:
            # The threshold needs to be adjusted according to the number of test classes in the custom case
            self.tracker.init_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.1
            self.tracker.obj_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.05

        x = self.extract_feat(img)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        if self.slack_ovtrack:
            outputs = self.roi_head.uncertainty_simple_test(x, img_metas, proposal_list, rescale)
        else:
            if hasattr(self.roi_head, 'use_cls_static_ratio') and self.roi_head.use_cls_static_ratio and hasattr(self.roi_head, 'use_motion_static_ratio') and self.roi_head.use_motion_static_ratio:
                # processing all the sequences here, special designed for the Uncertainty Tracker
                if not hasattr(self.tracker, 'fusion_head'):
                    self.tracker.set_fusion_head(self.roi_head.fusion_head, self.roi_head.track_head.loss_cyc)
                outputs = self.roi_head.uncertainty_simple_test(x, img_metas, proposal_list, rescale)
            else:
                outputs = self.roi_head.simple_test(x, img_metas, proposal_list, rescale)

        if len(outputs) == 4:
            det_bboxes, det_labels, cem_feats, track_feats = outputs
            if cem_feats is None:
                cem_feats = copy.deepcopy(track_feats)
        elif len(outputs) == 3:
            det_bboxes, det_labels, track_feats = outputs
            cem_feats = copy.deepcopy(track_feats)

        if track_feats is not None:
            # if not hasattr(self, 'motion') and self.motion_cfg is not None:
            #     self.init_motion()
            #     self.tracker.kf = self.motion

            # if not self.use_kf_motion:
            #     # original way
            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                embeds=track_feats,
                cls_embeds=cem_feats,
                frame_id=frame_id,
                method=self.method,
                filename=img_metas[0]['filename'][img_metas[0]['filename'].index('val'):]
            )
            # ovtrack slack
            # else:
            #     bboxes, labels, ids = self.tracker.match(
            #         # model=self,
            #         bboxes=det_bboxes,
            #         labels=det_labels,
            #         embeds=track_feats,
            #         cls_embeds=cem_feats,
            #         frame_id=frame_id,
            #         method=self.method,
            #     )



        bbox_result = bbox2result(det_bboxes, det_labels, self.roi_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids, self.roi_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.roi_head.num_classes)
            ]
        results = dict(bbox_results=bbox_result, track_results=track_result)

        return results

    def show_result(
        self,
        img,
        result,
        vis_thr=0.2,
        thickness=2,
        font_scale=0.1,
        show=False,
        out_file=None,
        wait_time=0,
        backend="cv2",
        **kwargs
    ):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get("track_results", None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        valid_ids = bboxes[:, -1] > vis_thr
        bboxes = bboxes[valid_ids]
        labels = labels[valid_ids]
        ids = ids[valid_ids]

        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.roi_head.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend,
        )
        return img




@MODELS.register_module()
class OVTrackSelfTrain(TwoStageDetector):
    def __init__(
            self,
            tracker=None,
            freeze_detector=False,
            save_dets=False,
            pickle_path=None,
            use_fixed_dets=False,
            method="ovtrack-teta",
            *args,
            **kwargs
    ):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker
        self.method = method
        print(self.method)
        self.freeze_detector = freeze_detector
        self.save_dets = save_dets
        self.use_fixed_dets = use_fixed_dets
        self.pickle_path = pickle_path
        if save_dets:
            assert pickle_path is not None, "please indicate pickle path"
            assert (
                    use_fixed_dets == False
            ), "use_fixed_dets and save_dets should not be used in the same time"
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_head,
        ]

        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def prepare_cfg(self, kwargs):
        if kwargs.get("train_cfg", False):
            if kwargs["train_cfg"].get("embed", None):
                kwargs["roi_head"]["track_train_cfg"] = kwargs["train_cfg"].get(
                    "embed", None
                )

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img) # 2, 3, 704, 1184 -> [2-256-176-296, 2-512-88-148, 2-1024-44-74, 2-2048-22-37]
        if self.with_neck:
            x = self.neck(x) # [2,256,(176-296, 88-148, 44-74, 22-37, 11-19)]
            return x

    def forward_train(
            self,
            img,
            img_metas,
            **kwargs
    ):

        img_x_list = []
        for i in range(len(img)):
            x = self.extract_feat(img[i])# 2, 3, 800, 1344 -> 2, 3, 704, 1184
            img_x_list.append(x)


        losses = dict()

        proposal_list_for_x = []
        if self.freeze_detector:
            for i in range(len(img_x_list)):
                proposal_list = self.rpn_head.simple_test_rpn(img_x_list[i], img_metas[i])
                proposal_list_for_x.append(proposal_list)


        roi_losses = self.roi_head.forward_train_self(
            img_x_list,
            img_metas,
            proposal_list_for_x,
            **kwargs
        )

        if roi_losses is not None:
            losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):

        assert self.roi_head.with_track, "Track head must be implemented."
        img_name = img_metas[0]["filename"]
        if img_name is not None:
            pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")

        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id == 0:
            self.init_tracker()
        if self.roi_head.custom_classes:
            # The threshold needs to be adjusted according to the number of test classes in the custom case
            self.tracker.init_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.1
            self.tracker.obj_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.05

        x = self.extract_feat(img)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        outputs = self.roi_head.simple_test(x, img_metas, proposal_list, rescale)

        if len(outputs) == 4:
            det_bboxes, det_labels, cem_feats, track_feats = outputs
            if cem_feats is None:
                cem_feats = copy.deepcopy(track_feats)
        elif len(outputs) == 3:
            det_bboxes, det_labels, track_feats = outputs
            cem_feats = copy.deepcopy(track_feats)

        if track_feats is not None:

            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                embeds=track_feats,
                cls_embeds=cem_feats,
                frame_id=frame_id,
                method=self.method,
            )

        bbox_result = bbox2result(det_bboxes, det_labels, self.roi_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids, self.roi_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.roi_head.num_classes)
            ]
        results = dict(bbox_results=bbox_result, track_results=track_result)

        return results

    def show_result(
            self,
            img,
            result,
            vis_thr=0.2,
            thickness=2,
            font_scale=0.1,
            show=False,
            out_file=None,
            wait_time=0,
            backend="cv2",
            **kwargs
    ):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get("track_results", None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        valid_ids = bboxes[:, -1] > vis_thr
        bboxes = bboxes[valid_ids]
        labels = labels[valid_ids]
        ids = ids[valid_ids]

        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.roi_head.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend,
        )
        return img


@MODELS.register_module()
class HybridSortSelfTrain(TwoStageDetector):
    def __init__(
            self,
            tracker=None,
            freeze_detector=False,
            save_dets=False,
            pickle_path=None,
            use_fixed_dets=False,
            method="ovtrack-teta",
            *args,
            **kwargs
    ):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker
        self.method = method
        print(self.method)
        self.freeze_detector = freeze_detector
        self.save_dets = save_dets
        self.use_fixed_dets = use_fixed_dets
        self.pickle_path = pickle_path
        if save_dets:
            assert pickle_path is not None, "please indicate pickle path"
            assert (
                    use_fixed_dets == False
            ), "use_fixed_dets and save_dets should not be used in the same time"
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_head,
        ]

        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def prepare_cfg(self, kwargs):
        if kwargs.get("train_cfg", False):
            if kwargs["train_cfg"].get("embed", None):
                kwargs["roi_head"]["track_train_cfg"] = kwargs["train_cfg"].get(
                    "embed", None
                )

    def init_tracker(self):
        self.tracker = build_tracker(self.tracker_cfg)
        # self.hybrid_tracker = build_tracker({"type":"Hybrid_Sort_ReID"})

    def init_motion(self):
        self.motion = build_motion(self.motion_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img) # 2, 3, 704, 1184 -> [2-256-176-296, 2-512-88-148, 2-1024-44-74, 2-2048-22-37]
        if self.with_neck:
            x = self.neck(x) # [2,256,(176-296, 88-148, 44-74, 22-37, 11-19)]
            return x

    def forward_train(
            self,
            img,
            img_metas,
            **kwargs
    ):

        img_x_list = []
        for i in range(len(img)):
            x = self.extract_feat(img[i])# 2, 3, 800, 1344 -> 2, 3, 704, 1184
            img_x_list.append(x)


        losses = dict()

        proposal_list_for_x = []
        if self.freeze_detector:
            for i in range(len(img_x_list)):
                proposal_list = self.rpn_head.simple_test_rpn(img_x_list[i], img_metas[i])
                proposal_list_for_x.append(proposal_list)


        roi_losses = self.roi_head.forward_train_self(
            img_x_list,
            img_metas,
            proposal_list_for_x,
            **kwargs
        )

        if roi_losses is not None:
            losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):

        assert self.roi_head.with_track, "Track head must be implemented."
        img_name = img_metas[0]["filename"]
        if img_name is not None:
            pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")

        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id == 0:
            self.init_tracker()
        # if self.roi_head.custom_classes:
        #     # The threshold needs to be adjusted according to the number of test classes in the custom case
        #     self.tracker.init_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.1
        #     self.tracker.obj_score_thr = 1 / (self.roi_head.num_classes + 1) + 0.05

        x = self.extract_feat(img)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        outputs = self.roi_head.simple_test(x, img_metas, proposal_list, rescale)

        if len(outputs) == 4:
            det_bboxes, det_labels, cem_feats, track_feats = outputs
            if cem_feats is None:
                cem_feats = copy.deepcopy(track_feats)
        elif len(outputs) == 3:
            det_bboxes, det_labels, track_feats = outputs
            cem_feats = copy.deepcopy(track_feats)

        # if img_name == 'data/tao/frames/val/YFCC100M/v_b74458f740348cd7c26b4c4339e0c5d6/frame1411.jpg':
        #     print()

        if track_feats is not None:
            labels_clone = det_labels.clone()

            bboxes_ovtrack, labels_ovtrack, ids_ovtrack = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                embeds=track_feats,
                cls_embeds=cem_feats,
                frame_id=frame_id,
                method=self.method,
            )


            # info_imgs = [torch.tensor([1080]), torch.tensor([1920]), torch.tensor([1]), torch.tensor([2]),
            #              ['dancetrack0005/img1/00000001.jpg']]
            info_imgs = [torch.tensor(img_metas[0]['img_shape'][0]), torch.tensor(img_metas[0]['img_shape'][1])]
            img_size = img_metas[0]['img_shape'][:2]
            # normalization confidence
            online_targets = self.hybrid_tracker.update(det_bboxes.cpu(), info_imgs, img_size, cls_labels=labels_clone.cpu(), id_feature=track_feats)  # [hgx0411] id_feature
            bboxes = online_targets[:, :4]
            if bboxes.shape[0] == 0:
                labels = None
                ids = None
                track_feats = None
            else:
                labels = online_targets[:, 5]
                ids = online_targets[:, 4]

        bbox_result = bbox2result(det_bboxes, det_labels, self.roi_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids, self.roi_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.roi_head.num_classes)
            ]
        results = dict(bbox_results=bbox_result, track_results=track_result)

        return results

    def show_result(
            self,
            img,
            result,
            vis_thr=0.2,
            thickness=2,
            font_scale=0.1,
            show=False,
            out_file=None,
            wait_time=0,
            backend="cv2",
            **kwargs
    ):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get("track_results", None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        valid_ids = bboxes[:, -1] > vis_thr
        bboxes = bboxes[valid_ids]
        labels = labels[valid_ids]
        ids = ids[valid_ids]

        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.roi_head.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend,
        )
        return img

