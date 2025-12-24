import copy
import mmcv
import cv2
import numpy as np
from mmdet.core import find_inside_bboxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import (MixUp, Mosaic, Normalize, Pad,
                                      RandomAffine, RandomFlip, Resize,
                                      YOLOXHSVRandomAug)
from numpy import random


@PIPELINES.register_module()
class SeqResize(Resize):
    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        outs, scale = [], None
        for i, _results in enumerate(results):
            if self.share_params and i > 0:
                _results["scale"] = scale
            _results = super().__call__(_results)
            if self.share_params and i == 0:
                scale = _results["scale"]
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqNormalize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        if self.share_params:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [
                    non_flip_ratio
                ]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            flip = cur_dir is not None
            flip_direction = cur_dir

            for _results in results:
                _results["flip"] = flip
                _results["flip_direction"] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqPad(Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):
    def __init__(
        self,
        crop_size,
        allow_negative_crop=False,
        share_params=False,
        bbox_clip_border=True,
    ):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            "gt_bboxes": ["gt_labels", "gt_instance_ids"],
            "gt_bboxes_ignore": ["gt_labels_ignore", "gt_instance_ids_ignore"],
        }
        self.bbox2mask = {
            "gt_bboxes": "gt_masks",
            "gt_bboxes_ignore": "gt_masks_ignore",
        }
        self.bbox_clip_border = bbox_clip_border

    def get_offsets(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results["img_info"]["crop_offsets"] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get("bbox_fields", []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array(
                [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
            )
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (
                key == "gt_bboxes"
                and not valid_inds.any()
                and not self.allow_negative_crop
            ):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2])
                )

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results

    def __call__(self, results):
        if self.share_params:
            offsets = self.get_offsets(results[0]["img"])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs


@PIPELINES.register_module()
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        share_params=True,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        params = dict()
        # delta
        if np.random.randint(2):
            params["delta"] = np.random.uniform(
                -self.brightness_delta, self.brightness_delta
            )
        else:
            params["delta"] = None
        # mode
        mode = np.random.randint(2)
        params["contrast_first"] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params["alpha"] = np.random.uniform(
                self.contrast_lower, self.contrast_upper
            )
        else:
            params["alpha"] = None
        # saturation
        if np.random.randint(2):
            params["saturation"] = np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )
        else:
            params["saturation"] = None
        # hue
        if np.random.randint(2):
            params["hue"] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params["hue"] = None
        # swap
        if np.random.randint(2):
            params["permutation"] = np.random.permutation(3)
        else:
            params["permutation"] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results["img_info"]["color_jitter"] = params

        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        img = results["img"]
        assert img.dtype == np.float32, (
            "PhotoMetricDistortion needs the input image of dtype np.float32,"
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        )
        # random brightness
        if params["delta"] is not None:
            img += params["delta"]

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params["saturation"] is not None:
            img[..., 1] *= params["saturation"]

        # random hue
        if params["hue"] is not None:
            img[..., 0] += params["hue"]
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # randomly swap channels
        if params["permutation"] is not None:
            img = img[..., params["permutation"]]

        results["img"] = img
        return results

    def __call__(self, results):
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class SeqMosaicWithTrack(Mosaic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        key_match_indices = []
        ref_match_indices = []
        key_labels = []
        ref_labels = []
        key_instances = []
        ref_instances = []
        key_labels.extend(results[0]['gt_labels'])
        for i in range(3):
            key_labels.extend(results[0]['mix_results'][i]['gt_labels'])
        ref_labels.extend(results[1]['gt_labels'])
        for i in range(3):
            ref_labels.extend(results[1]['mix_results'][i]['gt_labels'])

        for i, _results in enumerate(results):
            assert "mix_results" in _results
            if i == 0:
                # new transforms
                if 'gt_match_indices' not in _results:
                    key_match_indices.append(_results['ann_info']['match_indices'])
                    for _inner_mix_results in _results['mix_results']:
                        key_match_indices.append(_inner_mix_results['ann_info']['match_indices'])
                else:
                    key_match_indices.append(_results['gt_match_indices'])
                    for _inner_mix_results in _results['mix_results']:
                        key_match_indices.append(_inner_mix_results['gt_match_indices'])
            elif i == 1:
                if '' not in _results:
                    ref_match_indices.append(_results['ann_info']['match_indices'])
                    for _inner_mix_results in _results['mix_results']:
                        ref_match_indices.append(_inner_mix_results['ann_info']['match_indices'])
                else:
                    ref_match_indices.append(_results['gt_match_indices'])
                    for _inner_mix_results in _results['mix_results']:
                        ref_match_indices.append(_inner_mix_results['gt_match_indices'])
            else:
                assert len(results) == 2
            _results = self._mosaic_transform(_results)
            outs.append(_results)
        # TODO update the gt_match_indices here
        acc_key_match_indices = []
        acc_key_counter = 0
        acc_ref_match_indices = []
        acc_ref_counter = 0
        for i in range(len(key_match_indices)):
            inner_key_match_indices = key_match_indices[i]
            inner_ref_match_indices = ref_match_indices[i]
            for j, val in enumerate(inner_key_match_indices):
                if val == -1:
                    acc_key_match_indices.append(val)
                else:
                    acc_key_match_indices.append(val + acc_key_counter)
            for j, val in enumerate(inner_ref_match_indices):
                if val == -1:
                    acc_ref_match_indices.append(val)
                else:
                    acc_ref_match_indices.append(val + acc_ref_counter)
            acc_key_counter += len(inner_ref_match_indices)
            acc_ref_counter += len(inner_key_match_indices)

        # used to test
        key_match_label = [ref_labels[index] if index!= -1 else -1 for index in  acc_key_match_indices]
        ref_match_label = [key_labels[index] if index!= -1 else -1 for index in  acc_ref_match_indices]

        for i in range(len(acc_key_match_indices)):
            acc_key_match_indices[i] = [acc_key_match_indices[i], i]
        for i in range(len(acc_ref_match_indices)):
            acc_ref_match_indices[i] = [acc_ref_match_indices[i], i]


        for i, out in enumerate(outs):
            if i == 0:
                key_gt_match_indices_correct = [acc_key_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = key_gt_match_indices_correct
            else:
                ref_gt_match_indices_correct = [acc_ref_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = ref_gt_match_indices_correct
        # find the new index
        key_index_dict = {}
        for i, (val, idx) in enumerate(key_gt_match_indices_correct):
            key_index_dict[idx] = (val, i)

        ref_index_dict = {}
        for i, (val, idx) in enumerate(ref_gt_match_indices_correct):
            ref_index_dict[idx] = (val, i)

        for i in range(len(key_gt_match_indices_correct)):
            if key_gt_match_indices_correct[i][0] == -1:
                key_gt_match_indices_correct[i] = -1
            else:
                idx = key_gt_match_indices_correct[i][0]
                if idx in ref_index_dict:
                    key_gt_match_indices_correct[i] = ref_index_dict[idx][1]
                else:
                    key_gt_match_indices_correct[i] = -1

        for i in range(len(ref_gt_match_indices_correct)):
            if ref_gt_match_indices_correct[i][0] == -1:
                ref_gt_match_indices_correct[i] = -1
            else:
                idx = ref_gt_match_indices_correct[i][0]
                if idx in key_index_dict:
                    ref_gt_match_indices_correct[i] = key_index_dict[idx][1]
                else:
                    ref_gt_match_indices_correct[i] = -1
        # used to test
        key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
        ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]
        false_counter = 0
        for val in key_match_label:
            if val == False:
                false_counter += 1
        assert false_counter == 0, "wrong match gt pair"

        # conver the index map relationship to the instance id
        # current_instance_id = 1
        # for i, idx in enumerate(key_gt_match_indices_correct):
        #     if idx == -1:
        #         continue
        #     else:
        #         opposide_idx = idx
        #         key_gt_match_indices_correct[i] = current_instance_id
        #         ref_gt_match_indices_correct[opposide_idx] = current_instance_id
        #         current_instance_id += 1
        outs[0]['new_gt_match_indices'] = np.array(key_gt_match_indices_correct)
        outs[1]['new_gt_match_indices'] = np.array(ref_gt_match_indices_correct)

        # only used the right way to progress
        outs[0]['gt_match_indices'] = outs[0]['new_gt_match_indices']
        outs[1]['gt_match_indices'] = outs[1]['new_gt_match_indices']

        # clear the wrong alignment keys
        del_key_list = ['gt_instance_ids']
        for del_key in del_key_list:
            del outs[0][del_key]
            del outs[1][del_key]
        return outs

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_inds = np.arange(0, len(mosaic_labels))

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(
                    mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
                )
                mosaic_bboxes[:, 1::2] = np.clip(
                    mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
                )

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels, mosaic_inds = self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels, mosaic_inds
                )

        # remove outside bboxes
        inside_inds = find_inside_bboxes(
            mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
        )
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        mosaic_inds = mosaic_inds[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels
        results["gt_match_indices"] = mosaic_inds

        return results

    def _filter_box_candidates(self, bboxes, labels, inds):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds], inds[valid_inds]


@PIPELINES.register_module()
class SeqMosaicWithTrackNew(Mosaic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        key_match_indices = []
        ref_match_indices = []
        key_labels = []
        ref_labels = []
        key_instances = []
        ref_instances = []
        key_labels.extend(results[0]['gt_labels'])
        for i in range(3):
            key_labels.extend(results[0]['mix_results'][i]['gt_labels'])
        ref_labels.extend(results[1]['gt_labels'])
        for i in range(3):
            ref_labels.extend(results[1]['mix_results'][i]['gt_labels'])

        for i, _results in enumerate(results):
            assert "mix_results" in _results
            if i == 0:
                # new transforms
                if 'gt_match_indices' not in _results:
                    key_match_indices.append(_results['ann_info']['match_indices'])
                    key_instances.append(_results['ann_info']['instance_ids'])
                    for _inner_mix_results in _results['mix_results']:
                        key_match_indices.append(_inner_mix_results['ann_info']['match_indices'])
                        key_instances.append(_inner_mix_results['ann_info']['instance_ids'])
                else:
                    key_match_indices.append(_results['gt_match_indices'])
                    key_instances.append(_results['instance_ids'])
                    for _inner_mix_results in _results['mix_results']:
                        key_match_indices.append(_inner_mix_results['gt_match_indices'])
                        key_instances.append(_inner_mix_results['instance_ids'])
            elif i == 1:
                if '' not in _results:
                    ref_match_indices.append(_results['ann_info']['match_indices'])
                    ref_instances.append(_results['ann_info']['instance_ids'])
                    for _inner_mix_results in _results['mix_results']:
                        ref_match_indices.append(_inner_mix_results['ann_info']['match_indices'])
                        ref_instances.append(_inner_mix_results['ann_info']['instance_ids'])
                else:
                    ref_match_indices.append(_results['gt_match_indices'])
                    ref_instances.append(_results['instance_ids'])
                    for _inner_mix_results in _results['mix_results']:
                        ref_match_indices.append(_inner_mix_results['gt_match_indices'])
                        ref_instances.append(_inner_mix_results['instance_ids'])
            else:
                assert len(results) == 2
            _results = self._mosaic_transform(_results)
            outs.append(_results)
        # TODO update the gt_match_indices here
        acc_key_match_indices = []
        acc_key_counter = 0
        acc_ref_match_indices = []
        acc_ref_counter = 0
        for i in range(len(key_match_indices)):
            inner_key_match_indices = key_match_indices[i]
            inner_ref_match_indices = ref_match_indices[i]
            for j, val in enumerate(inner_key_match_indices):
                if val == -1:
                    acc_key_match_indices.append(val)
                else:
                    acc_key_match_indices.append(val + acc_key_counter)
            for j, val in enumerate(inner_ref_match_indices):
                if val == -1:
                    acc_ref_match_indices.append(val)
                else:
                    acc_ref_match_indices.append(val + acc_ref_counter)
            acc_key_counter += len(inner_ref_match_indices)
            acc_ref_counter += len(inner_key_match_indices)

        # used to test
        key_match_label = [ref_labels[index] if index!= -1 else -1 for index in  acc_key_match_indices]
        ref_match_label = [key_labels[index] if index!= -1 else -1 for index in  acc_ref_match_indices]

        for i in range(len(acc_key_match_indices)):
            acc_key_match_indices[i] = [acc_key_match_indices[i], i]
        for i in range(len(acc_ref_match_indices)):
            acc_ref_match_indices[i] = [acc_ref_match_indices[i], i]


        for i, out in enumerate(outs):
            if i == 0:
                key_gt_match_indices_correct = [acc_key_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = key_gt_match_indices_correct
            else:
                ref_gt_match_indices_correct = [acc_ref_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = ref_gt_match_indices_correct
        # find the new index
        key_index_dict = {}
        for i, (val, idx) in enumerate(key_gt_match_indices_correct):
            key_index_dict[idx] = (val, i)

        ref_index_dict = {}
        for i, (val, idx) in enumerate(ref_gt_match_indices_correct):
            ref_index_dict[idx] = (val, i)

        for i in range(len(key_gt_match_indices_correct)):
            if key_gt_match_indices_correct[i][0] == -1:
                key_gt_match_indices_correct[i] = -1
            else:
                idx = key_gt_match_indices_correct[i][0]
                if idx in ref_index_dict:
                    key_gt_match_indices_correct[i] = ref_index_dict[idx][1]
                else:
                    key_gt_match_indices_correct[i] = -1

        for i in range(len(ref_gt_match_indices_correct)):
            if ref_gt_match_indices_correct[i][0] == -1:
                ref_gt_match_indices_correct[i] = -1
            else:
                idx = ref_gt_match_indices_correct[i][0]
                if idx in key_index_dict:
                    ref_gt_match_indices_correct[i] = key_index_dict[idx][1]
                else:
                    ref_gt_match_indices_correct[i] = -1
        # used to test
        key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
        ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]
        false_counter = 0
        for val in key_match_label:
            if val == False:
                false_counter += 1
        assert false_counter == 0, "wrong match gt pair"

        # conver the index map relationship to the instance id
        # current_instance_id = 1
        # for i, idx in enumerate(key_gt_match_indices_correct):
        #     if idx == -1:
        #         continue
        #     else:
        #         opposide_idx = idx
        #         key_gt_match_indices_correct[i] = current_instance_id
        #         ref_gt_match_indices_correct[opposide_idx] = current_instance_id
        #         current_instance_id += 1
        outs[0]['new_gt_match_indices'] = np.array(key_gt_match_indices_correct)
        outs[1]['new_gt_match_indices'] = np.array(ref_gt_match_indices_correct)

        # only used the right way to progress
        outs[0]['gt_match_indices'] = outs[0]['new_gt_match_indices']
        outs[1]['gt_match_indices'] = outs[1]['new_gt_match_indices']

        # clear the wrong alignment keys
        outs[0]['gt_instance_ids'] = np.concatenate(key_instances)[outs[0]['valid_index']]
        outs[1]['gt_instance_ids'] = np.concatenate(ref_instances)[outs[1]['valid_index']]
        # del_key_list = ['gt_instance_ids']
        # for del_key in del_key_list:
        #     del outs[0][del_key]
        #     del outs[1][del_key]
        return outs

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_inds = np.arange(0, len(mosaic_labels))

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(
                    mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
                )
                mosaic_bboxes[:, 1::2] = np.clip(
                    mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
                )

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels, mosaic_inds = self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels, mosaic_inds
                )

        # remove outside bboxes
        inside_inds = find_inside_bboxes(
            mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
        )
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        mosaic_inds = mosaic_inds[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels
        results["gt_match_indices"] = mosaic_inds
        results['valid_index'] = inside_inds

        return results

    def _filter_box_candidates(self, bboxes, labels, inds):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds], inds[valid_inds]




@PIPELINES.register_module()
class SeqMosaic(Mosaic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        key_match_indices = []
        ref_match_indices = []
        key_labels = []
        ref_labels = []
        key_labels.extend(results[0]['gt_labels'])
        for i in range(3):
            key_labels.extend(results[0]['mix_results'][i]['gt_labels'])
        ref_labels.extend(results[1]['gt_labels'])
        for i in range(3):
            ref_labels.extend(results[1]['mix_results'][i]['gt_labels'])

        for i, _results in enumerate(results):
            assert "mix_results" in _results
            if i == 0:
                key_match_indices.append(_results['gt_match_indices'])
                for _inner_mix_results in _results['mix_results']:
                    key_match_indices.append(_inner_mix_results['gt_match_indices'])
            elif i == 1:
                ref_match_indices.append(_results['gt_match_indices'])
                for _inner_mix_results in _results['mix_results']:
                    ref_match_indices.append(_inner_mix_results['gt_match_indices'])
            else:
                assert len(results) == 2
            _results = self._mosaic_transform(_results)
            outs.append(_results)
        # TODO update the gt_match_indices here
        acc_key_match_indices = []
        acc_key_counter = 0
        acc_ref_match_indices = []
        acc_ref_counter = 0
        for i in range(len(key_match_indices)):
            inner_key_match_indices = key_match_indices[i]
            inner_ref_match_indices = ref_match_indices[i]
            for j, val in enumerate(inner_key_match_indices):
                if val == -1:
                    acc_key_match_indices.append(val)
                else:
                    acc_key_match_indices.append(val + acc_key_counter)
            for j, val in enumerate(inner_ref_match_indices):
                if val == -1:
                    acc_ref_match_indices.append(val)
                else:
                    acc_ref_match_indices.append(val + acc_ref_counter)
            acc_key_counter += len(inner_ref_match_indices)
            acc_ref_counter += len(inner_key_match_indices)

        # used to test
        key_match_label = [ref_labels[index] if index!= -1 else -1 for index in  acc_key_match_indices]
        ref_match_label = [key_labels[index] if index!= -1 else -1 for index in  acc_ref_match_indices]

        for i in range(len(acc_key_match_indices)):
            acc_key_match_indices[i] = [acc_key_match_indices[i], i]
        for i in range(len(acc_ref_match_indices)):
            acc_ref_match_indices[i] = [acc_ref_match_indices[i], i]


        for i, out in enumerate(outs):
            if i == 0:
                key_gt_match_indices_correct = [acc_key_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = key_gt_match_indices_correct
            else:
                ref_gt_match_indices_correct = [acc_ref_match_indices[idx] for idx in out['gt_match_indices']]
                out['new_gt_match_indices'] = ref_gt_match_indices_correct
        # find the new index
        key_index_dict = {}
        for i, (val, idx) in enumerate(key_gt_match_indices_correct):
            key_index_dict[idx] = (val, i)

        ref_index_dict = {}
        for i, (val, idx) in enumerate(ref_gt_match_indices_correct):
            ref_index_dict[idx] = (val, i)

        for i in range(len(key_gt_match_indices_correct)):
            if key_gt_match_indices_correct[i][0] == -1:
                key_gt_match_indices_correct[i] = -1
            else:
                idx = key_gt_match_indices_correct[i][0]
                if idx in ref_index_dict:
                    key_gt_match_indices_correct[i] = ref_index_dict[idx][1]
                else:
                    key_gt_match_indices_correct[i] = -1

        for i in range(len(ref_gt_match_indices_correct)):
            if ref_gt_match_indices_correct[i][0] == -1:
                ref_gt_match_indices_correct[i] = -1
            else:
                idx = ref_gt_match_indices_correct[i][0]
                if idx in key_index_dict:
                    ref_gt_match_indices_correct[i] = key_index_dict[idx][1]
                else:
                    ref_gt_match_indices_correct[i] = -1
        # used to test
        key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(key_gt_match_indices_correct)]
        ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index!= -1 else -1 for i, index in enumerate(ref_gt_match_indices_correct)]
        false_counter = 0
        for val in key_match_label:
            if val == False:
                false_counter += 1
        assert false_counter == 0, "wrong match gt pair"

        # conver the index map relationship to the instance id !!!!!!!! here is instance id then masked in SeqAffine, later in SeqCollect to be the match matrix in `with_track` version recover the function here
        current_instance_id = 1
        for i, idx in enumerate(key_gt_match_indices_correct):
            if idx == -1:
                continue
            else:
                opposide_idx = idx
                key_gt_match_indices_correct[i] = current_instance_id
                ref_gt_match_indices_correct[opposide_idx] = current_instance_id
                current_instance_id += 1
        outs[0]['new_gt_match_indices'] = np.array(key_gt_match_indices_correct)
        outs[1]['new_gt_match_indices'] = np.array(ref_gt_match_indices_correct)

        # only used the right way to progress
        outs[0]['gt_match_indices'] = outs[0]['new_gt_match_indices']
        outs[1]['gt_match_indices'] = outs[1]['new_gt_match_indices']

        return outs

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_inds = np.arange(0, len(mosaic_labels))

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(
                    mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
                )
                mosaic_bboxes[:, 1::2] = np.clip(
                    mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
                )

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels, mosaic_inds = self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels, mosaic_inds
                )

        # remove outside bboxes
        inside_inds = find_inside_bboxes(
            mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
        )
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        mosaic_inds = mosaic_inds[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels
        results["gt_match_indices"] = mosaic_inds

        return results

    def _filter_box_candidates(self, bboxes, labels, inds):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds], inds[valid_inds]



# @PIPELINES.register_module()
# class SeqMosaic(Mosaic):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, results):
#         outs = []
#         for _results in results:
#             _results = self._mosaic_transform(_results)
#             outs.append(_results)
#         return outs
#
#     def _mosaic_transform(self, results):
#         """Mosaic transform function.
#
#         Args:
#             results (dict): Result dict.
#
#         Returns:
#             dict: Updated result dict.
#         """
#
#         assert "mix_results" in results
#         mosaic_labels = []
#         mosaic_bboxes = []
#         if len(results["img"].shape) == 3:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
#                 self.pad_val,
#                 dtype=results["img"].dtype,
#             )
#         else:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
#                 self.pad_val,
#                 dtype=results["img"].dtype,
#             )
#
#         # mosaic center x, y
#         center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
#         center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
#         center_position = (center_x, center_y)
#
#         loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
#         for i, loc in enumerate(loc_strs):
#             if loc == "top_left":
#                 results_patch = copy.deepcopy(results)
#             else:
#                 results_patch = copy.deepcopy(results["mix_results"][i - 1])
#
#             img_i = results_patch["img"]
#             h_i, w_i = img_i.shape[:2]
#             # keep_ratio resize
#             scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
#             img_i = mmcv.imresize(
#                 img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
#             )
#
#             # compute the combine parameters
#             paste_coord, crop_coord = self._mosaic_combine(
#                 loc, center_position, img_i.shape[:2][::-1]
#             )
#             x1_p, y1_p, x2_p, y2_p = paste_coord
#             x1_c, y1_c, x2_c, y2_c = crop_coord
#
#             # crop and paste image
#             mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
#
#             # adjust coordinate
#             gt_bboxes_i = results_patch["gt_bboxes"]
#             gt_labels_i = results_patch["gt_labels"]
#
#             if gt_bboxes_i.shape[0] > 0:
#                 padw = x1_p - x1_c
#                 padh = y1_p - y1_c
#                 gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
#                 gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh
#
#             mosaic_bboxes.append(gt_bboxes_i)
#             mosaic_labels.append(gt_labels_i)
#
#         if len(mosaic_labels) > 0:
#             mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
#             mosaic_labels = np.concatenate(mosaic_labels, 0)
#             mosaic_inds = np.arange(0, len(mosaic_labels))
#
#             if self.bbox_clip_border:
#                 mosaic_bboxes[:, 0::2] = np.clip(
#                     mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
#                 )
#                 mosaic_bboxes[:, 1::2] = np.clip(
#                     mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
#                 )
#
#             if not self.skip_filter:
#                 mosaic_bboxes, mosaic_labels, mosaic_inds = self._filter_box_candidates(
#                     mosaic_bboxes, mosaic_labels, mosaic_inds
#                 )
#
#         # remove outside bboxes
#         inside_inds = find_inside_bboxes(
#             mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
#         )
#         mosaic_bboxes = mosaic_bboxes[inside_inds]
#         mosaic_labels = mosaic_labels[inside_inds]
#         mosaic_inds = mosaic_inds[inside_inds]
#
#         results["img"] = mosaic_img
#         results["img_shape"] = mosaic_img.shape
#         results["gt_bboxes"] = mosaic_bboxes
#         results["gt_labels"] = mosaic_labels
#         results["gt_match_indices"] = mosaic_inds
#
#         return results
#
#     def _filter_box_candidates(self, bboxes, labels, inds):
#         """Filter out bboxes too small after Mosaic."""
#         bbox_w = bboxes[:, 2] - bboxes[:, 0]
#         bbox_h = bboxes[:, 3] - bboxes[:, 1]
#         valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
#         valid_inds = np.nonzero(valid_inds)[0]
#         return bboxes[valid_inds], labels[valid_inds], inds[valid_inds]


def update_match_indices(key_gt_match_indices, ref_gt_match_indices, key_valid_index, ref_valid_index):
    """
    更新匹配索引关系，确保在筛选后索引关系保持正确

    示例:
    key_gt_match_indices = [1, 3, 0, 2]  # key[0]->ref[1], key[1]->ref[3], key[2]->ref[0], key[3]->ref[2]
    ref_gt_match_indices = [2, 0, 3, 1]  # ref[0]->key[2], ref[1]->key[0], ref[2]->key[3], ref[3]->key[1]
    key_valid_index = [True, False, True, True]  # 保留key中的0,2,3号目标
    ref_valid_index = [True, True, False, True]  # 保留ref中的0,1,3号目标
    """
    # 1. 获取保留目标的原始索引
    valid_key_indices = [i for i, valid in enumerate(key_valid_index) if valid]
    valid_ref_indices = [i for i, valid in enumerate(ref_valid_index) if valid]

    # 2. 创建原始索引到新索引的映射
    key_old_to_new = {old: new for new, old in enumerate(valid_key_indices)}
    ref_old_to_new = {old: new for new, old in enumerate(valid_ref_indices)}

    # 3. 初始化新的匹配数组，默认为-1（表示无匹配）
    # new_key_matches = [-1] * len(valid_key_indices)
    # new_ref_matches = [-1] * len(valid_ref_indices)

    # 4. 更新匹配关系
    # update the key index
    new_key_match_instances = []
    new_ref_match_instances = []
    for save_index in valid_key_indices:
        ori_mapped_index = key_gt_match_indices[save_index]
        if ori_mapped_index == -1:
            new_key_match_instances.append(-1)
        elif ori_mapped_index not in ref_old_to_new:
            new_key_match_instances.append(-1)
        else:
            new_key_match_instances.append(ref_old_to_new[ori_mapped_index])

    for save_index in valid_ref_indices:
        ori_mapped_index = ref_gt_match_indices[save_index]
        if ori_mapped_index == -1:
            new_ref_match_instances.append(-1)
        elif ori_mapped_index not in key_old_to_new:
            new_ref_match_instances.append(-1)
        else:
            new_ref_match_instances.append(key_old_to_new[ori_mapped_index])

    # new_key_match_instances = [ref_old_to_new[key_gt_match_indices[saved_index]] for saved_index in valid_key_indices]
    # new_ref_match_instances = [key_old_to_new[ref_gt_match_indices[saved_index]] for saved_index in valid_ref_indices]
    # update the ref index
    # for old_key_idx in valid_key_indices:
    #     new_key_idx = key_old_to_new[old_key_idx]
    #     old_ref_idx = key_gt_match_indices[old_key_idx]
    #
    #     # 如果匹配的参考帧目标仍然有效
    #     if old_ref_idx != -1 and ref_valid_index[old_ref_idx]:
    #         new_ref_idx = ref_old_to_new[old_ref_idx]
    #         new_key_matches[new_key_idx] = new_ref_idx
    #         new_ref_matches[new_ref_idx] = new_key_idx

    return new_key_match_instances, new_ref_match_instances


@PIPELINES.register_module()
class SeqRandomAffine(RandomAffine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.random_affine(_results)
            outs.append(_results)
        # update the gt_match_matrix
        # if 'new_gt_match_indices' in outs[0]:
        #     key_gt_match_indices = outs[0]['new_gt_match_indices']
        #     ref_gt_match_indices = outs[1]['new_gt_match_indices']
        #     key_valid_index = outs[0]['gt_bbox_mask']
        #     ref_valid_index = outs[1]['gt_bbox_mask']
        #
        #     new_key_matches, new_ref_matches = update_match_indices(
        #         key_gt_match_indices, ref_gt_match_indices,
        #         key_valid_index, ref_valid_index
        #     )
        #
        #     outs[0]['new_gt_match_indices'] = new_key_matches
        #     outs[0]['gt_match_indices'] = new_key_matches
        #
        #     outs[1]['new_gt_match_indices'] = new_ref_matches
        #     outs[1]['gt_match_indices'] = new_ref_matches
        #
        #     # test here
        #     key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index != -1 else -1 for
        #                        i, index in enumerate(new_key_matches)]
        #     ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index != -1 else -1 for
        #                        i, index in enumerate(new_ref_matches)]


        return outs

    def random_affine(self, results):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes
                    )
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ["gt_bboxes"]:
                    if "gt_labels" in results:
                        results["gt_labels"] = results["gt_labels"][valid_index]
                        results["gt_match_indices"] = results["gt_match_indices"][
                            valid_index
                        ]
                        results["new_gt_match_indices"] = results["new_gt_match_indices"][
                            valid_index
                        ]
                    if 'gt_instance_ids' in results:
                        results['gt_instance_ids'] = results['gt_instance_ids'][valid_index]
                    results["gt_bbox_mask"] = valid_index



                if "gt_masks" in results:
                    raise NotImplementedError("RandomAffine only supports bbox.")
        return results


@PIPELINES.register_module()
class SeqRandomAffineWithTrack(RandomAffine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.random_affine(_results)
            outs.append(_results)
        # update the gt_match_matrix
        if 'new_gt_match_indices' in outs[0]:
            key_gt_match_indices = outs[0]['new_gt_match_indices']
            ref_gt_match_indices = outs[1]['new_gt_match_indices']
            key_valid_index = outs[0]['gt_bbox_mask']
            ref_valid_index = outs[1]['gt_bbox_mask']

            new_key_matches, new_ref_matches = update_match_indices(
                key_gt_match_indices, ref_gt_match_indices,
                key_valid_index, ref_valid_index
            )

            outs[0]['new_gt_match_indices'] = new_key_matches
            outs[0]['gt_match_indices'] = new_key_matches

            outs[1]['new_gt_match_indices'] = new_ref_matches
            outs[1]['gt_match_indices'] = new_ref_matches

            # test here
            key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index != -1 else -1 for
                               i, index in enumerate(new_key_matches)]
            ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index != -1 else -1 for
                               i, index in enumerate(new_ref_matches)]


        return outs

    def random_affine(self, results):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes
                    )
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ["gt_bboxes"]:
                    if "gt_labels" in results:
                        results["gt_labels"] = results["gt_labels"][valid_index]
                        # if 'gt_match_indices' in results:
                        #     results["gt_match_indices"] = results["gt_match_indices"][
                        #         valid_index
                        #     ]
                        # if 'new_gt_match_indices' in results:
                        #     results["new_gt_match_indices"] = results["new_gt_match_indices"][
                        #         valid_index
                        #     ]
                        # if 'ann_info' in results and 'match_indices' in results['ann_info']:
                        #     # in case of masic or not
                        #     if len(valid_index) == len(results['ann_info']['match_indices']):
                        #         results['gt_match_indices'] = results['ann_info']['match_indices'][valid_index]
                    if 'gt_instance_ids' in results:
                        results['gt_instance_ids'] = results['gt_instance_ids'][valid_index]
                    results["gt_bbox_mask"] = valid_index



                if "gt_masks" in results:
                    raise NotImplementedError("RandomAffine only supports bbox.")
        return results


@PIPELINES.register_module()
class SeqRandomAffineWithTrackNew(RandomAffine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.random_affine(_results)
            outs.append(_results)
        # update the gt_match_matrix
        if 'new_gt_match_indices' in outs[0]:
            key_gt_match_indices = outs[0]['new_gt_match_indices']
            ref_gt_match_indices = outs[1]['new_gt_match_indices']
            key_valid_index = outs[0]['gt_bbox_mask']
            ref_valid_index = outs[1]['gt_bbox_mask']

            new_key_matches, new_ref_matches = update_match_indices(
                key_gt_match_indices, ref_gt_match_indices,
                key_valid_index, ref_valid_index
            )

            outs[0]['new_gt_match_indices'] = new_key_matches
            outs[0]['gt_match_indices'] = new_key_matches

            outs[1]['new_gt_match_indices'] = new_ref_matches
            outs[1]['gt_match_indices'] = new_ref_matches

            # test here
            key_match_label = [outs[1]['gt_labels'][index] == outs[0]['gt_labels'][i] if index != -1 else -1 for
                               i, index in enumerate(new_key_matches)]
            ref_match_label = [outs[0]['gt_labels'][index] == outs[1]['gt_labels'][i] if index != -1 else -1 for
                               i, index in enumerate(new_ref_matches)]


        return outs

    def random_affine(self, results):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
                random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
                random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes
                    )
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ["gt_bboxes"]:
                    if "gt_labels" in results:
                        results["gt_labels"] = results["gt_labels"][valid_index]
                        # if 'gt_match_indices' in results:
                        #     results["gt_match_indices"] = results["gt_match_indices"][
                        #         valid_index
                        #     ]
                        # if 'new_gt_match_indices' in results:
                        #     results["new_gt_match_indices"] = results["new_gt_match_indices"][
                        #         valid_index
                        #     ]
                        # if 'ann_info' in results and 'match_indices' in results['ann_info']:
                        #     # in case of masic or not
                        #     if len(valid_index) == len(results['ann_info']['match_indices']):
                        #         results['gt_match_indices'] = results['ann_info']['match_indices'][valid_index]
                    if 'gt_instance_ids' in results:
                        results['gt_instance_ids'] = results['gt_instance_ids'][valid_index]
                    results["gt_bbox_mask"] = valid_index



                if "gt_masks" in results:
                    raise NotImplementedError("RandomAffine only supports bbox.")
        return results


@PIPELINES.register_module()
class SeqMixUp(MixUp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self._mixup_transform(_results)
            outs.append(_results)
        return outs

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        assert len(results["mix_results"]) == 1, "MixUp only support 2 images now !"

        if results["mix_results"][0]["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results["mix_results"][0]
        retrieve_img = retrieve_results["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones(
                    (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                    dtype=retrieve_img.dtype,
                )
                * self.pad_val
            )
        else:
            out_img = (
                np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val
            )

        # 1. keep_ratio resize
        scale_ratio = min(
            self.dynamic_scale[0] / retrieve_img.shape[0],
            self.dynamic_scale[1] / retrieve_img.shape[1],
        )
        retrieve_img = mmcv.imresize(
            retrieve_img,
            (
                int(retrieve_img.shape[1] * scale_ratio),
                int(retrieve_img.shape[0] * scale_ratio),
            ),
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
        )

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(
                retrieve_gt_bboxes[:, 0::2], 0, origin_w
            )
            retrieve_gt_bboxes[:, 1::2] = np.clip(
                retrieve_gt_bboxes[:, 1::2], 0, origin_h
            )

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]
            )

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w
            )
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h
            )

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results["gt_labels"]
        retrieve_gt_match_indices = retrieve_results["gt_match_indices"] + 999
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(
                retrieve_gt_bboxes.T, cp_retrieve_gt_bboxes.T
            )

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            retrieve_gt_match_indices = retrieve_gt_match_indices[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results["gt_bboxes"], cp_retrieve_gt_bboxes), axis=0
        )
        mixup_gt_labels = np.concatenate(
            (results["gt_labels"], retrieve_gt_labels), axis=0
        )
        mixup_gt_match_indices = np.concatenate(
            (results["gt_match_indices"], retrieve_gt_match_indices), axis=0
        )

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]
        mixup_gt_match_indices = mixup_gt_match_indices[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_labels"] = mixup_gt_labels
        results["gt_match_indices"] = mixup_gt_match_indices

        return results


@PIPELINES.register_module()
class SeqYOLOXHSVRandomAug(YOLOXHSVRandomAug):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs
