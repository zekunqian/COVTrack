from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.datasets.pipelines.loading import FilterAnnotations
import numpy as np


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs



@PIPELINES.register_module()
class SeqLoadAnnotationsWithTrack(LoadAnnotations):
    """Sequence load annotations.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.

    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    def __init__(self, with_ins_id=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_ins_id = with_ins_id

    def _load_ins_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results["gt_match_indices"] = results["ann_info"]["match_indices"].copy()

        return results

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_ins_id:
                _results = self._load_ins_ids(_results)
            outs.append(_results)
        return outs


# @PIPELINES.register_module()
# class SeqFilterAnnotations(FilterAnnotations):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, results):
#         outs = []
#         for _results in results:
#             _results = self._filter(_results)
#             outs.append(_results)
#         return outs
#
#     def _filter(self, results):
#         assert "gt_bboxes" in results
#         gt_bboxes = results["gt_bboxes"]
#         if gt_bboxes.shape[0] == 0:
#             return results
#         w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
#         h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
#         keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
#         if not keep.any():
#             if self.keep_empty:
#                 return None
#             else:
#                 return results
#         else:
#             keys = (
#                 "gt_bboxes",
#                 "gt_labels",
#                 "gt_masks",
#                 "gt_semantic_seg",
#                 "gt_match_indices",
#             )
#             for key in keys:
#                 if key in results:
#                     results[key] = results[key][keep]
#             return results

@PIPELINES.register_module()
class SeqFilterAnnotations(FilterAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self._filter(_results)
            outs.append(_results)



        return outs

    def _filter(self, results):
        assert "gt_bboxes" in results
        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = (
                "gt_bboxes",
                "gt_labels",
                "gt_masks",
                "gt_semantic_seg",
                "gt_match_indices",
                "new_gt_match_indices"
            )
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results


@PIPELINES.register_module()
class SeqFilterAnnotationsWithTrack(FilterAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self._filter(_results)
            outs.append(_results)



        return outs

    def _filter(self, results):
        assert "gt_bboxes" in results
        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = (
                "gt_bboxes",
                "gt_labels",
                "gt_masks",
                "gt_semantic_seg",
                "gt_match_indices",
                "new_gt_match_indices",
                'gt_instance_ids'
            )
            for key in keys:
                if key in results:
                    results[key] = np.array(results[key]) if not isinstance(results[key], np.ndarray) else results[key]
                    results[key] = results[key][keep]
            return results
