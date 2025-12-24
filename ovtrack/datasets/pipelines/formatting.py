import torch
import numpy as np
from mmcv.parallel.data_container import DataContainer
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Collect, DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class SeqDefaultFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            _results["gt_match_indices"] = DC(to_tensor(_results["gt_match_indices"]))
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqDefaultFormatBundleWithTrack(object):
    """Sequence Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "img_metas", "proposals", "gt_bboxes", "gt_instance_ids",
    "gt_match_indices", "gt_bboxes_ignore", "gt_labels", "gt_masks",
    "gt_semantic_seg" and 'padding_mask'.
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_instance_ids: (1) to tensor, (2) to DataContainer
    - gt_match_indices: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor, \
                       (3) to DataContainer (stack=True)
    - padding_mask: (1) to tensor, (2) to DataContainer

    Args:
        ref_prefix (str): The prefix of key added to the second dict of input
            list. Defaults to 'ref'.
    """

    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        """Sequence Default formatting bundle call function.

        Args:
            results (list[dict]): List of two dicts.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle. Each key in the second dict of the input list
            adds `self.ref_prefix` as prefix.
        """
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            outs.append(_results)

        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f'{self.ref_prefix}_{k}'] = v

        return data
    def default_format_bundle(self, results):
        """Transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle.
        """
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'padding_mask' in results:
            results['padding_mask'] = DC(
                to_tensor(results['padding_mask'].copy()), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_instance_ids', 'gt_match_indices'
        ]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['img_metas', 'gt_masks']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
        if 'gt_semantic_seg' in results:
            semantic_seg = results['gt_semantic_seg']
            if len(semantic_seg.shape) == 2:
                semantic_seg = semantic_seg[None, ...]
            else:
                semantic_seg = np.ascontiguousarray(
                    semantic_seg.transpose(3, 2, 0, 1))
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg']), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__



@PIPELINES.register_module()
class VideoCollectWithTrack(object):
    """Collect data from the loader relevant to the specific task.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str]): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 keys,
                 meta_keys=None,
                 default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor',
                                    'flip', 'flip_direction', 'img_norm_cfg',
                                    'frame_id', 'is_video_data')):
        self.keys = keys
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.

        The keys in ``meta_keys`` and ``default_meta_keys`` will be converted
        to :obj:mmcv.DataContainer.

        Args:
            results (list[dict] | dict): List of dict or dict which contains
                the data to collect.

        Returns:
            list[dict] | dict: List of dict or dict that contains the
            following keys:

            - keys in ``self.keys``
            - ``img_metas``
        """
        results_is_dict = isinstance(results, dict)
        if results_is_dict:
            results = [results]
        outs = []
        for _results in results:
            _results = self._add_default_meta_keys(_results)
            _results = self._collect_meta_keys(_results)
            outs.append(_results)

        if results_is_dict:
            outs[0]['img_metas'] = DC(outs[0]['img_metas'], cpu_only=True)

        return outs[0] if results_is_dict else outs

    def _collect_meta_keys(self, results):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results['img_info']:
                img_meta[key] = results['img_info'][key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results



@PIPELINES.register_module()
class VideoCollect(Collect):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
            "frame_id",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys



@PIPELINES.register_module(force=True)
class SeqCollect(VideoCollect):
    def __init__(
        self,
        keys,
        ref_prefix="ref",
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.ref_prefix = ref_prefix
        self.meta_keys = meta_keys

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)

        assert len(outs) == 2
        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f"{self.ref_prefix}_{k}"] = v

        match_indices, ref_match_indices = self._match_gts(
            list(data["gt_match_indices"].data.numpy()),
            list(data["ref_gt_match_indices"].data.numpy()),
        )
        data["gt_match_indices"] = DataContainer(torch.tensor(match_indices))
        data["ref_gt_match_indices"] = DataContainer(torch.tensor(ref_match_indices))

        # repair the gt match indices bug by replacing it
        if 'new_gt_match_indices' in results[0]:
            new_key_match_indices, new_ref_match_indices = self._match_gts(
                list(results[0]['new_gt_match_indices']),
                list(results[1]['new_gt_match_indices']),
            )
            # used to debug
            outs_old = results
            key_match_label = [outs_old[1]['gt_labels'].data[index].item() == outs_old[0]['gt_labels'].data[i].item() if index!= -1 else -1 for i, index in enumerate(new_key_match_indices)]
            ref_match_label = [outs_old[0]['gt_labels'].data[index].item() == outs_old[1]['gt_labels'].data[i].item() if index!= -1 else -1 for i, index in enumerate(new_ref_match_indices)]
            false_counter = 0
            for val in key_match_label:
                if val == False:
                    false_counter += 1
            assert false_counter == 0, "wrong match gt pair"

            data["gt_match_indices"] = DataContainer(torch.tensor(new_key_match_indices))
            data["ref_gt_match_indices"] = DataContainer(torch.tensor(new_ref_match_indices))



        return data

    def _match_gts(self, inds, ref_inds):

        match_indices = np.array(
            [ref_inds.index(i) if (i in ref_inds and i != -1) else -1 for i in inds]
        )
        ref_match_indices = np.array(
            [inds.index(i) if (i in inds and i != -1) else -1 for i in ref_inds]
        )
        return match_indices, ref_match_indices



#
# @PIPELINES.register_module(force=True)
# class SeqCollect(VideoCollect):
#     def __init__(
#         self,
#         keys,
#         ref_prefix="ref",
#         meta_keys=(
#             "filename",
#             "ori_filename",
#             "ori_shape",
#             "img_shape",
#             "pad_shape",
#             "scale_factor",
#             "flip",
#             "flip_direction",
#             "img_norm_cfg",
#         ),
#     ):
#         self.keys = keys
#         self.ref_prefix = ref_prefix
#         self.meta_keys = meta_keys
#
#     def __call__(self, results):
#         outs = []
#         for _results in results:
#             _results = super().__call__(_results)
#             outs.append(_results)
#
#         assert len(outs) == 2
#         data = {}
#         data.update(outs[0])
#         for k, v in outs[1].items():
#             data[f"{self.ref_prefix}_{k}"] = v
#
#         match_indices, ref_match_indices = self._match_gts(
#             list(data["gt_match_indices"].data.numpy()),
#             list(data["ref_gt_match_indices"].data.numpy()),
#         )
#         data["gt_match_indices"] = DataContainer(torch.tensor(match_indices))
#         data["ref_gt_match_indices"] = DataContainer(torch.tensor(ref_match_indices))
#         return data
#
#     def _match_gts(self, inds, ref_inds):
#
#         match_indices = np.array(
#             [ref_inds.index(i) if i in ref_inds else -1 for i in inds]
#         )
#         ref_match_indices = np.array(
#             [inds.index(i) if i in inds else -1 for i in ref_inds]
#         )
#         return match_indices, ref_match_indices

@PIPELINES.register_module(force=True)
class SeqCollectNoPair(VideoCollect):
    def __init__(
            self,
            keys,
            meta_keys=(
                    "filename",
                    "ori_filename",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "img_norm_cfg",
            ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        img_meta_list = []
        img_list = []
        for _results in results:
            _results = super().__call__(_results)
            img_meta_list.append(_results['img_metas'])
            img_list.append(_results['img'])
        result = {'img': img_list, 'img_metas': img_meta_list}
        return result
