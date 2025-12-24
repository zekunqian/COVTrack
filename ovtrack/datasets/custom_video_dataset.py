import mmcv
import os
import os.path as osp
import copy
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv import Config
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from .pipelines import (LoadMultiImagesFromFile, SeqCollect,
                        SeqDefaultFormatBundle, SeqLoadAnnotations,
                        SeqNormalize, SeqPad, SeqRandomFlip, SeqResize)
from .seq_multi_image_mix_dataset import SeqMultiImageMixDataset
import random
from mmdet.datasets import build_dataloader, build_dataset

def random_crop(frames, crop_size=8, long_short_mode=False, long_short_length=16):
    # 确保列表至少有 crop_size 个元素
    if len(frames) < crop_size:
        raise ValueError("List has less than {} elements.".format(crop_size))

    # 随机选择截取的起始位置
    start_index = random.randint(0, len(frames) - crop_size)

    if not long_short_mode:
    # 截取连续的 crop_size 帧
        cropped_frames = frames[start_index:start_index + crop_size]
    else:
        cropped_frames = []
        # simple divide into two sequence
        sub_crop_size = crop_size//2
        sub_cropped_frame1 = frames[start_index:start_index + sub_crop_size]

        new_start_index = start_index+sub_crop_size
        new_start_index = random.randint(new_start_index, min(len(frames) - sub_crop_size, start_index+long_short_length-sub_crop_size))
        sub_cropped_frame2 = frames[new_start_index: new_start_index + sub_crop_size]

        cropped_frames.extend(sub_cropped_frame1)
        cropped_frames.extend(sub_cropped_frame2)



    return cropped_frames


@DATASETS.register_module()
class CustomVideoDataset(CustomDataset):
    def __init__(self, seq_length=8, min_seq_length=8, crop_size_ratio=4, long_short_mode=False, long_short_length=-1, *args, **kwargs):
        # kwargs['ann_file'] = '/data1/clark/dataset/LV-VIS/train/JPEGImages'
        # ann_info_template = {'bboxes': [[438.81, 106, 481.66, 216.49]], 'bboxes_ignore': [], 'instance_ids': [2188249], 'labels': [1073], 'masks': [], 'match_indices': [0], 'seg_map': '000000252625.png'}
        # file path use absolutaion path
        self.min_seq_length = min_seq_length
        self.ann_info_template = {'bboxes': np.array([]), 'instance_ids': np.array([]), 'labels': np.array([]), 'masks': np.array([]), 'match_indices': np.array([]), 'seg_map': '000000252625.png'}
        # self.ann_info_template = {'bboxes': [], 'bboxes_ignore': [], 'instance_ids': [], 'labels': [], 'masks': [], 'match_indices': [], 'seg_map': '000000252625.png'}
        self.img_info_template = {'flickr_url': 'http://farm1.staticflickr.com/46/188068269_52967f417f_z.jpg', 'id': 774096, 'neg_category_ids': [236, 173, 815, 653, 37, 771, 356, 975, 447, 385, 495, 513], 'not_exhaustive_category_ids': [595, 1074], 'width': 500, 'license': 3, 'coco_url': 'http://images.cocodataset.org/train2017/000000252625.jpg', 'date_captured': '2013-11-15 04:45:45', 'height': 333, 'file_name': '000000252625.jpg', 'video_id': 195470, 'frame_id': 0, 'filename': '000000252625.jpg'}
        # self.prepare_dict_template = {'img_prefix':"", 'seg_prefix' : None, 'proposal_file': None, 'bbox_fields' : [], 'mask_fields':[], 'seg_fields':[], 'frame_id':0}
        self.prepare_dict_template = {'seg_prefix' : None, 'proposal_file': None, 'bbox_fields' : [], 'mask_fields':[], 'seg_fields':[], 'frame_id':0}
        self.img_prefix = None # img_collections_dir + img_prefix + (video_list) + img_inner
        self.img_inner = None
        self.seq_length = seq_length
        self.min_length = 99999999
        self.crop_size_ratio = crop_size_ratio
        self.long_short_mode = long_short_mode
        self.long_short_length = long_short_length
        # LoadImageFromFile can analyse the shape of image
        super(CustomVideoDataset, self).__init__(*args, **kwargs)

        # self.ann_file = '/data1/clark/dataset/LV-VIS/train/JPEGImages'

    def load_annotations(self, ann_file):
        # 返回一个包含视频路径的列表
        if isinstance(ann_file, str):
            ann_file = [ann_file]
        # if self.img_prefix is not None:
        #     ann_file = os.path.join(ann_file, self.img_prefix)
        video_list = [os.path.join(ann_file[i], video_dir) for i in range(len(ann_file)) for video_dir in os.listdir(ann_file[i]) ]
        data_infos = []
        min_video_path = 'None'
        for video_path in video_list:
            video_frames = []
            # 获取视频文件夹下的所有图片
            img_list = sorted(
                # [osp.join(video_path, img) for img in os.listdir(video_path) if img.endswith('.jpg')]
                [osp.join(video_path, img) for img in os.listdir(video_path)]
            )

            for img_path in img_list:
                video_frames.append(img_path)
            # filter the too short videos:
            if len(video_frames) <= self.min_seq_length:
                continue
            if len(video_frames) < self.min_length:
                self.min_length = min(self.min_length, len(video_frames))
                min_video_path = video_path

            # split the too long sequence
            if len(img_list)  >= self.seq_length * self.crop_size_ratio:
                dir_path, _ = os.path.split(img_list[0])
                group_num = len(img_list) // (self.seq_length * self.crop_size_ratio)
                size = self.seq_length * self.crop_size_ratio
                # 使用列表切片将列表分成 n 份
                split_img_list = [img_list[i * size:(i + 1) * size] for i in range(group_num)]
                for img_list in split_img_list:
                    info = dict(video_frames=img_list)
                    # template
                    info.update(dict(width=640, height=480))
                    info['video'] = video_path
                    data_infos.append(info)

            else:
                info = dict(video_frames=video_frames)
                # template
                info.update(dict(width=640, height=480))
                info['video'] = video_path
                data_infos.append(info)
        print(f" The min length of seq is {self.min_length} in {min_video_path}, dataset length: {len(data_infos)}")
        return data_infos

    def get_ann_info(self, idx):
        # 重写此方法以返回注释信息
        return {}

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        video_info = self.data_infos[idx]

        frame_list = random_crop(video_info['video_frames'], min(self.seq_length, len(video_info['video_frames'])), self.long_short_mode, self.long_short_length)

        # results = self.prepare_results(img_info)
        # ref_results = self.prepare_results(ref_img_info)

        # if self.match_gts:
        #     results, ref_results = self.match_results(results, ref_results)
        #     nomatch = (results["ann_info"]["match_indices"] == -1).all()
        #     if self.skip_nomatch_pairs and nomatch:
        #         return None
        result_list = []
        for frame in frame_list:
            img_info = copy.deepcopy(self.img_info_template)
            img_info['file_name'] = frame
            img_info['filename'] = frame
            ann_info = copy.deepcopy(self.ann_info_template)
            ann_info['seg_map'] = frame
            result = dict(img_info = img_info, ann_info = ann_info)
            # padding
            result['img_prefix'] = ''
            result.update(copy.deepcopy(self.prepare_dict_template))
            result_list.append(result)


        # self.pre_pipeline([results, ref_results])
        # if isinstance(results, list):
        #     for _results in results:
        #         self._pre_pipeline(_results)
        # elif isinstance(results, dict):
        #     self._pre_pipeline(results)
        # else:
        #     raise TypeError("input must be a list or a dict")

        return self.pipeline(result_list)

    # def __getitem__(self, idx):
    #     # 重写此方法以获取数据
    #     data_info = self.data_infos[idx]
    #     video_frames = data_info['video_frames']
    #
    #     # 从每个视频中获取连续的五帧
    #     start_idx = np.random.randint(0, len(video_frames) - 5)
    #     img_infos = video_frames[start_idx:start_idx + 5]
    #
    #     # 读取图像
    #     img_list = [mmcv.imread(img_info) for img_info in img_infos]
    #
    #     # 构建数据字典
    #     results = dict(
    #         img_list=img_list,
    #         img_prefix='',
    #         img_info=data_info
    #     )
    #
    #     return self.pipeline(results)

if __name__ == '__main__':
    config = '/home/clark/workspace2/ovtrack/configs/ovtrack-teta/ovtrack_r50_self_train.py'
    cfg = Config.fromfile(config)

    dataset = CustomVideoDataset(pipeline=cfg.train_pipeline)
    # dataset.prepare_train_img(1)
    # 4. 创建 DataLoader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=3,
        workers_per_gpu=4,
        dist=False,
        shuffle=True
    )
    for i, data in enumerate(data_loader):
        print('hello world')