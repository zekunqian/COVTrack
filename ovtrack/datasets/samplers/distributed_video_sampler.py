import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler


# class DistributedVideoSampler(_DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank)
#         self.shuffle = shuffle
#         assert not self.shuffle, "Specific for video sequential testing."
#         self.num_samples = len(dataset)
#
#         first_frame_indices = []
#         for i, img_info in enumerate(self.dataset.data_infos):
#             if img_info["frame_id"] == 0:
#                 first_frame_indices.append(i)
#
#         chunks = np.array_split(first_frame_indices, num_replicas)
#         split_flags = [c[0] for c in chunks]
#         split_flags.append(self.num_samples)
#
#         self.indices = [
#             list(range(split_flags[i], split_flags[i + 1]))
#             for i in range(self.num_replicas)
#         ]
#
#     def __iter__(self):
#         indices = self.indices[self.rank]
#         return iter(indices)


class DistributedVideoSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, "Specific for video sequential testing."
        self.num_samples = len(dataset)

        first_frame_indices = []
        for i, img_info in enumerate(self.dataset.data_infos):
            if img_info["frame_id"] == 0:
                first_frame_indices.append(i)

        # chunks = np.array_split(first_frame_indices, num_replicas)
        # split_flags = [c[0] for c in chunks]
        # split_flags.append(self.num_samples)
        split_flags = self.get_balanced_splits(first_frame_indices, self.num_replicas, self.num_samples)

        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]

    @staticmethod
    def get_balanced_splits(first_frame_indices, num_replicas, total_samples):
        # 确保第一个索引是0
        assert first_frame_indices[0] == 0, "First index should be 0"

        # 计算每个GPU理想情况下应该处理的样本数
        target_size = total_samples // num_replicas

        split_flags = [0]  # 第一个分割点
        current_target = target_size

        # 遍历frame_indices，找到最接近目标值的分割点
        for i in range(1, num_replicas):
            target_position = i * target_size

            # 找到最接近目标位置的帧起始点
            closest_idx = min(range(len(first_frame_indices)),
                              key=lambda x: abs(first_frame_indices[x] - target_position))

            split_flags.append(first_frame_indices[closest_idx])

            # 添加末尾
        split_flags.append(total_samples)

        return split_flags

    def __iter__(self):
        indices = self.indices[self.rank]
        return iter(indices)