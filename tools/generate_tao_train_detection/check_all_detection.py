import os
import glob
from tqdm import tqdm

detection_dir = '/data1/clark/dataset/openDomain/TAO/tao/annotations/generated_detection_results/vovtrack'
saved_dir_path = '/data1/clark/dataset/openDomain/TAO/tao/annotations/generated_detection_results/vovtrack_detector.txt'

def get_image_paths(folder_path):
    # 定义常见的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')

    # 使用集合来存储图片路径，以避免重复
    image_paths = set()

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否是图片
            if file.lower().endswith(image_extensions):
                # 构建完整的文件路径并添加到集合中
                new_root = os.path.join('data/tao', root[root.index('frames/train/'):])
                full_path = os.path.join(new_root, file)
                image_paths.add(full_path)

    return image_paths

# get all the images
train_dataset_dir = '/data1/clark/dataset/openDomain/TAO/tao/frames/train'
img_path_set = get_image_paths(train_dataset_dir)


new_path_set = set()

file_name2json_name = dict()
with open(saved_dir_path, 'r') as f:
    for line in f:
        file_name, json_name = line.split()
        new_path_set.add(file_name)

"""
 problem frames are
    data/tao/frames/train/Charades/Z6HEA/frame0491.jpg 
    data/tao/frames/train/YFCC100M/v_5d2225140e1aa84f7419b3c9871e7bf/frame0851.jpg 
"""

print()



