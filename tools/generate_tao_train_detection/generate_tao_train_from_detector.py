import os
import json
import tqdm


# make the directory:

detection_dir = '/data1/clark/dataset/openDomain/TAO/tao/annotations/generated_detection_results/vovtrack'
saved_dir_path = '/data1/clark/dataset/openDomain/TAO/tao/annotations/generated_detection_results/vovtrack_detector.txt'
json_files = sorted(os.listdir(detection_dir))


def save_dict_to_json(data, filename):
    """
    将字典保存为JSON文件
    :param data: 要保存的字典
    :param filename: 文件名（包括路径）
    """
    # 确保目录存在
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

        # 将字典写入JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已成功保存到 {filename}")


complete_detection_dict = {}

for json_file in tqdm.tqdm(json_files):
    complete_json_file = os.path.join(detection_dir, json_file)

    with open(complete_json_file, 'r') as file:
        # 将JSON 数据加载为 Python 对象
        try:
            data = json.load(file)
            img_filename = data[0]['filename']
            json_filename = os.path.split(complete_json_file)[-1]
            with open(saved_dir_path, 'a')  as f:
                f.write(f"{img_filename} {json_filename}\n")
        except Exception as e:
            print(f"the problem one is {file}")


