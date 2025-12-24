import re
import csv


# def parse_log_file(input_file, output_file):
#     # 用于存储所有数据的列表
#     data = []
#
#     # 读取日志文件
#     with open(input_file, 'r') as f:
#         content = f.read()
#
#     # 解析每个实验块
#     experiments = content.strip().split('\n\n\n')
#
#     for exp in experiments:
#         lines = exp.strip().split('\n')
#
#         # 从第一行提取epoch和参数信息
#         header = lines[0]
#         epoch = re.search(r'epoch\[(\d+)\]', header).group(1)
#         match_score = re.search(r'match score thres\[([0-9.]+)', header).group(1)
#         max_bbox = re.search(r'max bbox \[([0-9.]+)\]', header).group(1)
#
#         # 解析数据行
#         for line in lines[2:]:  # 跳过表头行
#             if not line.strip():
#                 continue
#
#             values = line.split()
#             data_type = values[0]  # Combined/Base/Novel
#             metrics = values[1:]  # 所有指标值
#
#             # 组合一行数据
#             row = [epoch, match_score, max_bbox, data_type] + metrics
#             data.append(row)
#
#     # 写入CSV文件
#     headers = ['epoch', 'match_score_thres', 'max_bbox', 'TETA50_type',
#                'TETA', 'LocS', 'AssocS', 'ClsS', 'LocRe', 'LocPr',
#                'AssocRe', 'AssocPr', 'ClsRe', 'ClsPr']
#
#     with open(output_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)
#         writer.writerows(data)

def parse_log_file(input_file, output_file):
    # 用于存储所有数据的列表
    data = []

    # 读取日志文件
    with open(input_file, 'r') as f:
        content = f.read()

        # 解析每个实验块
    experiments = content.strip().split('\n\n')

    for exp in experiments:
        lines = exp.strip().split('\n')

        # 检查是否是有效的实验块
        if not lines or not lines[0].startswith('epoch'):
            continue

            # 从第一行提取epoch和参数信息
        header = lines[0]
        epoch = re.search(r'epoch\[(\d+)\]', header).group(1)
        match_score = re.search(r'match score thres\[([0-9.]+)', header).group(1)
        max_bbox = re.search(r'max bbox \[([0-9.]+)\]', header).group(1)

        # 解析数据行
        for line in lines[2:]:  # 跳过表头行
            if not line.strip() or line.startswith('TETA50:'):
                continue

                # 将连续的空格替换为单个空格，然后分割
            parts = ' '.join(line.split()).split()
            if len(parts) != 11:  # 确保有正确数量的列
                continue

            data_type = parts[0]  # Combined/Base/Novel
            values = [float(x) for x in parts[1:]]  # 转换所有数值为float

            # 组合一行数据
            row = [epoch, match_score, max_bbox, data_type] + values
            data.append(row)

            # 写入CSV文件
    headers = ['epoch', 'match_score_thres', 'max_bbox', 'TETA50_type',
               'TETA', 'LocS', 'AssocS', 'ClsS', 'LocRe', 'LocPr',
               'AssocRe', 'AssocPr', 'ClsRe', 'ClsPr']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    # 使用示例
input_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_repair_base_cls/finetune/assignment_way_01_top10_095thres_new_high_04memo_chkpoint/tracker_match_score_max_bbox_search_eval_result.txt'  # 输入的日志文件名
output_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_repair_base_cls/finetune/assignment_way_01_top10_095thres_new_high_04memo_chkpoint/tracker_match_score_max_bbox_search_eval_result.csv'  # 输出的CSV文件名
# input_file = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_5xdata_30_frame_range_load_from_detpro_ori_all_aug/tracker_match_score_max_bbox_search_eval_result.txt'  # 输入的日志文件名
# output_file = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_5xdata_30_frame_range_load_from_detpro_ori_all_aug/tracker_match_score_max_bbox_search_eval_result.csv'  # 输出的CSV文件名
parse_log_file(input_file, output_file)