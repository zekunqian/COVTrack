import re
import csv

def parse_log_file(input_file, output_file):
    # 用于存储所有数据的列表
    data = []

    # 读取日志文件
    with open(input_file, 'r') as f:
        content = f.read()

    # 解析每个实验块（现在只用一个空行分割）
    experiments = content.strip().split('\n\n')

    for exp in experiments:
        lines = exp.strip().split('\n')

        # 从第一行提取epoch和match score信息
        header = lines[0]
        epoch = re.search(r'epoch\[(\d+)\]', header).group(1)
        match_score = re.search(r'match score thres\[([0-9.]+)\]', header).group(1)

        # 解析数据行
        for line in lines[2:]:  # 跳过表头行
            if not line.strip():
                continue

            values = line.split()
            data_type = values[0]  # Combined/Base/Novel
            metrics = values[1:]  # 所有指标值

            # 组合一行数据（移除了max_bbox）
            row = [epoch, match_score, data_type] + metrics
            data.append(row)

    # 写入CSV文件（更新headers移除max_bbox）
    headers = ['epoch', 'match_score_thres', 'TETA50_type',
               'TETA', 'LocS', 'AssocS', 'ClsS', 'LocRe', 'LocPr',
               'AssocRe', 'AssocPr', 'ClsRe', 'ClsPr']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

# 使用示例
input_file = '/data1/clark/models/ovtrack/resutls/work_dirs/self_train_fintune/new_spatial_train/assignment_way_01_top10_095thres/match_score_search_eval_result_new.txt'  # 输入的日志文件名
output_file = '/data1/clark/models/ovtrack/resutls/work_dirs/self_train_fintune/new_spatial_train/assignment_way_01_top10_095thres/match_score_search_eval_result_new.csv'  # 输出的CSV文件名
parse_log_file(input_file, output_file)