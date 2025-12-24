import re
import csv


def parse_log_file(file_path):
    # 存储解析结果的列表
    results = []

    # 打开并读取文件
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式分割不同的实验块
    experiment_blocks = re.findall(r'epoch\[(\d+)\], match score thres\[([0-9.]+)\]:(.*?)(?=epoch\[|\Z)', content,
                                   re.DOTALL)

    # 处理每个实验块
    for block in experiment_blocks:
        epoch = block[0]
        match_score_thres = block[1]
        data_block = block[2].strip().split('\n')

        # 跳过标题行
        if len(data_block) < 2:
            continue

        # 解析数据行
        headers = data_block[0].split()

        # 处理 Combined, Base, Novel 三种类型的数据
        for row in data_block[1:]:
            values = row.split()
            category = values[0]  # Combined, Base, 或 Novel

            # 创建结果字典
            result_dict = {
                'Epoch': epoch,
                'Match Score Thres': match_score_thres,
                'Category': category
            }

            # 添加各个指标
            for i, header in enumerate(headers[1:], 1):
                result_dict[header] = values[i]

            results.append(result_dict)

    return results


def save_to_csv(results, output_file):
    # 如果结果为空，直接返回
    if not results:
        print("No results to save.")
        return

    # 获取所有可能的键（列名）
    fieldnames = list(results[0].keys())

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入标题行
        writer.writeheader()

        # 写入数据行
        writer.writerows(results)

    print(f"Results saved to {output_file}")


# 主执行流程
def main():
    input_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_after/tao_train_dataset/debug_seq_tao/ours_detpro_training_association_256_adding_dynamic_motion_cls_fusion_module_add_gt_cycloss_new_inference_add_asso_ori_aux_loss_100bbox_035thres/tracker_match_score_search_eval_result.txt'  # 请替换为您的日志文件路径
    output_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_after/tao_train_dataset/debug_seq_tao/ours_detpro_training_association_256_adding_dynamic_motion_cls_fusion_module_add_gt_cycloss_new_inference_add_asso_ori_aux_loss_100bbox_035thres/tracker_match_score_search_eval_result.csv'

    # 解析日志文件
    parsed_results = parse_log_file(input_file)

    # 保存到CSV
    save_to_csv(parsed_results, output_file)

    # 额外打印结果预览
    print("\n结果预览:")
    for result in parsed_results[:5]:  # 打印前5条
        print(result)


# 如果直接运行此脚本
if __name__ == '__main__':
    main()


