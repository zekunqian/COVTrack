import re
import csv


def parse_log_file(file_path):
    # 存储解析结果的列表
    results = []

    # 打开并读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 更复杂的正则表达式，匹配配置文件路径和后续的实验数据
    # 使用非贪婪匹配和更灵活的分组
    pattern = r'(work_dirs/[^\n]+\.py)\n(.*?)(?=work_dirs/[^\n]+\.py|\Z)'
    blocks = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

    # 打印找到的配置文件和块数量
    print(f"找到 {len(blocks)} 个配置文件块")

    # 处理每个实验块
    for config_path, experiment_data in blocks:
        print(f"处理配置文件: {config_path}")

        # 使用正则表达式提取每个epoch的数据
        epoch_blocks = re.findall(r'epoch\[(\d+)\]:(.*?)(?=epoch\[|\Z)', experiment_data, re.DOTALL)

        print(f"在此配置文件中找到 {len(epoch_blocks)} 个epoch块")

        # 处理每个epoch的数据
        for epoch, data_block in epoch_blocks:
            # 解析数据行
            data_lines = data_block.strip().split('\n')

            # 跳过没有数据的块
            if len(data_lines) < 4:
                continue

            headers = data_lines[0].split()

            # 处理 Combined, Base, Novel 三种类型的数据
            for row in data_lines[1:]:
                # 去除多余的空白
                values = row.split()

                # 跳过空行或格式不正确的行
                if len(values) < 2:
                    continue

                category = values[0]  # Combined, Base, 或 Novel

                # 创建结果字典
                result_dict = {
                    'Config File': config_path,
                    'Epoch': epoch,
                    'Category': category
                }

                # 添加各个指标
                for i, header in enumerate(headers[1:], 1):
                    # 确保有足够的值
                    if i < len(values):
                        result_dict[header] = values[i]
                    else:
                        result_dict[header] = 'N/A'

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
    # input_file = '/data/clark/models/ovtrack/tao_train_dataset/seq_tao_base_with_modify/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_2xdata_30_frame_range_load_from_detpro_ori_all_aug/eval_result.txt'  # 请替换为您的日志文件路径
    # output_file = '/data/clark/models/ovtrack/tao_train_dataset/seq_tao_base_with_modify/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_2xdata_30_frame_range_load_from_detpro_ori_all_aug/eval_result.csv'
    input_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_repair_base_cls/lvis_pair/association_256_no_gt_cycloss_80bbox_035thres_load_from_detpro_reverse/eval_result.txt'  # 请替换为您的日志文件路径
    output_file = '/data1/clark/models/ovtrack/resutls/work_dirs/VOVTrack_repair_base_cls/lvis_pair/association_256_no_gt_cycloss_80bbox_035thres_load_from_detpro_reverse/eval_result.csv'

    # 解析日志文件
    parsed_results = parse_log_file(input_file)

    # 保存到CSV
    save_to_csv(parsed_results, output_file)

    # 额外打印结果预览
    print("\n结果预览:")
    for result in parsed_results[:10]:  # 打印前10条
        print(result)

    # 打印总结信息
    print(f"\n总共解析了 {len(parsed_results)} 条结果")


# 如果直接运行此脚本
if __name__ == '__main__':
    main()

