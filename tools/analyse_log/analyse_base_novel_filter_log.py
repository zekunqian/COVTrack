import re
import csv


def parse_metrics_file(file_path):
    results = []

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 分割成不同的max_instance_per_cate块
    blocks = content.split('max_instance_per_cate:')
    blocks = [b.strip() for b in blocks if b.strip()]  # 移除空块

    for block in blocks:
        # 提取max_instance_per_cate值
        first_line = block.split('\n')[0]
        max_instance = re.search(r'(\d+)', first_line).group(1)

        # 提取指标行
        lines = block.split('\n')
        headers = [h.strip() for h in lines[1].split() if h.strip()]

        # 处理Combined, Base, Novel数据行
        for line in lines[2:5]:  # 只处理这三行
            values = [v.strip() for v in line.split() if v.strip()]
            if not values:
                continue

            category = values[0]
            metrics = values[1:]

            # 创建结果字典
            result = {
                'max_instance_per_cate': max_instance,
                'category': category
            }

            # 添加所有指标
            for header, value in zip(headers[1:], metrics):  # 跳过第一个header (TETA50:)
                result[header] = value

            results.append(result)

    return results


def save_to_csv(results, output_file):
    if not results:
        print("没有数据要保存")
        return

        # 获取所有列名
    fieldnames = ['max_instance_per_cate', 'category'] + list(results[0].keys())[2:]

    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"结果已保存到 {output_file}")


def main():
    input_file = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/debug/only_base_filter.txt'  # 替换为你的输入文件路径
    output_file = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/debug/only_base_filter.csv'  # 替换为你想要的输出文件路径

    # 解析文件
    results = parse_metrics_file(input_file)

    # 保存为CSV
    save_to_csv(results, output_file)

    # 打印预览
    print("\n结果预览:")
    for result in results[:3]:  # 显示前3条结果
        print(result)

    print(f"\n总共解析了 {len(results)} 条结果")


if __name__ == '__main__':
    main()