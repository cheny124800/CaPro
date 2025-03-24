import os


def keep_top_k_lines_in_txt_files(directory, k):
    # 遍历目录中的所有txt文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    # 读取文件的前k行
                    lines = file.readlines()

                # 保留前k行
                lines_to_keep = lines[:k]

                # 重新写入文件，仅保留前k行
                with open(file_path, 'w') as file:
                    file.writelines(lines_to_keep)

                print(f"Updated {filename}, kept top {k} lines.")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


# 使用示例：指定文件夹路径和k值
directory_path = r"C:\Users\86181\Desktop\sim2teal\test\top5"  # 目录路径
k_value = 5 # 保留前10行
keep_top_k_lines_in_txt_files(directory_path, k_value)
