import os

# 要生成的关键词列表
keywords = [
    "no_openmp",
    "no_parallelism",
    "notinbranch",
    "inbranch"
]
# 目标文件夹路径（可根据需要修改）
target_folder = "OpenMP_Classify/9"

# 如果文件夹不存在则创建
os.makedirs(target_folder, exist_ok=True)

# 遍历关键词生成对应的空文件
for keyword in keywords:
    filename = f"{keyword}_example.c"
    filepath = os.path.join(target_folder, filename)
    with open(filepath, 'w') as f:
        pass  # 创建空文件

print(f"已在 '{target_folder}' 文件夹中生成 {len(keywords)} 个示例文件。")
