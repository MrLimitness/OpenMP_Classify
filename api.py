import os
from openai import OpenAI

# ====== 参数设置 ======
client = OpenAI(
    api_key='sk-lxmpctywfxndtshonqgixypnockmmtaaqcfzcxilysircimv',
    base_url='https://api.siliconflow.cn/v1'
)

PROMPT_TEMPLATE = (
    "你好，你现在是一个openmp原语小专家，现在呢，我会扔给你一个openmp的指令，你需要做以下几件事："
    "1.生成一个能用到这个指令的c程序片段，最好这个片段是能用到并行化的片段（例如矩阵乘法，别的如果能运行10秒也行，"
    "你得自己编，能用矩阵乘法最好还是用他好了），并且呢运行时间最好在10s左右，时间长效果明显一点，这个片段作为串行版本 "
    "2.在这个串行程序片段的基础上，在适当的位置加上parallel for，提升其运行效率，减少运行时间，这是parallel for版本"
    "3.在这个串行程序片段的基础上，在适当的地方应用我扔给你的这个openmp的指令，可以结合其他合适的指令（甚至可以是parallel for，"
    "但这个指令里必须用到我扔给你的这个指令），想办法提升程序的运行效率，减少运行的时间，这个是版本3、4、5、6，"
    "也就是说我扔给你的这个指令你可以根据他的用法灵活运用多个指令，但最好也不要太多，4个以内就够啦。然后呢，还有一个要求，"
    "这三个版本生成后，请写一个完整的脚本，这几个版本写在一个脚本里就行了，每个版本的运行时间都要输出，然后以串行版本为基础计算加速比，"
    "打印输出每一个版本的加速比，最后呢，这个脚本是可以自主设置线程数的比如（16/64/144线程，我自己设置就好，"
    "请你给一个可以改线程的东西就行），好的，请输出这个脚本供我复制，记住，一定要是一个完整的脚本，我只用一次复制！"
    "生成的代码打印输出的东西一定要是中文！只用输出脚本就行，别的解释性的东西一样都不要输出！这个openmp的指令是：{}"
)

API_MODEL = "Pro/deepseek-ai/DeepSeek-R1"

def clean_code_block(text):
    # 去掉开头是 c``` 或 ```c 或 ``` 的代码块标记
    if text.startswith("c```"):
        text = text[4:]
    elif text.startswith("```c"):
        text = text[4:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ====== 主逻辑 ======
def process_example_files(root_dir):
    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('_example.c'):
                    file_path = os.path.join(subfolder_path, file)

                    # 如果文件不为空，则跳过
                    if os.path.getsize(file_path) > 0:
                        print(f"跳过已存在内容的文件: {file}")
                        continue

                    keyword = file.replace('_example.c', '')
                    prompt = PROMPT_TEMPLATE.format(keyword)
                    print(f"处理文件: {file}，关键词: {keyword}")

                    try:
                        response = client.chat.completions.create(
                            model=API_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        result = response.choices[0].message.content
                        result = clean_code_block(result)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(result)
                        print(f"写入完成: {file_path}\n")

                    except Exception as e:
                        print(f"调用 OpenAI API 失败: {e}\n")

if __name__ == '__main__':
    folder = "OpenMP_Classify"
    if os.path.isdir(folder):
        process_example_files(folder)
        print("全部处理完成。")
    else:
        print("路径无效，请检查输入。")
