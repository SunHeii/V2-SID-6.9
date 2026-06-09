import pandas as pd
import re
import os
from langdetect import detect, LangDetectException
import cld3

# ================= 配置路径 =================
# 替换为您本地的 Yelp 数据集路径
BUSINESS_JSON = "/home/mysjz/mywork/sentiment/yelp/yelp_academic_dataset_business.json"
REVIEW_JSON = "/home/mysjz/mywork/sentiment/yelp/yelp_academic_dataset_review.json"

# 输出文件路径：元数据存 CSV，纯文本存 TXT
OUTPUT_META_CSV = "/home/mysjz/mywork/V2-SID/yelp_interaction_base.csv"
OUTPUT_TEXT_TXT = "/home/mysjz/mywork/V2-SID/train.txt"


# ================= 辅助函数 =================
def clean_text(text):
    """
    清理文本：去除 HTML 标签、多余的换行符和连续空格
    确保每条评论严格占据单行，防止写入 txt 时产生错位！
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)  # 去除可能存在的少量 HTML 标签
    text = re.sub(r'[\r\n]+', ' ', text)  # 绝对禁止换行符，全替换为空格
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    return text.strip()


def is_english(text):
    """
    使用 langdetect 检测文本是否为英文
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


# ================= 主流程 =================
def process_stage_1_and_2_split():
    print("🚀 开始第一阶段：数据读取与融合...")

    # 1. 读取并过滤 Business 数据
    print("正在加载商户数据 (business.json)...")
    df_biz = pd.read_json(BUSINESS_JSON, lines=True)
    df_biz = df_biz[['business_id', 'latitude', 'longitude', 'categories']]

    # 仅保留餐饮类商户，提前降低内存压力
    df_biz = df_biz.dropna(subset=['categories'])
    df_biz = df_biz[df_biz['categories'].str.contains('Restaurant|Food', case=False)]
    print(f"✅ 筛选出 {len(df_biz)} 家餐饮类商户。")

    # ================= 核心修改区域 开始 =================

    # 2. 分块读取 Review 数据并即时过滤 (解决 137 内存溢出)
    print("正在加载评论数据 (review.json)，启动分块读取防内存溢出... ⏳")

    # 提取合法的餐饮 business_id 集合，利用 set 的 O(1) 极速查询
    valid_business_ids = set(df_biz['business_id'])

    chunk_size = 100000  # 每次只读 10 万行，吃进去一口，消化一口
    filtered_chunks = []

    # 使用 chunksize 迭代读取大文件
    for chunk in pd.read_json(REVIEW_JSON, lines=True, chunksize=chunk_size):
        # A. 仅保留需要的列，提前瘦身
        chunk = chunk[['review_id', 'user_id', 'business_id', 'date', 'text']]
        # B. 核心：即时丢弃非餐饮商户的评论！只保留 valid_business_ids 里的
        chunk = chunk[chunk['business_id'].isin(valid_business_ids)]
        # C. 把过滤后极小的数据块存起来
        filtered_chunks.append(chunk)

    # 将所有保留下来的小数据块拼成一个 DataFrame
    df_rev = pd.concat(filtered_chunks, ignore_index=True)
    print(f"✅ 分块读取完毕！成功提取到 {len(df_rev)} 条餐饮类相关评论。")

    # 3. 数据融合 (Inner Join)
    print("正在进行时空数据与评论数据的 Join 操作...")
    df_merged = pd.merge(df_rev, df_biz, on='business_id', how='inner')

    # 彻底清空临时变量，释放内存
    del df_biz, df_rev, filtered_chunks, valid_business_ids
    print(f"✅ 成功融合数据，当前剩余 {len(df_merged)} 条交互记录。")

    # 4. 文本清洗与长度截断
    print("正在清洗文本和统计字数 (强制单行化)...")
    df_merged['text'] = df_merged['text'].apply(clean_text)
    df_merged['word_count'] = df_merged['text'].apply(lambda x: len(x.split()))

    df_merged = df_merged[(df_merged['word_count'] >= 10) & (df_merged['word_count'] <= 500)]
    print(f"✅ 长度过滤后剩余 {len(df_merged)} 条评论。")

    # 5. 语言过滤
    print("正在执行语言检测 (剔除非英文评论)... ⏳")
    # 如果全量测试太慢，可以取消下面这行的注释进行小批量抽样测试：
    # df_merged = df_merged.sample(n=50000, random_state=42)
    df_merged['is_en'] = df_merged['text'].apply(is_english)
    df_merged = df_merged[df_merged['is_en']]

    # 6. 重命名与列规整 (对齐 SID 格式)
    df_merged = df_merged.rename(columns={
        'user_id': 'uid',
        'business_id': 'pid',
        'date': 'time'
    })

    # ================= 核心修改：数据分流输出 =================
    print("\n📦 开始分离数据并保存，确保严格对应关系...")

    # 为了保险，重置 index，确保顺序被固定下来
    df_merged = df_merged.reset_index(drop=True)

    # 任务 A：保存结构化时空特征 (不带文本)
    meta_cols = ['review_id', 'uid', 'pid', 'categories', 'latitude', 'longitude', 'time']
    df_meta = df_merged[meta_cols]
    df_meta.to_csv(OUTPUT_META_CSV, index=False)
    print(f"✅ 结构化交互数据已保存至: {OUTPUT_META_CSV}")

    # 任务 B：保存纯文本到 train.txt 供 SA-2 抽取
    # 模式为 'w'，按 df_merged 的顺序一行一行写入
    # with open(OUTPUT_TEXT_TXT, 'w', encoding='utf-8') as f:
    #     for text in df_merged['text']:
    #         f.write(text + '\n')
    # 任务 B：保存纯文本 (供 SA-2 训练，显式注入 ID)
    print("正在保存带有 Review_ID 前缀的 train.txt...")
    with open(OUTPUT_TEXT_TXT, 'w', encoding='utf-8') as f:
        # 使用 iterrows 遍历，同时拿到 review_id 和 text
        for _, row in df_merged.iterrows():
            r_id = row['review_id']  # 提取唯一的 review_id
            text = row['text']
            # 格式化为：[ID] 文本内容
            f.write(f"[{r_id}] {text}\n")

    print(f"✅ 纯文本评论已单行分离并保存至: {OUTPUT_TEXT_TXT}")

    print(f"\n🎉 阶段 1 & 2 完成！最终高质量交互数量: {len(df_merged)}")
    print("💡 对应关系说明: train.txt 中的第 N 行评论，精准对应 yelp_interaction_base.csv 中的第 N 行 (不包含表头)。")


if __name__ == "__main__":
    process_stage_1_and_2_split()