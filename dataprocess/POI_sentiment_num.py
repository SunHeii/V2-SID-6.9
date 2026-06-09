import pandas as pd
import os

print("启动 PID 跨界映射修复程序...")

# 1. 读取路径
data_dir = "/home/mysjz/mywork/V2-SID/data/NOLA"  # 请根据你的实际路径调整
sentiment_path = os.path.join(data_dir, "NOLA_poi_sentiment.csv")
pidmap_path = os.path.join(data_dir, "meta/pidmap.csv")

# 2. 读取数据
sent_df = pd.read_csv(sentiment_path)
pidmap_df = pd.read_csv(pidmap_path)

# 注意：原版作者在 save_mapping 时硬编码了表头，pidmap 的表头可能叫 original_uid, new_uid
# 我们强制把它重命名，防止歧义
pidmap_df.columns = ['original_pid', 'new_pid']

print(f"修复前：情感表有 {len(sent_df)} 行，PID 为字符串如 {sent_df['PId'].iloc[0]}")

# 3. 进行精确合并 (通过字符串 ID 对齐)
merged_sent = sent_df.merge(pidmap_df, left_on='PId', right_on='original_pid', how='inner')

# 4. 把数字 PID 覆盖掉原来的字符串 PID
merged_sent['PId'] = merged_sent['new_pid']

# 5. 丢弃多余的临时列
merged_sent = merged_sent.drop(columns=['original_pid', 'new_pid'])

print(f"修复后：情感表有 {len(merged_sent)} 行，PID 已转换为数字如 {merged_sent['PId'].iloc[0]}")

# 6. 保存覆盖原文件
merged_sent.to_csv(sentiment_path, index=False)
print(f"覆盖保存成功: {sentiment_path}")
print("ID 对齐完毕！现在系统可以真正注入情感了！")