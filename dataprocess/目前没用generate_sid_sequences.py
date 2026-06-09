import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm


def generate_sid_sequences(csv_path, output_dir, max_seq_len=50, min_seq_len=3):
    print("开始执行 SID V2 数据序列化与特征打包...")

    # 1. 读取 SAAA2 产出的高纯度交互宽表
    print(f"[{'1/4'}] 正在加载底座数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # 确保严格按照用户 ID 和时间戳排序（序列模型的核心要求）
    df = df.sort_values(by=['UId', 'UTCTime']).reset_index(drop=True)

    # 2. 重新构建 1-N 的稠密 ID 字典 (将 0 严格预留给 Padding)
    print(f"[{'2/4'}] 构建安全的 1-N 稠密索引 (预留 0 为 PAD)...")
    unique_users = df['UId'].unique()
    unique_pois = df['PId'].unique()

    # 映射字典
    user2id = {uid: idx + 1 for idx, uid in enumerate(unique_users)}
    poi2id = {pid: idx + 1 for idx, pid in enumerate(unique_pois)}

    df['UId_dense'] = df['UId'].map(user2id)
    df['PId_dense'] = df['PId'].map(poi2id)

    print(f"      => 独立用户数: {len(user2id)}, 独立 POI 数: {len(poi2id)}")

    # 定义情感特征列
    sentiment_cols = ['Food_Score', 'Service_Score', 'Ambience_Score', 'Price_Score',
                      'Food_Mask', 'Service_Mask', 'Ambience_Mask', 'Price_Mask']

    # 3. 滑动窗口切片 (核心逻辑)
    print(f"[{'3/4'}] 正在执行按用户的滑动窗口切片 (max_len={max_seq_len})...")

    all_sequences = []

    # 按用户分组处理
    grouped = df.groupby('UId_dense')

    for uid, group in tqdm(grouped, desc="Processing Users"):
        # 如果该用户的总交互数太少，直接过滤
        if len(group) < min_seq_len:
            continue

        # 提取当前用户的时间线特征转换为 numpy 数组
        pois = group['PId_dense'].values
        time_offsets = group['TimeOffset'].values
        latitudes = group['Latitude'].values
        longitudes = group['Longitude'].values

        # 提取 8 维情感张量矩阵 (N x 8)
        sentiments = group[sentiment_cols].values

        # 滑动窗口机制：用前 t-1 个节点预测第 t 个节点
        for t in range(min_seq_len, len(pois) + 1):
            # 获取历史轨迹
            hist_end = t - 1
            hist_start = max(0, hist_end - max_seq_len)  # 截断超出 max_len 的部分

            # 当前切片的长度
            seq_len = hist_end - hist_start

            # 序列化历史特征
            seq_pois = pois[hist_start:hist_end]
            seq_times = time_offsets[hist_start:hist_end]
            seq_lats = latitudes[hist_start:hist_end]
            seq_lons = longitudes[hist_start:hist_end]
            seq_sentiments = sentiments[hist_start:hist_end]

            # 预测目标 (Target)
            target_poi = pois[t - 1]
            target_lat = latitudes[t - 1]
            target_lon = longitudes[t - 1]

            # 打包为一个标准的 SID 训练样本字典
            sample = {
                'user_id': uid,
                'seq_len': seq_len,  # 记录实际长度，方便 POIdatasets.py 做 padding
                'history_pois': seq_pois,
                'history_times': seq_times,
                'history_lats': seq_lats,
                'history_lons': seq_lons,
                'history_sentiments': seq_sentiments,  # 【核心注入】: (seq_len, 8) 的情感矩阵
                'target_poi': target_poi,
                'target_lat': target_lat,
                'target_lon': target_lon
            }
            all_sequences.append(sample)

    # 4. 划分数据集并固化到本地 (.pkl)
    print(f"\n[{'4/4'}] 序列切片完成，共生成 {len(all_sequences)} 条轨迹。正在划分并保存...")

    # 简单的时序划分 (8:1:1)
    train_size = int(len(all_sequences) * 0.8)
    val_size = int(len(all_sequences) * 0.1)

    train_data = all_sequences[:train_size]
    val_data = all_sequences[train_size:train_size + val_size]
    test_data = all_sequences[train_size + val_size:]

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # 保存重构后的映射字典 (极其重要)
    with open(os.path.join(output_dir, 'poi_mapping.pkl'), 'wb') as f:
        pickle.dump({'user2id': user2id, 'poi2id': poi2id}, f)

    print(f"✅ 数据预处理完毕！文件已保存至: {output_dir}")


if __name__ == "__main__":
    # 配置输入路径 (请根据实际情况修改)
    CSV_INPUT = "../data/yelp/sid_interaction_dataset.csv"
    # 配置输出路径 (建议输出到 SID V2 数据目录下)
    PKL_OUTPUT_DIR = "../data/yelp/results/SID_processed_data"

    generate_sid_sequences(CSV_INPUT, PKL_OUTPUT_DIR)