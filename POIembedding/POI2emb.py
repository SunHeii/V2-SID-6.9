# build_poi_semantic_vectors.py (SA-SID 增强版)
import pandas as pd
import numpy as np
import pickle
import ast
import argparse
import os


def normalize_vector(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8)


def parse_time_dict(x):
    if pd.isna(x) or x == '' or x == '{}':
        return {}
    try:
        return ast.literal_eval(x)
    except:
        return {}


def extract_time_features(time_dict):
    vec = np.zeros(24)
    for h in range(24):
        vec[h] = time_dict.get(h, 0)

    if vec.sum() == 0:
        return np.zeros(27)  # 24 历史分布 + 3 统计指标

    hist = vec / vec.sum()
    hours = np.arange(24)
    mean_hour = (hours * hist).sum()
    variance = ((hours - mean_hour) ** 2 * hist).sum()
    peak_hour = hours[np.argmax(hist)]

    return np.concatenate([
        hist,
        [mean_hour / 24.0,
         variance / (24.0 ** 2),
         peak_hour / 24.0]
    ])


def build_poi_embeddings(csv_path, category_emb_path, sentiment_csv_path, output_dir):
    """
    核心融合管线：将物理地理空间、纯文本类别与自适应贝叶斯平滑情感进行融合。
    """
    print("启动情感感知 POI 密集特征构建管线...")

    # 1. 检查并加载基础设施数据
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到商户基础元数据表: {csv_path}")
    if not os.path.exists(category_emb_path):
        raise FileNotFoundError(f"找不到品类向量文件: {category_emb_path}")

    df_poi = pd.read_csv(csv_path)
    print(f"成功读取 {len(df_poi)} 个 POI 的基础元数据。")

    with open(category_emb_path, 'rb') as f:
        cat_emb_dict = pickle.load(f)
    print("品类 Embedding 字典加载成功。")

    # ========================================================
    # 【核心新增】：动态加载高鲁棒性贝叶斯时间衰减情感大表
    # ========================================================
    sentiment_dict = {}
    if sentiment_csv_path and os.path.exists(sentiment_csv_path):
        print(f"正在加载高鲁棒性全局情感字典: {sentiment_csv_path}")
        df_sent = pd.read_csv(sentiment_csv_path)

        # 强制将列名转为小写，完美容错大小写不一致问题
        df_sent.columns = [c.lower() for c in df_sent.columns]

        # 提取包含 final_ 的 5 维特征列
        final_cols = [c for c in df_sent.columns if 'final_' in c]
        if len(final_cols) == 0:
            # 兼容处理：如果没有 final_ 前缀，直接找经典的五个维度列
            classic_cols = ['service', 'environment', 'price', 'location', 'core_experience']
            final_cols = [c for c in df_sent.columns if c in classic_cols]

        print(f"侦测到有效目标情感维度: {final_cols}")

        # 转换为 O(1) 检索效率的 Python 字典
        for _, row in df_sent.iterrows():
            pid = row['pid']
            # 按顺序抽取得分
            sent_vec = np.array([row[col] for col in final_cols], dtype=np.float32)
            sentiment_dict[pid] = sent_vec
        print(f"成功将 {len(sentiment_dict)} 个商户的情感偏好内化进特征生成域。")
    else:
        print("未检测到有效的情感特征文件，将退退回常规纯物理特征拼接模式。")

    # 2. 遍历构建高维联合向量
    valid_pids = []
    full_vectors = []

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df_poi.iterrows():
        try:
            pid = row['pid']
            cat = row['category']

            # A. 提取经纬度空间地理特征
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            geo_feat = np.array([lat, lon], dtype=np.float32)

            # B. 检索品类语义特征
            if cat in cat_emb_dict:
                cat_feat = cat_emb_dict[cat].astype(np.float32)
            else:
                # 容错机制：未知品类用全 0 填充
                cat_feat = np.zeros(next(iter(cat_emb_dict.values())).shape, dtype=np.float32)

            # C. 提取访问时间段分布特征
            time_dict = parse_time_dict(row['visit_time_and_count'])
            time_feat = extract_time_features(time_dict).astype(np.float32)

            # D. 【核心注入】：无痛拼装 5 维静态平滑情感向量
            if pid in sentiment_dict:
                sent_feat = sentiment_dict[pid]
            else:
                # 兜底设计：新店或无评论店用全 0 向量做中性特征占位
                sent_feat = np.zeros(5, dtype=np.float32) if len(sentiment_dict) > 0 else np.array([], dtype=np.float32)

            # E. 特征高维大统一串联
            # 将地理、品类、时间与情感特征串联成全新的多模态表征向量
            poi_combined_vector = np.concatenate([geo_feat, cat_feat, time_feat, sent_feat])

            # 执行标准的单位向量归一化，防止由于特征数值跨度过大导致 K-Means 聚类失衡
            poi_normalized_vector = normalize_vector(poi_combined_vector)

            full_vectors.append(poi_normalized_vector)
            valid_pids.append(pid)

        except Exception as e:
            print(f"处理商户 pid 异常 {row.get('pid', 'UNKNOWN')}: {e}")
            continue

    full_vectors = np.array(full_vectors)
    print(f"\n密集图谱向量构建完毕，共服务 {len(full_vectors)} 个高质 POI。")
    print(f"融合后的 SA-SID 底层总特征维度为: {full_vectors.shape[1]} 维")

    # 3. 打包并序列化落盘
    poi_vector_dict = dict(zip(valid_pids, full_vectors))
    output_path = os.path.join(output_dir, "poi_Emb_dict.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(poi_vector_dict, f)

    print(f"SA-SID 基础表征矩阵已安全注入，文件落盘成功: {output_path}")


if __name__ == "__main__":
    # 建立命令行解析器，无缝承接原版训练命令
    parser = argparse.ArgumentParser(description="Build Sentiment-Aware POI semantic vectors for CRQ-VAE")
    parser.add_argument("--csv_path", default="../data/NOLA/poi_info.csv", help="Path to poi_info.csv")
    parser.add_argument("--category_emb_path", default="../data/NOLA/embeddings/category_emb.pkl",
                        help="Path to category Word2Vec pkl")

    # 【新增参数】：指定之前第二步生成的贝叶斯平滑特征表路径
    parser.add_argument("--sentiment_csv_path", default="../data/NOLA/NOLA_poi_sentiment.csv",
                        help="Path to robust poi sentiment csv")
    parser.add_argument("--output_dir", default="../data/NOLA/embeddings", help="Output directory")

    args = parser.parse_args()

    build_poi_embeddings(
        csv_path=args.csv_path,
        category_emb_path=args.category_emb_path,
        sentiment_csv_path=args.sentiment_csv_path,
        output_dir=args.output_dir
    )
# 启动情感感知 POI 密集特征构建管线...
# 成功读取 1088 个 POI 的基础元数据。
# 品类 Embedding 字典加载成功。
# 正在加载高鲁棒性全局情感字典: ../data/NOLA/NOLA_poi_sentiment.csv
# 侦测到有效目标情感维度: ['service', 'environment', 'price', 'location', 'core_experience']
# 成功将 1088 个商户的情感偏好内化进特征生成域。
#
# 密集图谱向量构建完毕，共服务 1088 个高质 POI。
# 融合后的 SA-SID 底层总特征维度为: 98 维
# 恭喜！SA-SID 基础表征矩阵已安全注入，文件落盘成功: ../data/NOLA/poi_Emb_dict.pkl