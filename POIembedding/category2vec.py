# category2vec.py (恢复为标准的 64 维版)
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import argparse
import pickle

def category2vec(csv_path, output_dir, model_name="all-MiniLM-L6-v2", n_components=64, category_column="category"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取基础元数据 CSV
    print(f"读取元数据文件: {csv_path}")
    df = pd.read_csv(csv_path,  encoding='utf-8')

    # 容错处理：填充可能为空的分类
    df[category_column] = df[category_column].fillna('Unknown')
    categories = df[category_column].unique().tolist()
    print(f"共发现 {len(categories)} 种独特的 POI 类别")

    # 2. 加载轻量级句子转化大模型
    print(f"正在加载语义提取模型: {model_name}")
    # 使用你本地的模型路径
    model = SentenceTransformer("/home/mysjz/mywork/Models/all-MiniLM-L6-v2")

    # 3. 提取高维稠密文本特征
    print("正在生成品类文本 Embedding...")
    embeddings = model.encode(categories, show_progress_bar=True)
    print(f"初始提取向量维度: {embeddings.shape}")

    # 4. 【恢复修改】：应用 PCA 降维至标准的 64 维
    if n_components is not None and n_components < embeddings.shape[1]:
        print(f"正在执行 PCA 降维，目标维度: {n_components} ...")
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"降维后最终向量维度: {embeddings_reduced.shape}")
        final_embeddings = embeddings_reduced
    else:
        final_embeddings = embeddings

    # 构建品类字符串到向量的字典映射
    category_to_embedding = dict(zip(categories, final_embeddings))

    # 5. 落盘保存
    npy_path = os.path.join(output_dir, "category_embeddings.npy")
    np.save(npy_path, final_embeddings)

    # 保持输出文件名为 category_emb.pkl，无缝适配 POI2emb.py
    pkl_path = os.path.join(output_dir, "category_emb.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(category_to_embedding, f)

    print(f"\n 品类向量提取完成！结果已保存至: {output_dir}")
    print(f"  -  矩阵文件: {npy_path}")
    print(f"  - 字典文件: {pkl_path} (此文件将被 POI2emb.py 读取)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate POI category embeddings for SA-SID")

    # 默认路径配置（你可以通过命令行参数灵活覆盖）
    parser.add_argument("--csv_path", default="../data/NOLA/poi_info.csv", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="../data/NOLA/embeddings", help="Output directory for results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")

    # ⚠️ 目标维度恢复锁定为 64
    parser.add_argument("--dim", type=int, default=64, help="Target dimensionality (PCA)")
    parser.add_argument("--column", default="category", help="Category column name")

    args = parser.parse_args()

    category2vec(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        n_components=args.dim,
        category_column=args.column
    )

# 读取元数据文件: ../data/NOLA/poi_info.csv
# 共发现 989 种独特的 POI 类别
# 正在加载语义提取模型: all-MiniLM-L6-v2
# 正在生成品类文本 Embedding...
# Batches: 100%|██████████| 31/31 [00:00<00:00, 60.43it/s]
# 初始提取向量维度: (989, 384)
# 正在执行 PCA 降维，目标维度: 64 ...
# 降维后最终向量维度: (989, 64)
#
#  品类向量提取完成！结果已保存至: ../data/NOLA/
#   -  矩阵文件: ../data/NOLA/category_embeddings.npy
#   - 字典文件: ../data/NOLA/category_emb.pkl (此文件将被 POI2emb.py 读取)