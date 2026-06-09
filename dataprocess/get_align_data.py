import pandas as pd
import json
from ast import literal_eval
import os
import random

# ==========================================
# 全局路径与参数配置区
# ==========================================




# ==========================================
# 1. 深度加载三模态特征 (物理 + 密码本 + 情感)
# ==========================================





# ==========================================
# ⚙️ 全局路径与参数配置区
# ==========================================
datafold = "NOLA"  # 替换为你的数据集名称
stage = "alignment"
sid_csv_name = f"SID/{datafold}_SID.csv"
sentiment_csv_name = f"{datafold}_poi_sentiment.csv"  # 高鲁棒性情感字典文件

data_dir = f"/home/mysjz/mywork/V2-SID/data/{datafold}"
out_dir = os.path.join(data_dir, stage)
os.makedirs(out_dir, exist_ok=True)

print(f"启动 SA-SID 语义对齐数据生成管线 (数据集: {datafold})...")

# ==========================================
# 1. 深度加载三模态特征
# ==========================================
poi_info = pd.read_csv(os.path.join(data_dir, "poi_info.csv"))
poi_codes = pd.read_csv(os.path.join(data_dir, sid_csv_name))
poi_sentiment = pd.read_csv(os.path.join(data_dir, sentiment_csv_name))

poi_info.columns = [c.lower() for c in poi_info.columns]
poi_codes.columns = [c.lower() for c in poi_codes.columns]
poi_sentiment.columns = [c.lower() for c in poi_sentiment.columns]

print(poi_info)
print(poi_sentiment)
print(poi_codes)

poi_codes["sid"] = poi_codes["sid"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

merged = poi_info.merge(poi_codes, on="pid", how="inner")
merged = merged.merge(poi_sentiment, on="pid", how="left")

print(f"成功融合 {len(merged)} 个商户的特征。准备计算全城情感分位数基准...")

# ==========================================
# 2. 核心大招：计算全城动态分位数 (Quantiles)
# ==========================================
# 找出所有包含 'Final_' 的情感列
sentiment_cols = [c for c in merged.columns if 'Final_' in c]
if not sentiment_cols:
    # 容错：如果没有 Final_ 前缀，找经典的小写列名
    sentiment_cols = ['service', 'environment', 'price', 'location', 'core_experience']
    sentiment_cols = [c for c in sentiment_cols if c in merged.columns]

quantiles_dict = {}
for col in sentiment_cols:
    # 只针对有真实打分（非缺失）的商户计算分位数
    valid_scores = merged[col].dropna()
    if len(valid_scores) > 0:
        quantiles_dict[col] = {
            'q90': valid_scores.quantile(0.90),  # Top 10%
            'q70': valid_scores.quantile(0.70),  # Top 30%
            'q30': valid_scores.quantile(0.30),  # Bottom 30%
            'q10': valid_scores.quantile(0.10)  # Bottom 10%
        }
        print(
            f"{col} 阈值分布: Top10%={quantiles_dict[col]['q90']:.3f}, Top30%={quantiles_dict[col]['q70']:.3f}, Bottom30%={quantiles_dict[col]['q30']:.3f}, Bottom10%={quantiles_dict[col]['q10']:.3f}")
    else:
        quantiles_dict[col] = {'q90': 1, 'q70': 0.5, 'q30': -0.5, 'q10': -1}  # 极端兜底


# ==========================================
# 3. 细腻度文本映射器 (Dynamic Quantile Mapper)
# ==========================================
def map_sentiment_by_quantile(row):
    excellent = []
    good = []
    poor = []
    terrible = []

    for col in sentiment_cols:
        score = row.get(col)
        if pd.isna(score): continue

        # 提取真实维度名，例如 "Final_service" -> "service"
        aspect_name = col.replace('Final_', '').replace('_', ' ')
        q_thresholds = quantiles_dict.get(col)
        if not q_thresholds: continue

        # 阶梯式细腻度判定
        if score >= q_thresholds['q90']:
            excellent.append(aspect_name)
        elif score >= q_thresholds['q70']:
            good.append(aspect_name)
        elif score <= q_thresholds['q10']:
            terrible.append(aspect_name)
        elif score <= q_thresholds['q30']:
            poor.append(aspect_name)
        # 中间的 40% (q30 ~ q70) 直接忽略，保持大模型注意力纯净

    parts = []
    if excellent: parts.append(f"excellent {', '.join(excellent)}")
    if good: parts.append(f"good {', '.join(good)}")
    if poor: parts.append(f"poor {', '.join(poor)}")
    if terrible: parts.append(f"terrible {', '.join(terrible)}")

    if not parts:
        return "neutral overall"

    return ", and ".join(parts)


# ==========================================
# 4. 组装知识图谱
# ==========================================
mapping = {}
for _, row in merged.iterrows():
    code_tuple = tuple(row["sid"])
    code_key = str(list(code_tuple))
    sentiment_desc = map_sentiment_by_quantile(row)

    mapping[code_key] = {
        "category": row["category"],
        "region": row["region"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "visit_time_and_count": row.get("visit_time_and_count", "{}"),
        "overall_sentiment_profile": sentiment_desc
    }

mapping_file = os.path.join(out_dir, "semantic_code_mapping.json")
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)


# ==========================================
# 5. 构建 Instruction Dataset
# ==========================================
def code_to_tag(code_list):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join([f"<{letters[i]}_{v}>" if i < len(letters) else f"<x{i}_{v}>" for i, v in enumerate(code_list)])


dataset = []
for code_key, meta in mapping.items():
    code_list = json.loads(code_key.replace("'", '"'))
    tag = code_to_tag(code_list)

    attributes_str = (
        f"Category: {meta['category']}; "
        f"Region: {meta['region']}; "
        f"Latitude: {meta['latitude']}; "
        f"Longitude: {meta['longitude']}; "
        f"Visit_time_and_count: {meta['visit_time_and_count']}; "
        f"Overall_Sentiment_Profile: {meta['overall_sentiment_profile']}"
    )

    dataset.append({
        "instruction": "Given a POI's physical attributes and overall sentiment profile, describe its semantic code.",
        "input": f"Can you based on the attributes {{{attributes_str}}} predict the POI semantic code?",
        "output": f"{tag}"
    })

    dataset.append({
        "instruction": "Given a semantic code, describe its POI's physical attributes and overall sentiment profile.",
        "input": f"Can you describe the attributes and sentiment profile of the POI with semantic code {tag}?",
        "output": f"{{{attributes_str}}}"
    })

dataset_file = os.path.join(out_dir, "semantic_instruction_dataset.json")
with open(dataset_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

# 预览生成结果
print("\n[效果预览] 大模型将要学习的第一条细腻情感知识:")
print(json.dumps(dataset[0], indent=4, ensure_ascii=False))

# 切分保存
random.shuffle(dataset)
split_idx = len(dataset) // 10
valid_data, train_data = dataset[:split_idx], dataset[split_idx:]

with open(os.path.join(out_dir, 'train_align.json'), 'w', encoding="utf-8") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
with open(os.path.join(out_dir, 'valid_align.json'), 'w', encoding="utf-8") as f:
    json.dump(valid_data, f, indent=4, ensure_ascii=False)

print(f"\n对齐数据生成完毕！所有 neutral overall 的假象均被破除！")

# 启动 SA-SID 语义对齐数据生成管线 (数据集: NOLA)...
#        pid  ...                               visit_time_and_count
# 0        0  ...  {14: 3, 22: 2, 15: 2, 21: 2, 19: 2, 16: 1, 20:...
# 1        1  ...  {16: 9, 15: 8, 0: 6, 20: 5, 13: 5, 4: 4, 3: 4,...
# 2        2  ...  {16: 8, 20: 8, 23: 6, 21: 6, 17: 5, 2: 5, 15: ...
# 3        3  ...  {23: 2, 4: 2, 22: 2, 3: 1, 0: 1, 7: 1, 10: 1, ...
# 4        4  ...  {22: 12, 13: 11, 20: 10, 2: 10, 0: 10, 19: 9, ...
# ...    ...  ...                                                ...
# 1083  1083  ...  {22: 4, 1: 3, 2: 3, 14: 1, 3: 1, 19: 1, 23: 1,...
# 1084  1084  ...  {19: 3, 16: 2, 23: 2, 22: 2, 15: 2, 21: 2, 18:...
# 1085  1085  ...  {18: 4, 22: 4, 19: 3, 2: 3, 21: 3, 12: 2, 1: 1...
# 1086  1086  ...  {3: 12, 2: 9, 21: 8, 0: 7, 19: 6, 5: 6, 17: 5,...
# 1087  1087  ...  {17: 12, 20: 11, 22: 9, 1: 9, 21: 8, 3: 7, 2: ...
#
# [1088 rows x 6 columns]
#       pid  total_interactions  service  ...   price  location  core_experience
# 0     659                  12   0.4494  ... -0.2302    0.4794           0.4059
# 1      17                  26   0.5342  ... -0.1699    0.5486           0.5632
# 2     195                  12   0.4352  ... -0.0649    0.4761           0.4927
# 3     361                  16   0.3788  ... -0.2130    0.5075           0.4847
# 4     826                  50   0.1874  ... -0.1791    0.3039           0.2282
# ...   ...                 ...      ...  ...     ...       ...              ...
# 1083  666                  64   0.4740  ... -0.1660    0.4399           0.5096
# 1084  622                  14   0.4289  ... -0.3073    0.4449           0.4121
# 1085  745                  19   0.4593  ... -0.2886    0.4393           0.4095
# 1086  985                  18   0.4427  ... -0.1438    0.4705           0.4816
# 1087  517                  10   0.5422  ... -0.0562    0.5333           0.5101
#
# [1088 rows x 7 columns]
#        pid              sid                                             vector
# 0        0   [9, 26, 29, 0]  [1.4174116849899292, 0.7163196802139282, 0.834...
# 1        1   [47, 15, 9, 0]  [1.5774601697921753, 0.9686583280563354, 0.905...
# 2        2     [25, 61, 23]  [1.6101516485214233, 0.9199033379554749, 0.813...
# 3        3      [23, 2, 22]  [0.9492941498756409, 2.2374894618988037, -0.31...
# 4        4     [47, 15, 37]  [2.1789650917053223, 1.2187577486038208, 1.319...
# ...    ...              ...                                                ...
# 1083  1083  [35, 26, 60, 2]  [2.2062268257141113, 1.0048741102218628, 1.122...
# 1084  1084  [47, 61, 32, 2]  [1.8981971740722656, 1.0830833911895752, 1.121...
# 1085  1085      [9, 58, 23]  [1.5433531999588013, 0.7139634490013123, 0.794...
# 1086  1086      [48, 1, 58]  [6.055662155151367, 5.446054458618164, 0.49412...
# 1087  1087   [0, 35, 21, 1]  [1.5947552919387817, 1.1887098550796509, 0.822...
#
# [1088 rows x 3 columns]
# 成功融合 1088 个商户的特征。准备计算全城情感分位数基准...
# service 阈值分布: Top10%=0.514, Top30%=0.465, Bottom30%=0.400, Bottom10%=0.340
# environment 阈值分布: Top10%=0.397, Top30%=0.343, Bottom30%=0.266, Bottom10%=0.206
# price 阈值分布: Top10%=-0.104, Top30%=-0.160, Bottom30%=-0.237, Bottom10%=-0.286
# location 阈值分布: Top10%=0.557, Top30%=0.510, Bottom30%=0.438, Bottom10%=0.378
# core_experience 阈值分布: Top10%=0.522, Top30%=0.476, Bottom30%=0.410, Bottom10%=0.354
#
# [效果预览] 大模型将要学习的第一条细腻情感知识:
# {
#     "instruction": "Given a POI's physical attributes and overall sentiment profile, describe its semantic code.",
#     "input": "Can you based on the attributes {Category: Restaurants, American (Traditional); Region: 12; Latitude: 29.9492153; Longitude: -90.0706026; Visit_time_and_count: {14: 3, 22: 2, 15: 2, 21: 2, 19: 2, 16: 1, 20: 1, 0: 1}; Overall_Sentiment_Profile: excellent environment, core experience, and good service, location} predict the POI semantic code?",
#     "output": "<a_9><b_26><c_29><d_0>"
# }