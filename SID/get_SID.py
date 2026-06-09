# import torch
# from torch.utils.data import DataLoader
# import pandas as pd
# import csv
# from collections import Counter
# import os
# import argparse
# import random
# import numpy as np
# import logging
# from tqdm import tqdm
# import datetime
# from POIdatasets import EmbDataset
# from CRQVAE.crqvae import CRQVAE
#
# def ensure_dir(dir_path):
#
#     os.makedirs(dir_path, exist_ok=True)
#
# def set_color(log, color, highlight=True):
#     color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
#     try:
#         index = color_set.index(color)
#     except:
#         index = len(color_set) - 1
#     prev_log = "\033["
#     if highlight:
#         prev_log += "1;3"
#     else:
#         prev_log += "0;3"
#     prev_log += str(index) + "m"
#     return prev_log + log + "\033[0m"
#
# def get_local_time():
#     r"""Get current time
#
#     Returns:
#         str: current time
#     """
#     cur = datetime.datetime.now()
#     cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
#
#     return cur
#
# def delete_file(filename):
#     if os.path.exists(filename):
#         os.remove(filename)
#
#
# def parse_args(datafold):
#     parser = argparse.ArgumentParser(description="Index")
#
#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
#     parser.add_argument('--batch_size', type=int, default=128, help='batch size')
#     parser.add_argument('--num_workers', type=int, default=4, )
#     parser.add_argument('--eval_step', type=int, default=10, help='eval step')
#     parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
#     parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
#     parser.add_argument('--warmup_epochs', type=int, default=100, help='warmup epochs')
#     parser.add_argument("--data_path", type=str, default=f"", help="Path to POI embedding dict (.pkl)")
#
#     parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
#     parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
#     parser.add_argument("--bn", type=bool, default=True, help="use bn or not")
#     parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
#     parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
#     parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
#     parser.add_argument('--use_sk', type=bool, default=False, help="use sinkhorn or not")
#     parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.1, 0.1, 0.1], help="sinkhorn epsilons")
#     parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
#     parser.add_argument("--use-linear", type=int, default=1, help="use-linear")
#
#     parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
#
#     parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64,64,64], help='emb num of every vq')
#     parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
#     parser.add_argument('--quant_loss_weight', type=float, default=0.5, help='vq quantion loss weight')
#     parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
#     parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128], help='hidden sizes of every layer')
#
#     parser.add_argument('--save_limit', type=int, default=5)
#     parser.add_argument("--ckpt_dir", type=str, default=f"", help="output directory for model")
#
#     return parser.parse_args()
#
#
# def get_quantization():
#
#     """fix the random seed"""
#     seed = 2024
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     datafold = ""
#     args = parse_args(datafold)
#     print("=================================================")
#     print(args)
#     print("=================================================")
#
#     logging.basicConfig(level=logging.DEBUG)
#
#     """build dataset"""
#     data = EmbDataset(args.data_path)
#
#     model = CRQVAE(in_dim=data.dim,
#                    num_emb_list=args.num_emb_list,
#                    e_dim=args.e_dim,
#                    layers=args.layers,
#                    dropout_prob=args.dropout_prob,
#                    bn=args.bn,
#                    loss_type=args.loss_type,
#                    quant_loss_weight=args.quant_loss_weight,
#                    beta=args.beta,
#                    kmeans_init=args.kmeans_init,
#                    kmeans_iters=args.kmeans_iters,
#                    sk_epsilons=args.sk_epsilons,
#                    sk_iters=args.sk_iters,
#                    use_linear=args.use_linear,
#                   )
#     # print(model)
#     data_loader = DataLoader(data,num_workers=args.num_workers,
#                              batch_size=args.batch_size, shuffle=True,
#                              pin_memory=True)
#
#     best_loss_ckpt = "best_loss_model.pth"
#     best_collision_ckpt = "best_collision_model.pth"
#     time_dir = ""
#     best_loss_ckpt_file = args.ckpt_dir + f"{time_dir}/{best_loss_ckpt}"
#     best_collision_ckpt_file = args.ckpt_dir + f"{time_dir}/{best_collision_ckpt}"
#
#     checkpoint = torch.load(best_loss_ckpt_file, map_location=args.device, weights_only=False)
#
#     model = CRQVAE(in_dim=data.dim,
#                    num_emb_list=args.num_emb_list,
#                    e_dim=args.e_dim,
#                    layers=args.layers,
#                    dropout_prob=args.dropout_prob,
#                    bn=args.bn,
#                    loss_type=args.loss_type,
#                    quant_loss_weight=args.quant_loss_weight,
#                    beta=args.beta,
#                    kmeans_init=args.kmeans_init,
#                    kmeans_iters=args.kmeans_iters,
#                    sk_epsilons=args.sk_epsilons,
#                    sk_iters=args.sk_iters,
#                    use_linear=args.use_linear,
#                   )
#
#     # 加载权重
#     model.load_state_dict(checkpoint["state_dict"])
#     model = model.to(args.device)
#     model.eval()
#
#     SIDs = {}
#     vectors = {}
#
#     iter_data = tqdm(
#                 data_loader,
#                 total=len(data_loader),
#                 ncols=100,
#                 desc=set_color(f"Generate codebooks ", "pink"),
#                 )
#
#     for batch_idx, data in enumerate(iter_data):
#             pids, data = data[0], data[1]
#             pids = pids.tolist()
#             data = data.to(args.device)
#             vector, indices = model.get_indices(data)
#             for indx, poi in enumerate(pids):
#                 SIDs[poi] = indices[indx].tolist()
#                 vectors[poi] = vector[indx].tolist()
#
#     # print(SIDs)
#
#     value_counts = Counter(tuple(value) for value in SIDs.values())
#     seen_values = {}
#
#     updated_dict = {}
#     # for key, value in SIDs.items():
#     for key in sorted(SIDs.keys()):
#         value = SIDs[key]
#         value_tuple = tuple(value)
#         if value_counts[value_tuple] > 1:
#             if value_tuple not in seen_values:
#                 seen_values[value_tuple] = 0
#             else:
#                 seen_values[value_tuple] += 1
#             updated_dict[key] = value + [seen_values[value_tuple]]
#         else:
#             updated_dict[key] = value
#
#     csv_file = f""
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)
#
#     with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#
#         writer.writerow(["pid", "sid", "vector"])
#
#         for key, value in updated_dict.items():
#             writer.writerow([key, value, vectors[key]])
#
# if __name__ == "__main__":
#     get_quantization()

import torch
from torch.utils.data import DataLoader
import pandas as pd
import csv
from collections import Counter
import os
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
import datetime
from POIdatasets import EmbDataset
from CRQVAE.crqvae import CRQVAE


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Sentiment-Aware Semantic ID (SA-SID)")

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )

    # 指明底座特征文件的路径（供加载器识别自动维度，如98维）
    parser.add_argument("--data_path", type=str, default="../data/NOLA/embeddings/poi_Emb_dict.pkl",
                        help="Path to POI embedding dict (.pkl)")

    # 指明刚才 train_SID.py 保存权重的目录
    parser.add_argument("--ckpt_dir", type=str, default="../data/NOLA/Jun-08-2026_14-28-31/",
                        help="Directory where model checkpoints are saved")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=True, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--use_sk', type=bool, default=False, help="use sinkhorn or not")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.1, 0.1, 0.1], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use-linear", type=int, default=1, help="use-linear")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 64, 64], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=0.5, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128], help='hidden sizes of every layer')

    return parser.parse_args()


def get_quantization():
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print("开始提取情感感知语义 ID (SA-SID) 密码本...")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    # 这里非常关键，它会自动读出输入维度（98维）
    data = EmbDataset(args.data_path)
    print(f"成功加载数据集，输入特征维度自动识别为: {data.dim} 维")

    # 动态寻找最新/最好的模型权重
    # 我们优先加载 best_collision_model.pth，如果没有，就加载 best_loss_model.pth
    ckpt_path_collision = os.path.join(args.ckpt_dir, "best_collision_model.pth")
    ckpt_path_loss = os.path.join(args.ckpt_dir, "best_loss_model.pth")

    if os.path.exists(ckpt_path_collision):
        chosen_ckpt = ckpt_path_collision
        print(f"找到最低冲突率的优秀模型: {chosen_ckpt}")
    elif os.path.exists(ckpt_path_loss):
        chosen_ckpt = ckpt_path_loss
        print(f" 找到最低 Loss 的优秀模型: {chosen_ckpt}")
    else:
        raise FileNotFoundError(
            f"在 {args.ckpt_dir} 下找不到任何训练好的模型权重！请确保你已经成功跑完了 train_SID.py！")

    checkpoint = torch.load(chosen_ckpt, map_location=args.device, weights_only=False)

    # 初始化模型架构 (确保和 train_SID 一致)
    model = CRQVAE(in_dim=data.dim,
                   num_emb_list=args.num_emb_list,
                   e_dim=args.e_dim,
                   layers=args.layers,
                   dropout_prob=args.dropout_prob,
                   bn=args.bn,
                   loss_type=args.loss_type,
                   quant_loss_weight=args.quant_loss_weight,
                   beta=args.beta,
                   kmeans_init=args.kmeans_init,
                   kmeans_iters=args.kmeans_iters,
                   sk_epsilons=args.sk_epsilons,
                   sk_iters=args.sk_iters,
                   use_linear=args.use_linear,
                   )

    # 加载权重
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()

    SIDs = {}
    vectors = {}

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False,  # 提取特征时不需要打乱
                             pin_memory=True)

    iter_data = tqdm(
        data_loader,
        total=len(data_loader),
        ncols=100,
        desc=set_color("Extracting SA-SIDs ", "pink"),
    )

    # 无梯度计算，加速提取
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iter_data):
            pids, features = batch_data[0], batch_data[1]
            pids = pids.tolist()
            features = features.to(args.device)

            # 获取离散的树形索引和重建向量
            vector, indices = model.get_indices(features)
            for indx, poi in enumerate(pids):
                SIDs[poi] = indices[indx].tolist()
                vectors[poi] = vector[indx].tolist()

    # 处理冲突 (Collision Handling): 如果同一片云的 ID 重复，追加一位区分
    value_counts = Counter(tuple(value) for value in SIDs.values())
    seen_values = {}
    updated_dict = {}

    for key in sorted(SIDs.keys()):
        value = SIDs[key]
        value_tuple = tuple(value)
        if value_counts[value_tuple] > 1:
            if value_tuple not in seen_values:
                seen_values[value_tuple] = 0
            else:
                seen_values[value_tuple] += 1
            updated_dict[key] = value + [seen_values[value_tuple]]
        else:
            updated_dict[key] = value

            # 指定最终生成的密码本存放位置
    csv_file = "../data/NOLA/SID/NOLA_SID.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["pid", "sid", "vector"])
        for key, value in updated_dict.items():
            writer.writerow([key, value, vectors[key]])

    print(f"\n SA-SID 密码本提取大功告成！")
    print(f"全城商户身份证（附带情感记忆）已安全落盘至: {csv_file}")


if __name__ == "__main__":
    get_quantization()

# =================================================
# 开始提取情感感知语义 ID (SA-SID) 密码本...
# Namespace(batch_size=128, num_workers=4, data_path='../data/NOLA/embeddings/poi_Emb_dict.pkl', ckpt_dir='../data/NOLA/Jun-08-2026_14-28-31/', device='cuda:0', dropout_prob=0.1, bn=True, loss_type='mse', kmeans_init=True, kmeans_iters=100, use_sk=False, sk_epsilons=[0.1, 0.1, 0.1], sk_iters=50, use_linear=1, num_emb_list=[64, 64, 64], e_dim=64, quant_loss_weight=0.5, beta=0.25, layers=[512, 256, 128])
# =================================================
# 成功加载数据集，输入特征维度自动识别为: 98 维
# 找到最低冲突率的优秀模型: ../data/NOLA/Jun-08-2026_14-28-31/best_collision_model.pth
# Extracting SA-SIDs : 100%|████████████████████████████████████████████| 9/9 [00:00<00:00, 58.15it/s]
#
#  SA-SID 密码本提取大功告成！
# 全城商户身份证（附带情感记忆）已安全落盘至: ../data/NOLA/SID/NOLA_SID.csv