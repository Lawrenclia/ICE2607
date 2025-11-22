import sys
import cv2
import numpy as np
import os
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt


cwd = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(cwd, "dataset")
TARGET_IMG = os.path.join(cwd, "target.jpg")
OUTPUT_DIR = os.path.join(cwd, "output")

# 基于PPT的LSH参数
# d = 12 (12维颜色直方图)
D_DIM = 12
# C = 3 (量化为 3 个区间: 0, 1, 2)
C_BINS = 3
# d' = d * C = 36 (Hamming空间的总维度)
D_PRIME_DIM = D_DIM * C_BINS


def extract_features(image_path):
    """
    每幅图像用一个12维的颜色直方图p表示 (R, G, B 各4个bin)，并将其归一化
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # 转换为RGB，因为cv2默认加载为BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算4个bin的直方图
    hist_r = cv2.calcHist([image_rgb], [0], None, [4], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [4], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [4], [0, 256])

    # 将直方图连接成一个12维向量
    hist = np.concatenate((hist_r, hist_g, hist_b)).flatten()

    # 归一化直方图
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist
        

def quantize_vector(p_vec):
    """
    根据 [0, 0.3), [0.3, 0.6), [0.6, 1]，将归一化的12维向量量化为 3 个区间 (0, 1, 2)。
    """

    quantized_vec = np.zeros_like(p_vec, dtype=int)

    # 位于 [0.3, 0.6) 区间的值为 1
    quantized_vec[p_vec >= 0.3] = 1
    # 位于 [0.6, 1] 区间的值为 2
    quantized_vec[p_vec >= 0.6] = 2

    return quantized_vec


def calculate_hash(p_quantized, projection_set_I, d, C):
    """
    计算LSH哈希值 g(p)。
    p_quantized: d维的量化向量 (值为 0, 1, 或 2)
    projection_set_I: 一个索引列表，范围从 1 到 d*C 
    d: D_DIM (12)
    C: C_BINS (3)
    """

    hash_val = []
    
    # 对投影集进行排序
    sorted_I = sorted(projection_set_I)
    
    for i in range(d):
        p_i = p_quantized[i]
        
        # 定义这个维度在 d' 空间中的范围
        range_start = i * C + 1
        range_end = (i + 1) * C
        
        # 找到所有落入此维度范围内的投影索引
        I_i = [k for k in sorted_I if range_start <= k <= range_end]
        
        if not I_i:
            # 没有来自 I 的索引在此维度范围内
            continue
            
        # 1的个数 = I|i 中小于等于 (i-1)*C + p_i 的元素个数
        threshold = i * C + p_i
        
        num_ones = sum(1 for k in I_i if k <= threshold)
        
        # 投影是 `num_ones` 个 1 后面跟着 (len(I_i) - num_ones) 个 0
        proj_i = [1] * num_ones + [0] * (len(I_i) - num_ones)
        
        hash_val.extend(proj_i)
        
    # 返回一个不可变的元组，用作字典的键
    return tuple(hash_val)


def run_nn_search(db_features, target_features):
    """
    执行NN搜索。
    """

    start_time_nn = time.time()
    
    best_match_nn = None
    min_dist_nn = float('inf')
    
    for img_name, vec in db_features.items():

        dist = np.linalg.norm(target_features - vec)
        
        if dist < min_dist_nn:
            min_dist_nn = dist
            best_match_nn = img_name
            
    end_time_nn = time.time()
    query_time_nn = end_time_nn - start_time_nn
    
    return best_match_nn, query_time_nn


def run_lsh_search(projection_set, db_features, db_quantized_features, target_quantized_vec, target_features, d, C):
    """
    执行LSH搜索。
    """

    k = len(projection_set)

    # 1. LSH 预处理 (构建哈希表)
    start_time_lsh_build = time.time()
    hash_buckets = defaultdict(list)
    
    for img_name, vec in db_quantized_features.items():
        hash_key = calculate_hash(vec, projection_set, d, C)
        hash_buckets[hash_key].append(img_name)
        
    end_time_lsh_build = time.time()
    build_time_lsh = end_time_lsh_build - start_time_lsh_build
    
    # 2. LSH 检索
    start_time_lsh_query = time.time()
    
    # 找到查询的桶
    query_key = calculate_hash(target_quantized_vec, projection_set, d, C)
    candidate_list = hash_buckets.get(query_key, [])
    
    best_match_lsh = None
    min_dist_lsh = float('inf')
    
    if not candidate_list:
        print("Query hash not found in any bucket.")
        best_match_lsh = "No match in bucket"
    else:
        # 3. 在候选桶内搜索
        for img_name in candidate_list:
            dist = np.linalg.norm(target_features - db_features[img_name])
            
            if dist < min_dist_lsh:
                min_dist_lsh = dist
                best_match_lsh = img_name

    end_time_lsh_query = time.time()
    query_time_lsh = end_time_lsh_query - start_time_lsh_query
    
    return best_match_lsh, build_time_lsh, query_time_lsh, len(candidate_list)


def create_image(target_img_path, result_img_path, title, output_filename):

    target_img_bgr = cv2.imread(target_img_path)
    result_img_bgr = cv2.imread(result_img_path)
    
    if target_img_bgr is None:
        print(f"Error: Failed to load target image {target_img_path}.")
        return
    if result_img_bgr is None:
        print(f"Error: Failed to load result image {result_img_path}.")
        return

    target_img_rgb = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2RGB)
    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制目标图像
    axes[0].imshow(target_img_rgb)
    axes[0].set_title("Target Image")
    axes[0].axis('off') 

    # 绘制最佳匹配图像
    axes[1].imshow(result_img_rgb)
    axes[1].set_title("Best Match")
    axes[1].axis('off')

    # 添加标题
    fig.suptitle(title, fontsize=14)
    

    # 保存图像
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close(fig) 
        

def main():
    # --- 1. 定义投影集 ---

    k_values = [8, 16, 24]

    projection_sets = {}
    for k in k_values:
        # D_PRIME_DIM 变量需要从您的代码的全局部分获取
        projection_sets[k] = random.sample(range(1, D_PRIME_DIM + 1), k)

    # --- 2. 预处理: 加载并处理所有图像 ---
    print("Loading and processing images")
    db_features = {}       # 用于 NN 搜索 (原始特征)
    db_quantized = {}    # 用于 LSH 搜索 (量化特征)
    
    image_files = os.listdir(DATASET_DIR)
    
    for img_name in image_files:
        path = os.path.join(DATASET_DIR, img_name)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
            
        features = extract_features(path)
        if features is not None:
            # D_DIM 和 C_BINS 变量需要从您的代码的全局部分获取
            quantized_p = quantize_vector(features)
            db_features[img_name] = features
            db_quantized[img_name] = quantized_p
            
    if not db_features:
        print(f"Error: No images found in {DATASET_DIR}. Exiting.")
        return

    # 处理目标图像
    if not os.path.exists(TARGET_IMG):
        print(f"Error: Target image {TARGET_IMG} not found. Exiting.")
        return
        
    print("Loading and processing target image")
    target_features = extract_features(TARGET_IMG)
    target_quantized = quantize_vector(target_features)
    print("Preprocessing complete.")

    # --- 3. 运行 NN 搜索 ---
    print("\n--- Running NN Search ---")
    ave_total_time = 0
    for i in range(5):
        nn_match, nn_time = run_nn_search(db_features, target_features)
        if nn_match:
            result_path_nn = os.path.join(DATASET_DIR, nn_match)
            title_nn = f"NN Search {i + 1}: {nn_match} (Time: {nn_time:.6f}s)"
            create_image(TARGET_IMG, result_path_nn, title_nn, f"result_nn_{i + 1}.jpg")

        ave_total_time += nn_time
        print(f"{i + 1}th NN Search:")
        print(f"\tResult: {nn_match}")
        print(f"\tTotal Time: {nn_time:.6f} s")
    print(f"\nAverage NN Search Time over 5 runs: {ave_total_time / 5:.6f} s")

    # --- 4. 循环运行 LSH 搜索 ---
    for k in k_values:
        print(f"\n--- Running LSH Search (k={k}) ---")
        ave_total_build_time = 0
        ave_total_query_time = 0
        for i in range(5):
            # 运行 LSH 搜索
            lsh_match, build_time, query_time, bucket_size = run_lsh_search(
                projection_sets[k], db_features, db_quantized, target_quantized, target_features, D_DIM, C_BINS
            )
            
            if lsh_match != "No match in bucket":
                result_path_lsh = os.path.join(DATASET_DIR, lsh_match)
                title_lsh = f"LSH (k={k}) {i + 1}: {lsh_match} (Query: {query_time:.6f}s, Bucket: {bucket_size})"
                output_filename = f"result_lsh_k{k}_{i + 1}.jpg"
                create_image(TARGET_IMG, result_path_lsh, title_lsh, output_filename)

            ave_total_build_time += build_time
            ave_total_query_time += query_time
            print(f"\n{i + 1}th LSH Search (k={k}):")
            print(f"\tResult: {lsh_match}")
            total_time = build_time + query_time
            print(f"\tTotal Time (Build + Query): {total_time:.6f} s")
            print(f"\t(Build: {build_time:.6f} s, Query: {query_time:.6f} s, Bucket Size: {bucket_size})")
        print(f"\nAverage LSH Build Time over 5 runs (k={k}): {ave_total_build_time / 5:.6f} s")
        print(f"Average LSH Query Time over 5 runs (k={k}): {ave_total_query_time / 5:.6f} s")
        print(f"Average LSH Search Time over 5 runs (k={k}): {(ave_total_build_time + ave_total_query_time) / 5:.6f} s")

if __name__ == "__main__":
    log_file_path = os.path.join(OUTPUT_DIR, "terminal_output.txt")
    
    original_stdout = sys.stdout 
    
    print(f"Terminal output will be redirected to {log_file_path}")
    
    try:
        # 写入日志文件
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # 重定向 stdout 到日志文件
            sys.stdout = log_file
            
            main()
    
    finally:
        sys.stdout = original_stdout 
    
    print("Terminal output saved.")