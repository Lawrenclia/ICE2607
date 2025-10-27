import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
#   多尺度特征检测
# ===============================
def detect_keypoints(image, max_corners_per_level=300, levels=4, scale_factor=0.75):
    """
    Args:
        image: 输入图像（BGR格式）
        max_corners_per_level: 每层金字塔最多检测的角点数量
        levels: 金字塔层数
        scale_factor: 每层缩放比例
    Returns:
        keypoints: cv2.KeyPoint 列表
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 构建金字塔
    pyramid = [gray]
    for _ in range(1, levels):
        prev = pyramid[-1]
        resized = cv2.resize(prev, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        pyramid.append(resized)

    keypoints = []
    for i, img_lvl in enumerate(pyramid):
        corners = cv2.goodFeaturesToTrack(img_lvl, maxCorners=max_corners_per_level, qualityLevel=0.01, minDistance=5)
        if corners is None:
            continue
        
        scale = (1 / scale_factor) ** i
        for (x, y) in corners.reshape(-1, 2):
            kp = cv2.KeyPoint(
                x=float(x * scale),
                y=float(y * scale),
                size=10 * scale,
                response=1
            )
            keypoints.append(kp)

    print(f"Step 1: Found {len(keypoints)} keypoints.")
    return keypoints


# ===============================
#   自定义 SIFT 描述子计算
# ===============================
def compute_sift_descriptors_final(image, keypoints):
    """
    为关键点计算自定义 SIFT 描述子。
    Args:
        image: 输入图像（BGR格式）
        keypoints: 关键点列表
    Returns:
        descriptors: ndarray (N, 128)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    descriptors = []

    for kp in keypoints:
        x, y = map(int, map(round, kp.pt))
        radius = max(1, int(round(1.5 * kp.size)))

        # 提取关键点邻域
        patch = cv2.getRectSubPix(gray, (2 * radius, 2 * radius), (x, y))
        if patch is None:
            continue

        # 计算梯度幅值和方向
        dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        # 计算方向直方图（36维）
        gauss = cv2.getGaussianKernel(2 * radius, radius)
        weight = gauss @ gauss.T
        hist = np.zeros(36, np.float32)
        for r in range(2 * radius):
            for c in range(2 * radius):
                bin_idx = int(ang[r, c] // 10) % 36
                hist[bin_idx] += mag[r, c] * weight[r, c]

        # 选择主方向
        main_orientation = np.argmax(hist) * 10

        # 构建旋转归一化窗口（16×16）
        desc_size = 16
        M = cv2.getRotationMatrix2D((kp.pt[0], kp.pt[1]), main_orientation, desc_size / kp.size)
        M[:, 2] += [desc_size / 2 - kp.pt[0], desc_size / 2 - kp.pt[1]]
        aligned = cv2.warpAffine(gray, M, (desc_size, desc_size), flags=cv2.INTER_LINEAR)

        # 在 4×4 网格上统计 8-bin 方向直方图
        dx = cv2.Sobel(aligned, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(aligned, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        descriptor = []
        for r0 in range(0, desc_size, 4):
            for c0 in range(0, desc_size, 4):
                block_hist = np.zeros(8, np.float32)
                for r in range(r0, r0 + 4):
                    for c in range(c0, c0 + 4):
                        bin_idx = int(ang[r, c] // 45) % 8
                        block_hist[bin_idx] += mag[r, c]
                descriptor.extend(block_hist)

        # 归一化处理
        desc = np.array(descriptor, np.float32)
        desc /= (np.linalg.norm(desc) + 1e-7)
        desc = np.clip(desc, 0, 0.2)
        desc /= (np.linalg.norm(desc) + 1e-7)
        descriptors.append(desc)

    print(f"Step 2: Generated {len(descriptors)} SIFT descriptors.")
    return np.array(descriptors)


# ===============================
#   特征匹配（基于 L2 距离 + 比率检测）
# ===============================
def match_features(des1, des2, ratio_threshold=0.75):
    """
    使用 BFMatcher 和比率测试进行特征匹配。
    """
    matches = []
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return matches

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    for m, n in (pair for pair in knn_matches if len(pair) == 2):
        if m.distance < ratio_threshold * n.distance:
            matches.append(m)

    print(f"Step 3: Found {len(matches)} candidate matches.")
    return matches


# ===============================
#   特征匹配流程
# ===============================
def run_feature_matching_pipeline(ratio_threshold=0.75):
    cwd = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(cwd, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 读取目标图像
    target_path = os.path.join(cwd, 'target.jpg')
    target = cv2.imread(target_path)
    print("Processing target.jpg...")
    kp_t = detect_keypoints(target)
    des_t = compute_sift_descriptors_final(target, kp_t)

    best = {'match': -1}

    # 遍历场景图像
    print("\nprocessing scene images...")
    for i in range(1, 6):
        scene_path = os.path.join(cwd, 'dataset', f'{i}.jpg')
        scene = cv2.imread(scene_path)
        print(f"\nProcessing {i}.jpg...")
        kp_s = detect_keypoints(scene)
        des_s = compute_sift_descriptors_final(scene, kp_s)

        matches = match_features(des_t, des_s, ratio_threshold)
        match = len(matches)
        print(f"Total matches: {match}")

        if match > best['match']:
            best = {'match': match, 'img': scene, 'kp': kp_s, 'path': scene_path, 'matches': matches}


    # 绘制最佳匹配结果
    if best['match'] > 0:
        print(f"\nBest match found: '{best['path']}' with a match of {best['match']}.")
        result = cv2.drawMatches(target, kp_t, best['img'], best['kp'], best['matches'], None, flags=2)
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Best Match(HARRIS): {os.path.basename(best['path'])}, Matches: {best['match']}")
        plt.axis('off')

        out_path = os.path.join(output_dir, "best_match_HARRIS.png")
        plt.savefig(out_path)
        plt.close()
    else:
        print("No effective matches found.")


if __name__ == "__main__":
    run_feature_matching_pipeline()
