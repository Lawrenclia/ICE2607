import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
TARGET_IMG_PATH = os.path.join(BASE_DIR, "target.jpg")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 加载目标图像
img_target = cv2.imread(TARGET_IMG_PATH, cv2.IMREAD_GRAYSCALE)

# 初始化 SIFT
sift = cv2.SIFT_create()

# 计算目标图像的特征点和描述子
kp_target, des_target = sift.detectAndCompute(img_target, None)

# 初始化特征匹配器
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# crossCheck=False: 我们将使用 k-NN 匹配，所以这里设为 False。
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 初始化变量以记录最佳匹配结果
max_good_matches = 0  # 记录最高的“好的匹配”数量
best_match_info = {
    "image_name": None,      # 最佳匹配图像的文件路径
    "result_img": None,      # 最终绘制了匹配结果的图像
    "matches_count": 0       # 最佳匹配的“好的匹配”数量
}

# 遍历图像
for i in range(1, 6):
    scene_img_path = os.path.join(DATASET_DIR, f"{i}.jpg")

    img_scene = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    # 计算图像的特征点和描述子
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    # 使用 k-NN 算法进行特征匹配 
    matches = bf.knnMatch(des_target, des_scene, k=2)

    # 应用 Lowe's Ratio Test 筛选优质匹配点
    good_matches = []
    if matches and len(matches[0]) == 2:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    print(f"Found {len(good_matches)} good matches with {i}.jpg.")

    # 检查当前是否为最佳匹配，并生成结果图 
    if len(good_matches) > max_good_matches:
        max_good_matches = len(good_matches)
        
        img_best = cv2.imread(scene_img_path)
        
        # 当有足够多的匹配点时，尝试计算变换矩阵来框出目标
        if len(good_matches) > 10:
            # 获取两张图片中所有“好的匹配点”的坐标
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算单应性矩阵(Homography Matrix)。它描述了物体在两个不同视角下的几何变换关系。
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                matchesMask = mask.ravel().tolist()
                
                # 获取目标图像的四个角点坐标
                h, w = img_target.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
                # 使用计算出的单应性矩阵 M，将目标图像的角点变换到场景图像中的对应位置。
                dst = cv2.perspectiveTransform(pts, M)

            else:
                matchesMask = None # 如果单应性矩阵计算失败
        else:
            matchesMask = None # 匹配点太少，不进行矩阵计算

        # 设置绘图参数
        draw_params = dict(matchColor = (-1, -1, -1, -1), # 设置为-1，OpenCV会为每条线随机选择颜色
                           singlePointColor = None,       # 不单独绘制特征点
                           matchesMask = matchesMask,     # 只绘制RANSAC筛选后的内点匹配
                           flags = 2)

        # 绘制匹配结果
        img_result = cv2.drawMatches(cv2.imread(TARGET_IMG_PATH), kp_target, img_best, kp_scene, good_matches, None, **draw_params)

        # 更新最佳匹配信息
        best_match_info["image_name"] = f"{i}.jpg"
        best_match_info["result_img"] = img_result
        best_match_info["matches_count"] = len(good_matches)


#  显示结果
if best_match_info["result_img"] is not None:
    print(f"The best match is: '{best_match_info['image_name']}'")
    print(f"Found {best_match_info['matches_count']} good matches.")

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(best_match_info["result_img"], cv2.COLOR_BGR2RGB))
    plt.title(f"Best Match(SIFT): {best_match_info['image_name']}, Matches: {best_match_info['matches_count']}")
    plt.axis('off') 

    output_path = os.path.join(OUTPUT_DIR, "best_match_SIFT.png")
    plt.savefig(output_path)
    plt.close()
else:
    print("No effective matches found.")
