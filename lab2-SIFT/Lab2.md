# Lab2 实验报告



------



## 一、实验概览



### 1. 实验目的

本实验旨在深入理解并实践经典的 SIFT (Scale-Invariant Feature Transform) 算法。主要任务是在 `dataset` 文件夹中的一组场景图像中，搜索并匹配 `target.jpg` 图像中的目标物体。

为达到此目的，实验将通过两种方式完成：

1. **手动实现核心算法**：使用简化的方法（Harris 角点 + 图像金字塔）进行多尺度特征点提取，但独立编程实现 SIFT 特征描述子的计算。
2. **使用标准库函数**：直接调用 OpenCV 内置的 SIFT 函数库完成整个匹配流程。

最终，通过对比两种方法的结果，分析手动实现与标准库在性能等方面上的差异，从而加深对 SIFT 算法内在机制的理解。



### 2. 实验环境

- **主要库**：OpenCV (cv2), NumPy, Matplotlib

- **文件结构**：

  ```
  ├── target.jpg                # 目标图像
  ├── lab2_HARRIS.py            # 自定义描述子实现
  ├── lab2_SIFT.py              # OpenCV SIFT 实现
  ├── dataset/
  │   ├── 1.jpg                 # 场景图像
  │   ├── 2.jpg
  │   ├── 3.jpg
  │   ├── 4.jpg
  │   └── 5.jpg
  └── output/
      ├── best_match_HARRIS.png # 自定义方法匹配结果
      └── best_match_SIFT.png   # OpenCV SIFT 库匹配结果
  ```

------



## 二、 练习题的解决思路



### 1. 基于OpenCV内置SIFT函数的实现 (`lab2_SIFT.py`)

本方法直接调用`cv2`库提供的SIFT相关功能，流程清晰高效。

**核心流程概览：**

1. **初始化**：加载目标图像`target.jpg`，并创建一个SIFT检测器对象 `cv2.SIFT_create()`。

2. **目标特征提取**：对目标图像使用`sift.detectAndCompute()`方法，一次性提取出关键点 `kp_scene`和它们对应的SIFT描述子`des_scene`。

3. **遍历场景图像**：循环读取`dataset`文件夹中的每一张待匹配图像。

4. **场景特征提取**：对每一张场景图像，同样使用`sift.detectAndCompute()`提取其关键点和描述子。

5. **特征匹配（k-NN算法）**：用`bf.knnMatch(des_target, des_scene, k=2)`方法进行k-近邻匹配（k=2），即为目标图像中的每个描述子，在场景图像中寻找两个最相似的描述子。

6. **筛选优质匹配（Lowe's Ratio Test）**：

   - 遍历`knnMatch`返回的匹配对 `(m, n)`。
   - 如果第一个匹配`m`的距离小于第二个匹配`n`的距离的0.75倍（`m.distance < 0.75 * n.distance`），则认为这是一个足够好的、具有区分度的匹配点，并将其保留。

7. **寻找最佳匹配图像**：

   - 在所有场景图像中，记录下拥有最多“优质匹配点”的图像作为最佳匹配结果。
   - 当优质匹配点数量超过10个时，使用`cv2.findHomography`来计算变换矩阵，以验证匹配点的几何一致性。
8. **结果可视化**：使用`cv2.drawMatches`函数将目标图像和最佳匹配场景图像及其匹配关系绘制出来，并使用`plt.savefig()`保存结果。 

---



### 2. 自定义关键点检测与SIFT描述子实现 (`lab2_HARRIS.py`)

本方法遵循了SIFT的核心思想，但对关键点检测步骤进行了简化，并手动实现了描述子的生成过程。 

**核心流程：**

1. **多尺度关键点检测** (`detect_keypoints`): 
- **构建图像金字塔**：通过循环调用`cv2.resize`函数，将原始灰度图像按固定比例（如0.75）进行多次缩小，生成一系列不同尺寸的图像。这个金字塔模拟了在不同尺度下观察物体的效果 。
	- **逐层角点检测**：在金字塔的每一层图像上，使用`cv2.goodFeaturesToTrack`函数进行角点检测。该函数是Harris角点检测的一个优化版本，能有效找出图像中的关键特征点 。
	
- **坐标与尺度归一化**：将在较小尺度图像上检测到的角点坐标，按比例换算回原始图像的坐标系中。同时，将当前金字塔层级的尺度信息保存为`cv2.KeyPoint`对象的大小（size），从而记录下每个关键点被发现时的尺度。


2. **自定义SIFT描述子计算** (`compute_sift_descriptors_final`)

- **分配主方向 (Orientation Assignment)**

  - **目的**：确定一个标准方向，使得后续的描述子计算都基于这个方向进行，从而实现旋转不变性。

    

  - **实现步骤**：

    1. 在关键点周围的邻域内，计算每个像素的梯度幅值和方向 。

    2. 创建一个36个区间的方向直方图（每个区间10度）。

    3. 使用高斯加权的梯度幅值为直方图投票 。离关键点中心越近的像素，其投票权重越大。

    4. 直方图的峰值所对应的方向，即为该关键点的主方向。

       

- **生成描述子 (Descriptor Generation)**

  - **目的**：将关键点邻域内的图像信息转化为一个紧凑且具有高区分度的特征向量。

    

  - **实现步骤**：

    1. 在关键点周围定义一个16x16的像素窗口。

    2. 将这个窗口根据刚刚计算出的主方向进行旋转，以对齐坐标系。

    3. 将旋转后的16x16窗口划分为一个4x4的网格，每个网格是4x4的子区域。

    4. 在每个4x4的子区域内，计算一个8个区间的梯度方向直方图。

    5. 将这16个子区域的8维直方图向量按顺序拼接起来，形成一个`16 * 8 = 128`维的最终描述子。

    6. 对这个128维向量进行归一化处理，以降低光照变化的影响。

       


3. **特征匹配** (`match_features`)

- **目的**：负责比较目标图像和场景图像的描述子，找出匹配的特征点对。



- **实现步骤**：

  - **暴力匹配 (Brute-Force)**：使用`cv2.BFMatcher`，它会计算目标图像中的每个描述子与场景图像中所有描述子之间的欧氏距离 23。

  - **K-近邻匹配**：设置`k=2`，对于目标图像的每个描述子，找出在场景图像中与它最相似的两个描述子（即距离最近的两个）。

  - **Lowe's Ratio Test (比率测试)**：这是筛选可靠匹配的关键。对于每一组匹配（最佳匹配m和次佳匹配n），只有当最佳匹配的距离显著小于次佳匹配的距离时（例如，`m.distance < 0.75 * n.distance`），才认为这是一个有效、无歧义的匹配。



4. **整体流程与可视化** (`run_feature_matching_pipeline`)主函数

- **实现步骤**：
  - 加载目标图像 (`target.jpg`)，并调用上述函数提取其关键点和SIFT描述子。
  - 遍历`dataset`文件夹中的每一张场景图像。
  - 对每张场景图像，重复提取关键点和描述子的过程。
  - 调用特征匹配函数，找出当前场景图像与目标图像之间的所有优质匹配点。
  - 比较并记录哪一张场景图像产生了最多的匹配点。
  - 最后，使用`cv2.drawMatches`函数将目标图像与最佳匹配的场景图像并排显示，并用线条连接所有成功匹配的特征点对，将结果可视化并保存。


---



## 三、代码运行结果

代码运行后，会在`output`文件夹下生成两张结果图，分别对应两种方法的最佳匹配结果。 

### **控制台输出：**

- **`lab2_SIFT.py`**

  ```
  Found 54 good matches with 1.jpg.
  Found 53 good matches with 2.jpg.
  Found 427 good matches with 3.jpg.
  Found 72 good matches with 4.jpg.
  Found 40 good matches with 5.jpg.
  The best match is: '3.jpg'
  Found 427 good matches.
  ```

- **`lab2_HARRIS.py`**

  ```
  Processing target.jpg...
  Step 1: Found 1182 keypoints.
  Step 2: Generated 1182 SIFT descriptors.
  
  processing scene images...
  
  Processing 1.jpg...
  Step 1: Found 754 keypoints.
  Step 2: Generated 754 SIFT descriptors.
  Step 3: Found 2 candidate matches.
  Total matches: 2
  
  Processing 2.jpg...
  Step 1: Found 973 keypoints.
  Step 2: Generated 973 SIFT descriptors.
  Step 3: Found 18 candidate matches.
  Total matches: 18
  
  Processing 3.jpg...
  Step 1: Found 961 keypoints.
  Step 2: Generated 961 SIFT descriptors.
  Step 3: Found 205 candidate matches.
  Total matches: 205
  
  Processing 4.jpg...
  Step 1: Found 404 keypoints.
  Step 2: Generated 404 SIFT descriptors.
  Step 3: Found 1 candidate matches.
  Total matches: 1
  
  Processing 5.jpg...
  Step 1: Found 683 keypoints.
  Step 2: Generated 683 SIFT descriptors.
  Step 3: Found 5 candidate matches.
  Total matches: 5
  
  Best match found: 'd:\Files\openCV\lab2\codes\dataset\3.jpg' with a match of 205.
  ```



### **结果图像：**

- `best_match_SIFT.png`:


<div style="display: flex; justify-content: space-around;">
    <img src="D:\Files\openCV\lab2\codes\output\best_match_SIFT.png" width="700" alt=SIFT"/>
  </div>
图像右侧为场景图像（一架直升机），左侧为目标图像。大量的彩色线条连接了场景图像与目标图像中的对应特征点，线条密集地覆盖了直升机的主要结构，显示出极高的匹配度。

---


- `best_match_HARRIS.png`:


<div style="display: flex; justify-content: space-around;">
    <img src="D:\Files\openCV\lab2\codes\output\best_match_HARRIS.png" width="700" alt=HARRIS"/>
  </div>
与图1类似，图中也通过彩色线条展示了匹配关系。虽然线条同样准确地连接了目标与场景，但其数量和密度明显低于使用标准SIFT算法的结果。

---



## 四、实验结果分析与思考



本次实验的两种方法都成功完成了图像检索任务，验证了SIFT特征描述子在物体识别中的有效性。通过对比两张结果图（`best_match_SIFT.jpg` 和 `best_match_HARRIS.jpg`）及实验过程，可以得出以下结论：



- **整体效果对比**

  -   **匹配质量与数量**：OpenCV内置的SIFT算法效果远优于自定义实现。从结果图可见，OpenCV版本找到了**427**个匹配点对，而自定义的Harris角点结合SIFT描述子的方法仅找到**205**个匹配点对。前者匹配点数量更多、分布更密集，覆盖了目标的更多细节，鲁棒性更强。
  -   **运行效率**：在实际运行中，OpenCV的SIFT算法速度非常快，而自定义的Python脚本运行时间则长得多。


- **自定义实现效果不佳的原因分析**

  - **关键点检测策略的简化**：我采用了简化的“多尺度Harris角点检测”方案来替代标准SIFT算法中基于高斯差分金字塔（DoG）的极值点检测。这种简化可能导致了两个问题：一是检测到的稳定关键点数量不足；二是这些关键点的特征区分度不够强，直接影响了后续的匹配成功率。

  - **SIFT描述子计算的局限性**：我自行实现的SIFT描述子计算方法相对简单，尤其是在处理图像边界、插值计算等方面可能不够完善。这可能导致在提取关键点邻域信息时不够精确，生成的128维描述子无法充分表达关键点的独特性，从而导致效果较差。

  - **代码实现效率**：整个流程，特别是关键点检测和描述子计算部分，是使用Python循环实现的。相对于OpenCV底层优化的C++代码，Python的执行效率较低，导致整体运行时间过长。

- **OpenCV实现表现优异的原因分析**

  - **完备的关键点检测**：OpenCV使用了更复杂和成熟的高斯差分金字塔来寻找尺度空间的极值点，并包含了关键点精确定位、去除边缘响应等步骤。这确保了检测出的关键点具有高度的稳定性和重复性。
  - **精确、高效的描述子计算**：OpenCV的描述子计算方法经过了高度优化和严格实现，能够更好地提取关键点的局部特征，生成区分度极高的描述子，从而显著提高匹配的准确率。
  - **底层实现**：OpenCV的算法由C++编写，其运行速度都远超纯Python实现。其使用的`knnMatch`匹配算法本身虽然也是暴力匹配，但得益于高效的底层代码，速度非常快。

综上所述，标准SIFT算法凭借其在尺度空间中检测稳定极值点的完备理论，在特征检测的全面性和鲁棒性上优于简化的多尺度Harris角点法，因此能够找到更多、更可靠的匹配点。虽然我的自定义算法验证了SIFT的基本思想，但与工业级的OpenCV库相比，在关键点检测的鲁棒性和代码实现的效率上还有巨大差距。

---



## 五、实验感想

这次实验让我收获很大。亲手实现SIFT算法的核心部分，让我对高斯金字塔、主方向分配这些理论知识有了更具体的理解。通过对比自己写的简化版（用Harris角点）和OpenCV的标准SIFT，我直观地看到了完整算法和简化方案的差距。虽然我的方法也能跑出结果，但效果和稳定性远不如标准SIFT，这也让我认识到在实际应用中，算法的复杂度和效果之间需要权衡。

最后，这次实验让我真实体会到了OpenCV这类专业库的强大。自己写的Python代码跑起来很慢，而OpenCV运行快、结果好，这让我明白了底层优化的重要性。整个编码和调试的过程也是一次很好的锻炼，让我把理论用到了实践中。

---



## 六、源程序



### 1. `lab2_HARRIS.py`

```python
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

```



### 2. `lab2_SIFT.py`

```python
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
```