# Lab3 实验报告



- **姓名：蒋昊翔**
- **学号：524030910083**

- **班级：电院2401**

------



## 一、实验概览



### 1. 实验目的

本次实验的目的是理解并实现`Locality-Sensitive Hashing(LSH)`算法，并将其应用于图像检索任务。

在大型数据集中，传统的`Nearest neighbor (NN) `或`k-nearest neighbor (KNN)`在数据库中检索和输入数据距离最近的 $1$ 个或 $k$ 个数据，一般情况下算法复杂度为$O(n)$（例如暴力搜索），优化情况下可达到$O(log~ n)$（例如二叉树搜索），其中$n$为数据库中的数据量。当数据库很大（即 $N$ 很大时），搜索速度很慢。`LSH` 算法通过设计一种特殊的哈希函数 $g(p)$，使得原始空间中相似的数据点（如特征向量）能以高概率映射到同一个哈希桶（Hash bucket）中，而距离较远的点则以高概率映射到不同的桶中。

这样，在检索时，我们不再需要与数据库中的所有 $N$ 个数据进行比较，而只需计算查询数据 $q$ 的哈希值 $g(q)$，并仅在其对应的桶内进行搜索，速度能够大大加快。



### 2. 实验任务

本实验的具体任务是，利用 `LSH` 算法在图片数据库（`dataset/`）中搜索与目标图片（`target.jpg`）最相似的图片 。主要内容包括：

1. **实现 LSH**：根据 `lab3-LSH.pptx` 中描述的原理，实现图像特征提取、特征量化及 LSH 哈希计算。

2. **实现 NN**：实现一个暴力的近邻搜索（`run_nn_search`）作为对比基准。

3. **对比分析**：对比 `NN` 搜索与 `LSH` 搜索（在不同 $k$ 值下）的执行时间与搜索结果。

   

### 2. 实验环境

- **主要库**：opencv(cv2), numpy, matplotlib，collections

- **文件结构**：

  ```
  ├── target.jpg                  # 目标图像
  ├── lab3.py                     # 代码内容           
  ├── dataset/
  │   ├── 1.jpg                   # 待匹配图像
  │   ├── …………
  │   └── 5.jpg
  └── output/
      ├── result_nn_i.jpg         #NN算法
      ├── result_lsh_k8_i.jpg     #LSH算法（k = 8）
      ├── result_lsh_k16_i.jpg    #LSH算法（k = 16）
      └── result_lsh_k24_i.jpg    #LSH算法（k = 24）
  ```

------



## 二、 练习题的解决思路



### 1. 图像特征提取 (`extract_features`)

根据实验要求，每幅图像使用一个12维的颜色直方图 $p$ 来表示。

1. 使用 `cv2.imread` 读取图像，并将其从 BGR 转换为 RGB 色彩空间。
2. 使用 `cv2.calcHist` 分别计算 R、G、B 三个通道的颜色直方图，每个通道分为 4 个 bin。
3. 将三个直方图（$4+4+4$）拼接（`np.concatenate`）成一个 12 维的特征向量。
4. 对该向量进行归一化（除以向量所有元素之和），使其满足 $p_i \in [0, 1]$。



### 2. 特征向量量化 (`quantize_vector`)

为了将特征向量 $p$ 映射到 LSH 所需的整数空间，我们对其进行量化。

1. 将 12 维向量 $p$ 的每个分量 $p_i$ 映射为 0, 1, 2 三个整数值之一。

   映射规则为：
   - $p_i \in [0, 0.3) \to 0$
   - $p_i \in [0.3, 0.6) \to 1$
   - $p_i \in [0.6, 1] \to 2$

2. `lab3.py` 中的实现通过 `np.zeros_like` 初始化为0，然后将 $\ge 0.3$ 的设为1，再将 $\ge 0.6$ 的设为2，以实现该过程。

4. 最终得到一个 12 维的整数向量 $p_{quantized}$，其中 $p_i \in \{0, 1, 2\}$。



### 3. LSH 哈希计算 (`calculate_hash`)

这是 LSH 算法的核心，代码中根据 `lab3-LSH.pptx` 提供的简化公式直接计算其在投影集 $I$ 上的投影（即哈希值） $g(p)$。

1. **输入**：12维量化向量 `p_quantized`、投影集 `projection_set_I`（一个包含 $k$ 个索引的列表，索引范围 $1 \sim 36$）、$d=12$、$C=3$。
2. **遍历 $d$ 个维度**：对 `p_quantized` 的 12 个维度进行循环（`for i in range(d)`）。
3. **计算 $I_i$**：对于第 $i$ 维，其在 36 维 Hamming 空间中对应的范围是 `[i*C+1, (i+1)*C]`。`I_i` 是指投影集 `projection_set_I` 中所有落入此范围的索引。
4. **计算 $p_i$ 阈值**：根据 `lab3-LSH.pptx` 的公式，第 $i$ 维的量化值 $p_i$ 对应的 Hamming 空间阈值为 `threshold = i * C + p_i`。
5. **计算1的个数**：在 $I_i$ 中，计算有多少个索引值小于等于 `threshold`，这个数量记为 `num_ones`。
6. **生成投影**：该维度 $i$ 上的投影即是 `num_ones` 个 1 后面跟着 `(len(I_i) - num_ones)` 个 0。
7. **拼接**：将 12 个维度各自生成的投影（0和1的序列）全部拼接（`hash_val.extend(proj_i)`），得到最终的哈希值。
8. **返回**: 一个元组（`tuple(hash_val)`）使其可作为字典的键。



### 4. 检索流程(`run_xx_search`)

- #### NN 搜索 (`run_nn_search`)

NN 搜索（暴力搜索）的流程如下：

1. 遍历数据库中所有图像的原始特征向量（`db_features`）。
2. 使用 `np.linalg.norm` 计算目标图像特征与每张数据库图像特征之间的欧氏距离。
3. 返回距离最小的图像 `best_match_nn`。



- #### LSH 搜索 (`run_lsh_search`)

LSH 搜索分为预处理 (build) 和检索 (query) 两步：

1. **预处理（构建哈希表）**：
   - 初始化一个哈希桶（`defaultdict(list)`）。
   - 遍历所有数据库图像的量化特征向量（`db_quantized`）。
   - 使用 `calculate_hash` 计算每张图像的哈希键 `hash_key`。
   - 将图像名 `img_name` 存入 `hash_buckets[hash_key]` 对应的列表中。
2. **检索**：
   - 计算目标图像（`target_quantized`）的哈希键 `query_key`。
   - 从哈希表中获取对应的候选列表 `candidate_list = hash_buckets.get(query_key, [])`。
   - **关键**：只遍历 `candidate_list` 中的图像（而不是整个数据库）。
   - 在桶内，使用原始特征向量（`db_features`）计算欧氏距离。
   - 返回桶内距离最小的图像 `best_match_lsh`。



### 5. 结果可视化 (`create_image`)

为了直观地对比检索结果，使用 `matplotlib` 库编写了 `create_image` 函数。该函数读取目标图像和匹配到的结果图像，将它们并排绘制在一张图上，并添加标题，最后保存到 `output/` 文件夹中。



### 6. 主函数 (`main`)

1. 设置 $k$ 值列表 `k_values = [8, 16, 24]`。
2. 从 $1 \sim 36$（`D_PRIME_DIM`）的范围内随机采样，为每个 $k$ 值生成一个投影集 `projection_sets[k]`。
3. 加载并预处理（提取特征、量化）`dataset/` 中的所有图像和 `target.jpg`。
4. **进行NN算法计算**：运行 `run_nn_search` 5 次，记录并打印平均查询时间，绘图。
5. **进行LSH哈希计算**：循环 $k$ 值，对每个 $k$ 运行 `run_lsh_search` 5 次，记录并打印构建时间、查询时间、桶大小和平均时间，绘图。




---



## 三、代码运行结果



代码运行后，会在`output`文件夹下生成20张结果图（`NN`、`LSH k=8、k=16、k=24` 各5张）和一个 `terminal_output.txt` 日志文件。 

### **控制台输出（`terminal_output.txt`)：**

```
Loading and processing images
Loading and processing target image
Preprocessing complete.

--- Running NN Search ---
1th NN Search:
	Result: 38.jpg
	Total Time: 0.000133 s
2th NN Search:
	Result: 38.jpg
	Total Time: 0.000216 s
3th NN Search:
	Result: 38.jpg
	Total Time: 0.000145 s
4th NN Search:
	Result: 38.jpg
	Total Time: 0.000151 s
5th NN Search:
	Result: 38.jpg
	Total Time: 0.000145 s

Average NN Search Time over 5 runs: 0.000158 s

--- Running LSH Search (k=8) ---

1th LSH Search (k=8):
	Result: 38.jpg
	Total Time (Build + Query): 0.000643 s
	(Build: 0.000498 s, Query: 0.000145 s, Bucket Size: 50)

2th LSH Search (k=8):
	Result: 38.jpg
	Total Time (Build + Query): 0.000558 s
	(Build: 0.000403 s, Query: 0.000155 s, Bucket Size: 50)

3th LSH Search (k=8):
	Result: 38.jpg
	Total Time (Build + Query): 0.000544 s
	(Build: 0.000407 s, Query: 0.000137 s, Bucket Size: 50)

4th LSH Search (k=8):
	Result: 38.jpg
	Total Time (Build + Query): 0.000546 s
	(Build: 0.000406 s, Query: 0.000140 s, Bucket Size: 50)

5th LSH Search (k=8):
	Result: 38.jpg
	Total Time (Build + Query): 0.000569 s
	(Build: 0.000422 s, Query: 0.000147 s, Bucket Size: 50)

Average LSH Build Time over 5 runs (k=8): 0.000427 s
Average LSH Query Time over 5 runs (k=8): 0.000145 s
Average LSH Search Time over 5 runs (k=8): 0.000572 s

--- Running LSH Search (k=16) ---

1th LSH Search (k=16):
	Result: 38.jpg
	Total Time (Build + Query): 0.000766 s
	(Build: 0.000621 s, Query: 0.000145 s, Bucket Size: 50)

2th LSH Search (k=16):
	Result: 38.jpg
	Total Time (Build + Query): 0.000839 s
	(Build: 0.000616 s, Query: 0.000223 s, Bucket Size: 50)

3th LSH Search (k=16):
	Result: 38.jpg
	Total Time (Build + Query): 0.000940 s
	(Build: 0.000766 s, Query: 0.000174 s, Bucket Size: 50)

4th LSH Search (k=16):
	Result: 38.jpg
	Total Time (Build + Query): 0.000852 s
	(Build: 0.000615 s, Query: 0.000236 s, Bucket Size: 50)

5th LSH Search (k=16):
	Result: 38.jpg
	Total Time (Build + Query): 0.000774 s
	(Build: 0.000625 s, Query: 0.000148 s, Bucket Size: 50)

Average LSH Build Time over 5 runs (k=16): 0.000649 s
Average LSH Query Time over 5 runs (k=16): 0.000185 s
Average LSH Search Time over 5 runs (k=16): 0.000834 s

--- Running LSH Search (k=24) ---

1th LSH Search (k=24):
	Result: 38.jpg
	Total Time (Build + Query): 0.000960 s
	(Build: 0.000815 s, Query: 0.000144 s, Bucket Size: 49)

2th LSH Search (k=24):
	Result: 38.jpg
	Total Time (Build + Query): 0.000947 s
	(Build: 0.000803 s, Query: 0.000144 s, Bucket Size: 49)

3th LSH Search (k=24):
	Result: 38.jpg
	Total Time (Build + Query): 0.001000 s
	(Build: 0.000851 s, Query: 0.000149 s, Bucket Size: 49)

4th LSH Search (k=24):
	Result: 38.jpg
	Total Time (Build + Query): 0.001151 s
	(Build: 0.000999 s, Query: 0.000152 s, Bucket Size: 49)

5th LSH Search (k=24):
	Result: 38.jpg
	Total Time (Build + Query): 0.000957 s
	(Build: 0.000810 s, Query: 0.000147 s, Bucket Size: 49)

Average LSH Build Time over 5 runs (k=24): 0.000856 s
Average LSH Query Time over 5 runs (k=24): 0.000147 s
Average LSH Search Time over 5 runs (k=24): 0.001003 s
```



### **结果图像：**

- `result_nn.jpg`:

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab3\codes\output\result_nn_1.jpg" width="370" alt="nn_1"/>
    <img src="D:\Files\openCV\lab3\codes\output\result_nn_2.jpg" width="370" alt="nn_2"/>
</div>
<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab3\codes\output\result_nn_3.jpg" width="250" alt="nn_3"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_nn_4.jpg" width="250" alt="nn_4"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_nn_5.jpg" width="250" alt="nn_5"/>
</div>



---


- `result_lsh_k8.jpg`:

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k8_1.jpg" width="370" alt="k8_1"/>
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k8_2.jpg" width="370" alt="k8_2"/>
</div>
<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k8_3.jpg" width="250" alt="k8_3"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k8_4.jpg" width="250" alt="k8_4"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k8_5.jpg" width="250" alt="k8_5"/>
</div>



---


- `result_lsh_k16.jpg`:

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k16_1.jpg" width="370" alt="k16_1"/>
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k16_2.jpg" width="370" alt="k16_2"/>
</div>
<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k16_3.jpg" width="250" alt="k16_3"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k16_4.jpg" width="250" alt="k16_4"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k16_5.jpg" width="250" alt="k16_5"/>
</div>



---


- `result_lsh_k24.jpg`:

<div style="display: flex; justify-content: center; gap: 0px;">
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k24_1.jpg" width="370" alt="k24_1"/>
    <img src="D:\Files\openCV\lab3\codes\output\result_lsh_k24_2.jpg" width="370" alt="k24_2"/>
</div>
<div style="display: flex; justify-content: center; gap: 0px;">
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k24_3.jpg" width="250" alt="k24_3"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k24_4.jpg" width="250" alt="k24_4"/>
	<img src="D:\Files\openCV\lab3\codes\output\result_lsh_k24_5.jpg" width="250" alt="k24_5"/>
</div>



---



## 四、实验结果分析与思考



根据输出的结果，我进行如下分析：

### 1. 结果准确性分析

- **NN 搜索**：5 次运行均准确找到了 `38.jpg`。
- **LSH (k=8)**：5 次运行均准确找到了 `38.jpg`。
- **LSH (k=16)**：5 次运行均准确找到了 `38.jpg`。
- **LSH (k=24)**：5 次运行均准确找到了 `38.jpg`。

**结论**：在本次实验中，`LSH 算法` 是有效的。它成功地将目标图像及其最近邻（`38.jpg`）哈希到了同一个桶中，因此检索结果与 `NN算法` 搜索一致。



### 2. 检索效率分析

为了更准确地分析，我们根据5次运行数据计算平均时间：

| **方法**       | **平均构建时间 (Build)** | **平均查询时间 (Query)** | **总时间 (Build+Query)** |
| -------------- | ------------------------ | ------------------------ | ------------------------ |
| **NN **        | **无Build过程**          | **0.000158 s**           | **0.000158 s**           |
| **LSH (k=8)**  | 0.000427 s               | 0.000145 s               | 0.000572 s               |
| **LSH (k=16)** | 0.000649 s               | 0.000185 s               | 0.000834 s               |
| **LSH (k=24)** | 0.000856 s               | 0.000147 s               | 0.001003 s               |

从上表可以得出几个关键结论：

1. **LSH 查询时间**：LSH 的查询时间（Query Time）确实略快于 NN 搜索，或至少在同一数量级。NN 平均为 0.158ms，而 LSH (k=8) 仅为 0.145ms。这验证了 LSH 的核心思想：通过缩小搜索范围（仅搜索一个桶）来加速查询。
2. **LSH 总时间**：LSH 需要一个构建哈希表（Build Time）的预处理步骤。在此实验中，这个构建时间（0.4ms - 0.8ms）远大于查询时间。
3. **二者比较**：对于单次查询任务，LSH 的总时间（Build + Query）远慢于 NN 搜索。例如 LSH (k=8) 的总时间是 NN 的 3 倍左右（0.572ms 何 0.158ms）。



### 3. LSH 为什么在本实验中更慢？

**根本原因：数据集太小。**

LSH 算法的优势在于处理海量数据。它的时间复杂度为：

- 预处理（构建哈希表）：$O(N)$，其中 $N$ 是数据库大小。
- 查询：$O(1)$ 或 $O(k)$（取决于实现和数据分布），理想情况下与 $N$ 无关。

而暴力 NN 搜索的复杂度为：

- 预处理：$O(1)$
- 查询：$O(N)$

在本次实验中，数据库大小 $N = 50$，NN 搜索的 $O(N)$ 复杂度带来的开销（0.158ms）本就微乎其微。LSH 算法 $O(N)$ 的构建开销（0.4ms - 0.8ms）反而成为了主要负担，其 $O(1)$ 查询所节省的时间完全不足以弥补构建哈希表的代价。

LSH 的优势体现在 $N$ 达到百万级或千万级时。那时 NN 的 $O(N)$ 查询可能需要几秒钟甚至几分钟，而 LSH 仍然可以保持毫秒级的查询速度（在一次 $O(N)$ 预处理之后）。



### 4. $k$ 值（投影集大小）的影响

- **对构建时间的影响**：随着 $k$ 从 8 增加到 16 和 24，`calculate_hash` 函数需要计算更长的哈希键，`Build Time` 也随之增加（从 0.42ms 增加到 0.86ms 左右）。
- **对桶大小的影响**：理论上， $k$ 越大，哈希键越长，区分度越高，桶应该越小。但在本实验中，桶大小基本不变（49 或 50）。这说明我们所有的 $N \approx 50$ 张图片，即使在 $k=24$ 时，依然全部哈希到了同一个桶里。
- **原因**：这可能是因为
  - 数据集内的图像在颜色直方图特征上本就高度相似；
  - 12 维的颜色直方图特征本身区分度不高；、
  - $[0, 0.3, 0.6]$ 的量化区间划分不合理。



### 5. 拓展思考

1. 检索效果符合预期吗？

   符合预期。检索出的 38.jpg 与目标图像在颜色分布上高度相似（例如，两者都以黄色、褐色和少量深色为主），这证明了基于颜色直方图的 LSH 检索是有效的。

2. 相似性体现在哪里？

   体现在12维颜色直方图向量的欧氏距离上。LSH 算法的设计使得它们在量化和哈希后依然落入同一（或邻近）的桶中。

3. 能否设计其他特征？

   可以。12维颜色直方图是一个非常粗糙的全局特征，它丢失了所有空间和纹理信息。如果使用更鲁棒的特征，如 SIFT（ Lab2 所用）、SURF、ORB，或者现代的深度学习特征（如 CNN 提取的特征向量），LSH 的检索效果（尤其是区分度）会好得多。

---



## 五、实验感想

本次实验让我对 LSH（局部敏感哈希）有了非常深刻的理解。

首先，我通过 `lab3.py` 的代码，亲手实现了 LSH 的完整流程：从特征提取（颜色直方图）、特征量化（三区间量化），到最核心的哈希计算。这让我明白了 LSH 理论是如何一步步转化为代码的。

其次，实验结果让我认识到算法的适用场景。LSH 并不是在所有情况下都优于暴力 NN 搜索。实验数据明确显示，在一个 $N = 50$ 的极小数据集上，LSH 的预处理（Build Time）开销远大于它在查询时节省的时间，导致其总体性能劣于 NN 搜索。这让我体会到，LSH 的真正威力在于处理传统 $O(N)$ 算法无法承受的海量数据集，它是一种用空间（哈希表）和预处理时间来换取极快查询时间（$O(1)$）的算法。

最后，我们使用的12维颜色直方图特征，导致 $k$ 值增大到24时，所有图像依然哈希到同一个桶，这说明特征的区分度很低。这也启发了我，LSH 的效果不仅依赖于哈希函数本身，也高度依赖于输入特征的质量。

---



## 六、源程序



###  `lab3.py`

```python
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
        ave_total_time = 0
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
                ave_total_time += (build_time + query_time)
            print(f"\n{i + 1}th LSH Search (k={k}):")
            print(f"\tResult: {lsh_match}")
            total_time = build_time + query_time
            print(f"\tTotal Time (Build + Query): {total_time:.6f} s")
            print(f"\t(Build: {build_time:.6f} s, Query: {query_time:.6f} s, Bucket Size: {bucket_size})")
        print(f"\nAverage LSH Search Time over 5 runs (k={k}): {ave_total_time / 5:.6f} s")

if __name__ == "__main__":
    main()
```
