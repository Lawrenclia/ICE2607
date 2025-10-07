## Lab1





-------


### 实验概览

本次实验旨在通过编程实现对图像的特征提取和分析。实验的核心内容是利用Python中的**OpenCV**库进行图像处理，并使用**Matplotlib**库绘制图像的各种直方图。本次实验主要研究**颜色直方图**、**灰度直方图**和**梯度直方图**这三种图像特征特征 。

- **颜色直方图**：彩色图像RGB三种颜色分量在整张图像中所占的相对比例，反映了图像全局的主体色调。
某一颜色分量的总能量：
$$
E(c) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} I(x,y,c)
$$
某一颜色分量的能量的相对比例：
$$
H(c) = \frac{E(c)}{\sum_{i=0}^{2}E(i)}
$$



- **灰度直方图**：灰度图像灰度值的分布情况，反映了灰度图像的明暗程度。

图像中灰度值为 $ i $ 的像素总个数为：
$$
N(i) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} I(x,y) == i?1:0
$$
灰度直方图公式：
$$
H(i) = \frac{N(i)}{\sum_{j=0}^{255}N(j)},i=0,\cdots, 255
$$



- **梯度直方图**：灰度图像的梯度强度分布情况，反映了图像的纹理的疏密程度。

把梯度强度均匀分成361个区间，$ (x, y) $ 处的像素所在区间为：
$$
B(x,y)=i, if~~ i \leq M(x,y) < i+1, 0 \leq i \leq 360
$$
落在第i个区间总的像素数目为：        
$$
N(i) = \sum_{x=1}^{W-2}\sum_{y=1}{H-2}B(x,y) == i?1:0
$$

比例为：
$$
H(i) = \frac{N(i)}{\sum_{j=0}^{360}N(j)}
$$



------



### 练习题的解决思路



本次实验的核心任务是为三张图像（img1.jpg, img2.jpg, img3.jpg）分别计算并绘制颜色直方图、灰度直方图和梯度直方图。



#### 流程概览

1. 导入所需的库，包括 `cv2`、`matplotlib.pyplot` 和 `numpy`。
2. 定义一个包含所有图像文件名的列表`images`。
3. 遍历图像列表，对每一张图像执行以下操作：
   - 以彩色方式读入图像，并计算其颜色直方图。
   - 以灰度方式读入图像，并计算其灰度直方图和梯度直方图。
   - 使用 Matplotlib 库将每种直方图绘制并保存为 PNG 文件。



#### 代码内容

```
import cv2
from matplotlib import pyplot as plt
import numpy as np

# 使图片显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块

# 图片列表
images = ['img1', 'img2', 'img3']

for img in images:
  img_color = cv2.imread(f'images/{img}.jpg', cv2.IMREAD_COLOR)
  img_gray = cv2.imread(f'images/{img}.jpg', cv2.IMREAD_GRAYSCALE)

  #绘制颜色直方图
  b, g, r = cv2.split(img_color)
  blue = np.sum(b)
  green= np.sum(g)
  red = np.sum(r)
  total = blue + green + red

  bgr_ratios = [float(blue / total), float(green / total), float(red / total)]
  plt.bar(['蓝色', '绿色', '红色'], bgr_ratios, color=['blue', 'green', 'red'])
  plt.title(f'{img}颜色直方图')
  plt.xlabel('颜色')
  plt.ylabel('占比')
  for i, ratio in enumerate(bgr_ratios):
  	plt.text(i, ratio, f'{ratio:.3f}', ha='center', va='bottom')
  plt.savefig(f'outputs/{img}颜色直方图.png')
  plt.close()

  #绘制灰度直方图
  plt.hist(img_gray.ravel(), bins=256, range=(0, 256), color='black')
  plt.title(f'{img}灰度直方图')
  plt.xlabel('灰度值')
  plt.ylabel('像素数量')

  plt.savefig(f'outputs/{img}灰度直方图.png')
  plt.close()

  #绘制梯度直方图
  grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
  grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
  grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
  plt.hist(grad_magnitude.ravel(), bins=361, range=(0, 360), color='black')
  plt.title(f'{img}梯度直方图')
  plt.xlabel('梯度方向')
  plt.ylabel('梯度强度')
  plt.savefig(f'outputs/{img}梯度直方图.png')
  plt.close()
```



#### 核心代码解释

- **读取图像**: `cv2.imread()` 函数用于读取图像。通过`f'images/{img}.jpg'`参数传入图片所在的相对地址，通过`cv2.IMREAD_COLOR`或`cv2.IMREAD_GRAYSCALE`参数指定读取为彩色图像或者灰度图像。

- **颜色直方图**:

  - 首先，使用 `cv2.split()` 函数将彩色图像分离为蓝`b`、绿`g`、红`r`三个颜色通道。

  - 然后，通过 `np.sum()` 计算每个通道的总像素值。

  - 最后，将每个通道的总像素值除以所有通道的总像素值，得到每个颜色分量的相对比例，并使用 

    `plt.bar()` 绘制柱状图 。

- **灰度直方图**:

  - `plt.hist()` 函数用于绘制直方图。

  - `img_gray.ravel()` 将二维灰度图像数组展平为一维数组，方便 `plt.hist()` 处理。

  - 设置 `bins=256` 和 `range=(0, 256)` 以覆盖所有可能的灰度值，即 0 到 255 。

- **梯度直方图**:

  - 使用 `cv2.Sobel()` 函数分别计算图像在 x 和 y 方向的梯度（`grad_x` 和 `grad_y`）。

  - 使用 `np.sqrt(grad_x**2 + grad_y**2)` 计算梯度强度 。

  - 最后，使用 `plt.hist()` 绘制梯度强度的直方图，通过设置 `bins=361` 和 `range=(0, 360)` 来反映梯度的强度分布 。


在每一张图代码后，添加`plt.savefig()`和`plt.close()`来保存图像。

​    

------



### 代码运行结果

以下是 `lab1.py` 运行后生成的直方图图像。

<div style="display: flex; justify-content: space-around;">
  <img src="D:\Files\openCV\lab1\codes\outputs\img1颜色直方图.png" width="200" alt=img1颜色直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img1灰度直方图.png" width="200" alt="img1灰度直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img1梯度直方图.png" width="200" alt="img1梯度直方图"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="D:\Files\openCV\lab1\codes\outputs\img2颜色直方图.png" width="200" alt="img2颜色直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img2灰度直方图.png" width="200" alt="img2灰度直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img2梯度直方图.png" width="200" alt="img2梯度直方图"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="D:\Files\openCV\lab1\codes\outputs\img3颜色直方图.png" width="200" alt="img3颜色直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img3灰度直方图.png" width="200" alt="img3灰度直方图"/>
  <img src="D:\Files\openCV\lab1\codes\outputs\img3梯度直方图.png" width="200" alt="img3梯度直方图"/>
</div>



------



### 实验结果分析与思考

本次实验通过对三张图像的直方图分析，深入理解了图像的色彩、亮度和纹理特征。

**颜色直方图（色调）**：颜色直方图直观地展示了图像的色彩构成。例如，**img1** 以绿色为主，**img2** 以蓝色为主，而 **img3** 也是绿色占主导。这表明颜色直方图可以快速量化图像的主体色调，但它忽略了像素的空间分布信息。

**灰度直方图（明暗）**：灰度直方图反映了图像的亮度分布。直方图的峰值位置决定了图像的整体明暗程度：

- **img1** 峰值在右侧，图像偏亮。
- **img2** 峰值居中，亮度适中。
- **img3** 峰值在左侧，图像偏暗。

**梯度直方图（纹理复杂度）**：梯度直方图揭示了图像的纹理细节。梯度值越高，表示灰度变化越剧烈，纹理越复杂。

- **img1** 和 **img3** 的梯度直方图峰值集中在低梯度区域，说明纹理简单，平坦区域多。
- **img2** 的梯度分布更广，表明图像包含更多边缘和细节，纹理更复杂。


------

### 拓展思考
1. 示例代码中 `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) `的作用是什么？
    OpenCV 库读取彩色图像时，默认的通道顺序是**BGR**（蓝、绿、红），而不是常见的**RGB**（红、绿、蓝）。`cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) `的作用是将图像从 BGR 通道顺序转换为 RGB 通道顺序 。这个转换对于使用 Matplotlib显示图像非常重要，因为 Matplotlib 默认顺序是 RGB，如果不进行转换，图像的颜色会显示不正确 。

2. 如何使得 pyplot 正确显示灰度图的颜色？
    在` plt.imshow() `中明确指定颜色映射表为**gray**，即`cmap = 'gray'`。这个参数确保了灰度值为0的像素显示为黑色，灰度值为255的像素显示为白色，并且中间的灰度值以相应的灰度级显示，符合我们对灰度图像的视觉认知。

--------

### 实验感想

本次实验让我对数字图像在计算机中的表示有了更深入的理解，特别是灰度图像和彩色图像的存储方式 。通过动手实践，我学会了如何使用 OpenCV 和 Matplotlib 库来处理图像并提取其特征。在编写代码的过程中，我体会到了图像直方图作为一种全局特征的强大之处，它能够简洁地概括图像的整体属性。同时，我也意识到，要准确地理解每一种直方图的含义和计算方式，才能避免出现逻辑上的错误，例如在绘制梯度直方图时误将梯度强度当作梯度方向。未来，我希望能够探索更多图像特征提取的方法，并将其应用于更复杂的图像处理任务中。





注：本报告的撰写过程中，AI（Gemini 2.5 Flash）提供了多方面的协助。在“练习题的解决思路”、“实验结果分析与思考”及“实验感想”三个部分，AI对内容进行了语言润色，使表达更加流畅。此外，“拓展思考”部分的答案也是通过向AI提问得到的。