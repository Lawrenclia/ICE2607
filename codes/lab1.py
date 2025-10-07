import cv2
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块


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



