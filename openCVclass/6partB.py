import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def hog_feature_extraction(image_path):
    """
    对图片进行 HOG 特征提取，并绘制 HOG 归一化后的直方图
    :param image_path: 输入图像路径
    """
    # 读取灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用 skimage 提供的 HOG 提取方法
    hog_features, hog_image = hog(
        img,
        orientations=9,  # 梯度方向的分箱数量
        pixels_per_cell=(8, 8),  # 每个单元格的像素数
        cells_per_block=(2, 2),  # 每个块包含的单元格数量
        block_norm='L2-Hys',  # 块归一化方法
        visualize=True,  # 返回 HOG 特征图
        feature_vector=True  # 返回特征向量
    )
    # 归一化直方图
    normalized_histogram, bin_edges = np.histogram(hog_features, bins=50, density=True)
    # 绘制结果
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("原始图片")
    plt.axis('off')

    # HOG 特征图
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[:-1], normalized_histogram, color='blue')
    plt.title("HOG 特征归一化直方图")
    plt.xlabel("特征值")
    plt.ylabel("归一化频率")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# 测试函数
image_path = 'my.jpg'  # 替换为你的图片路径
hog_feature_extraction(image_path)
