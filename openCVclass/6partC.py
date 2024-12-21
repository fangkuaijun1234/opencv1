#对图片进行Harris 角点检测

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def harris_corner_detection(image_path, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    对图像进行 Harris 角点检测
    :param image_path: 输入图像路径
    :param block_size: 角点检测的邻域大小
    :param ksize: Sobel 算子使用的窗口大小
    :param k: Harris 角点检测的自由参数
    :param threshold: 角点响应值的阈值
    """
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 使用 Harris 角点检测
    dst = cv2.cornerHarris(gray, block_size, ksize, k)

    # 结果进行膨胀，便于显示角点
    dst = cv2.dilate(dst, None)

    # 设置阈值，保留角点
    img[dst > threshold * dst.max()] = [0, 0, 255]  # 标记角点为红色

    # 显示结果
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Harris 角点检测")
    plt.axis('off')
    plt.show()

# 调用函数，输入图像路径
image_path = 'my.jpg'  # 替换为你的图片路径
harris_corner_detection(image_path)
