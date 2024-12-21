import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def pca_image_reconstruction(image_path, num_components):
    """
    对图片进行主成分提取和恢复
    :param image_path: 输入图像路径
    :param num_components: 保留的特征值个数
    """
    # 读取灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 图像的形状
    h, w = img.shape

    # 转换为 2D 矩阵，每行为一个样本
    img_flat = img.reshape(-1, w).astype(np.float32)

    # 计算均值并减去均值进行中心化
    mean = np.mean(img_flat, axis=0)
    img_centered = img_flat - mean

    # 计算协方差矩阵
    covariance_matrix = np.cov(img_centered, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 对特征值从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前 num_components 个特征向量
    selected_eigenvectors = eigenvectors[:, :num_components]

    # 将图像映射到低维空间
    reduced_data = np.dot(img_centered, selected_eigenvectors)

    # 从低维空间恢复图像
    reconstructed = np.dot(reduced_data, selected_eigenvectors.T) + mean

    # 恢复图像到原始形状
    reconstructed_image = reconstructed.reshape(h, w).astype(np.uint8)

    # 绘制对比图
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("原始图片")
    plt.axis('off')

    # 恢复图像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"恢复图片（特征值个数: {num_components}）")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 测试函数
image_path = 'my.jpg'  # 替换为你的图片路径
pca_image_reconstruction(image_path, num_components=20)
