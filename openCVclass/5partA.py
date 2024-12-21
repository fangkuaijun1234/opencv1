'''
以下代码实现 Prewitt 算子在水平和垂直方向上的边缘检测，
并考虑对角线 Prewitt 算子处理。
代码还加入了图像平滑步骤和阈值化操作以增强边缘效果.
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def prewitt_edge_detection(image, smooth=False, diagonal=False, threshold=None):
    """
    使用 Prewitt 梯度算子进行边缘检测
    :param image: 输入灰度图像
    :param smooth: 是否先进行平滑处理
    :param diagonal: 是否采用对角线 Prewitt 算子
    :param threshold: 阈值化处理，若为 None，则不进行
    :return: 边缘检测后的图像
    """
    # 平滑处理（可选）
    if smooth:
        image = cv.GaussianBlur(image, (5, 5), 0)

    # Prewitt 算子
    if diagonal:
        kernel_dx = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])  # 对角线 Prewitt 算子 (45度)
        kernel_dy = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])  # 对角线 Prewitt 算子 (-45度)
    else:
        kernel_dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 水平方向
        kernel_dy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # 垂直方向

    # 计算梯度
    grad_x = cv.filter2D(image, cv.CV_64F, kernel_dx)
    grad_y = cv.filter2D(image, cv.CV_64F, kernel_dy)

    # 梯度幅值
    gradient_magnitude = cv.magnitude(grad_x, grad_y)

    # 阈值化处理
    if threshold is not None:
        _, gradient_magnitude = cv.threshold(gradient_magnitude, threshold, 255, cv.THRESH_BINARY)

    return gradient_magnitude.astype(np.uint8)
def plot_images(original, results, titles):
    """
    显示图像对比
    :param original: 原始图像
    :param results: 结果图像列表
    :param titles: 标题列表
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    for i, (result, title) in enumerate(zip(results, titles)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 主函数
def main(image_path):
    # 读取灰度图像
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # # 直接 Prewitt 边缘检测
    # edges_direct = prewitt_edge_detection(img, smooth=False, diagonal=False)
    # # 平滑后 Prewitt 边缘检测
    # edges_smoothed = prewitt_edge_detection(img, smooth=True, diagonal=False)
    # 对角线 Prewitt 边缘检测
    edges_diagonal = prewitt_edge_detection(img, smooth=False, diagonal=True)
    # # 阈值化增强边缘
    # edges_threshold = prewitt_edge_detection(img, smooth=False, diagonal=False, threshold=50)

    # 图像显示对比
    # results = [edges_direct, edges_smoothed, edges_diagonal, edges_threshold]
    results = [edges_diagonal]
    titles = [ '对角线 Prewitt']
    plot_images(img, results, titles)


# 调用主函数
main('my.jpg')
