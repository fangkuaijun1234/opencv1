import cv2 as cv
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def canny_edge_detection(image_path, low_threshold, high_threshold):
    """
    使用 Canny 算子进行边缘检测
    :param image_path: 图像路径
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return: 边缘检测结果图像
    """
    # 读取灰度图像
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # 使用 Canny 算子进行边缘检测
    edges = cv.Canny(img, low_threshold, high_threshold)

    # 显示原始图像与边缘图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny 边缘检测')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数
canny_edge_detection('my.jpg', low_threshold=50, high_threshold=150)
