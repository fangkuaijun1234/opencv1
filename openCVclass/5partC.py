import cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取图像并转换为灰度图
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)

# 计算图像的直方图
hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))

# 归一化直方图，作为每个灰度级的概率
hist_norm = hist / hist.sum()

# 计算灰度级的累积概率和累积均值
cum_prob = np.cumsum(hist_norm)
cum_mean = np.cumsum(hist_norm * np.arange(256))

# 总均值
total_mean = cum_mean[-1]

# 初始化类间方差
max_between_class_variance = 0
optimal_threshold = 0

# 遍历每一个可能的阈值
for threshold in range(256):
    # 前景的累积概率与均值
    w0 = cum_prob[threshold]
    if w0 == 0:
        continue  # 避免分母为0
    m0 = cum_mean[threshold] / w0

    # 背景的累积概率与均值
    w1 = 1 - w0
    if w1 == 0:
        continue  # 避免分母为0
    m1 = (total_mean - cum_mean[threshold]) / w1

    # 类间方差
    between_class_variance = w0 * w1 * (m0 - m1) ** 2

    # 更新最大类间方差和最佳阈值
    if between_class_variance > max_between_class_variance:
        max_between_class_variance = between_class_variance
        optimal_threshold = threshold

# 使用最佳阈值进行图像分割
_, binary_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

# 输出计算的最佳阈值
print(f"手动计算的Otsu阈值: {optimal_threshold}")

# 显示原图和分割结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('原始图像')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('二值化后的图像')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.show()
