import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载灰度图像
image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)

# 定义运动模糊核（PSF）
def motion_blur_kernel(size=15, angle=45):
    kernel = np.zeros((size, size))
    center = size // 2
    tan_angle = np.tan(np.deg2rad(angle))
    for i in range(size):
        offset = int(center + tan_angle * (i - center))
        if 0 <= offset < size:
            kernel[offset, i] = 1
    kernel /= kernel.sum()  # 归一化
    return kernel

# 对图像添加运动模糊
def add_motion_blur(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 维纳滤波
def wiener_filter(degraded, kernel, K=10):
    kernel_padded = np.pad(kernel, [(0, degraded.shape[0] - kernel.shape[0]),
                                    (0, degraded.shape[1] - kernel.shape[1])], mode='constant')
    kernel_ft = fft2(kernel_padded)
    degraded_ft = fft2(degraded)
    wiener_ft = np.conj(kernel_ft) / (np.abs(kernel_ft) ** 2 + K)
    result = np.abs(ifft2(degraded_ft * wiener_ft))
    return np.clip(result, 0, 255).astype(np.uint8)

# 约束最小二乘方滤波
def constrained_least_squares(degraded, kernel, gamma=0.01):
    kernel_padded = np.pad(kernel, [(0, degraded.shape[0] - kernel.shape[0]),
                                    (0, degraded.shape[1] - kernel.shape[1])], mode='constant')
    kernel_ft = fft2(kernel_padded)
    degraded_ft = fft2(degraded)

    # 拉普拉斯算子
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_padded = np.pad(laplacian, [(0, degraded.shape[0] - 3), (0, degraded.shape[1] - 3)], mode='constant')
    laplacian_ft = fft2(laplacian_padded)

    cls_ft = np.conj(kernel_ft) / (np.abs(kernel_ft) ** 2 + gamma * np.abs(laplacian_ft) ** 2)
    result = np.abs(ifft2(degraded_ft * cls_ft))
    return np.clip(result, 0, 255).astype(np.uint8)

# 应用运动模糊和高斯噪声
motion_kernel = motion_blur_kernel(size=15, angle=45)
motion_blurred = add_motion_blur(image, motion_kernel)
motion_blurred_noisy = add_gaussian_noise(motion_blurred)

# 进行维纳滤波和约束最小二乘方滤波
restored_wiener = wiener_filter(motion_blurred_noisy, motion_kernel, K=10)
restored_cls = constrained_least_squares(motion_blurred_noisy, motion_kernel, gamma=0.01)

# 绘制对比图
# 绘制对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
titles = ['原始图像', '运动模糊 + 高斯噪声', '恢复图像 - 维纳滤波', '恢复图像 - CLS滤波']
images = [image, motion_blurred_noisy, restored_wiener, restored_cls]

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("原始图像")
axes[0, 1].imshow(motion_blurred, cmap='gray')
axes[0, 1].set_title("运动模糊图像")
axes[0, 2].imshow(motion_blurred_noisy, cmap='gray')
axes[0, 2].set_title("运动模糊 + 高斯噪声")

axes[1, 0].imshow(restored_wiener, cmap='gray')
axes[1, 0].set_title("恢复图像 - 维纳滤波")
axes[1, 1].imshow(restored_cls, cmap='gray')
axes[1, 1].set_title("恢复图像 - CLS滤波")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
