
#灰度图上加上高斯噪声、均匀噪声和椒盐噪声
#分别给出原图和对应的污染后的图片，并给出对应的四个直方图

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def add_gaussian_noise(image, mean=0, var=0.01):
    """添加高斯噪声"""
    noise = np.random.normal(mean, var**0.5, image.shape)  # 生成高斯噪声
    noisy_image = np.clip(image + noise, 0, 255)  # 添加噪声并限制范围
    return noisy_image.astype(np.uint8)

def add_uniform_noise(image, low=0, high=0.1):
    """添加均匀噪声"""
    noise = np.random.uniform(low, high, image.shape)  # 生成均匀噪声
    noisy_image = np.clip(image + noise * 255, 0, 255)  # 添加噪声并限制范围
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """添加椒盐噪声"""
    noisy_image = np.copy(image)
    total_pixels = image.size
    salt = np.ceil(salt_prob * total_pixels).astype(int)
    pepper = np.ceil(pepper_prob * total_pixels).astype(int)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # 添加椒噪声
    coords = [np.random.randint(0, i - 1, pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def plot_histograms(original, noisy, noise_type):
    """绘制原图和噪声图像的直方图"""
    plt.figure(figsize=(12, 6))

    # 显示原图
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    # 显示噪声图
    plt.subplot(2, 2, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'{noise_type}噪声图')
    plt.axis('off')

    # 原图直方图
    plt.subplot(2, 2, 3)
    plt.hist(original.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('原图直方图')
    plt.xlim([0, 256])

    # 噪声图直方图
    plt.subplot(2, 2, 4)
    plt.hist(noisy.ravel(), bins=256, range=[0, 256], color='black')
    plt.title(f'{noise_type}噪声图直方图')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


# 读取灰度图像
img = cv.imread('my.jpg', cv.IMREAD_GRAYSCALE)

    # 添加高斯噪声
noisy_gaussian = add_gaussian_noise(img)
plot_histograms(img, noisy_gaussian, '高斯')

    # 添加均匀噪声
noisy_uniform = add_uniform_noise(img)
plot_histograms(img, noisy_uniform, '均匀')

    # 添加椒盐噪声
noisy_sp = add_salt_and_pepper_noise(img)
plot_histograms(img, noisy_sp, '椒盐')
