'''
灰度图上加上高斯噪声、均匀噪声和椒盐噪声，分别给出原图和对应的污染后的图片，并给出对应的四个直方图
在这个基础上选择合适的滤波器来对这三张噪声图片进行清除噪声，并给出前后对比图
'''

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

def denoise_image(noisy_image, noise_type):
    """对不同噪声类型选择合适的滤波器进行去噪"""
    if noise_type == '高斯':
        # 对于高斯噪声，使用均值滤波
        return cv.GaussianBlur(noisy_image, (5, 5), 0)
    elif noise_type == '均匀':
        # 对于均匀噪声，使用中值滤波
        return cv.medianBlur(noisy_image, 5)
    elif noise_type == '椒盐':
        # 对于椒盐噪声，使用中值滤波
        return cv.medianBlur(noisy_image, 5)
    else:
        return noisy_image

def plot_comparison(original, noisy, denoised, noise_type):
    """绘制原图、噪声图和去噪后图的对比"""
    plt.figure(figsize=(15, 5))

    # 显示原图
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    # 显示噪声图
    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'{noise_type}噪声图')
    plt.axis('off')

    # 显示去噪后图
    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title(f'去噪后图 ({noise_type})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path):
    # 读取灰度图像
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # 添加高斯噪声并去噪
    noisy_gaussian = add_gaussian_noise(img)
    denoised_gaussian = denoise_image(noisy_gaussian, '高斯')
    plot_comparison(img, noisy_gaussian, denoised_gaussian, '高斯')

    # 添加均匀噪声并去噪
    noisy_uniform = add_uniform_noise(img)
    denoised_uniform = denoise_image(noisy_uniform, '均匀')
    plot_comparison(img, noisy_uniform, denoised_uniform, '均匀')

    # 添加椒盐噪声并去噪
    noisy_sp = add_salt_and_pepper_noise(img)
    denoised_sp = denoise_image(noisy_sp, '椒盐')
    plot_comparison(img, noisy_sp, denoised_sp, '椒盐')

# 调用主函数，输入图像路径
main('my.jpg')

