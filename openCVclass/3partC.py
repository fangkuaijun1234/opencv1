# 使用拉普拉斯算子进行空间域锐化和频率域

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def laplace_filter(image):
    # 使用拉普拉斯算子进行空间域锐化
    laplacian = cv.Laplacian(image, cv.CV_16S, ksize=3)
    laplacian_abs = cv.convertScaleAbs(laplacian)  # 转换为 uint8 格式
    return laplacian_abs

def laplace_sharpening(image_path):
    # 读取图像
    img = cv.imread(image_path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换为 RGB 格式以便 Matplotlib 显示

    # 空间域拉普拉斯锐化
    spatial_laplacian = laplace_filter(gray_image)

    # 频域拉普拉斯增强
    img_float = np.float32(gray_image)
    dft = cv.dft(img_float, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # 生成拉普拉斯滤波器
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    laplacian_kernel = -4 * np.ones((rows, cols))
    laplacian_kernel[crow-1:crow+2, ccol-1:ccol+2] = 1

    # 应用拉普拉斯滤波器
    filtered_dft = dft_shifted * laplacian_kernel[:, :, np.newaxis]

    # 进行逆变换
    img_reconstructed = cv.idft(np.fft.ifftshift(filtered_dft))
    img_reconstructed = cv.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])

    # 显示结果
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(spatial_laplacian, cmap='gray')
    plt.title('空间域拉普拉斯锐化')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title('频域拉普拉斯增强')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数，输入图像路径
laplace_sharpening('my.jpg')
