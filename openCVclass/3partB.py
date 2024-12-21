#

###对my.jpg进行理想、巴特沃思以及高斯低通滤波处理，给出对比结果并分析

#
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#理想低通滤波处理转换函数
def ideal_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    mask = (x - crow)**2 + (y - ccol)**2 <= cutoff**2
    return mask.astype(np.float32)

#巴特沃思低滤波处理转换函数
def butterworth_lowpass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    d = np.sqrt((x - crow)**2 + (y - ccol)**2)
    h = 1 / (1 + (d / cutoff)**(2 * order))
    return h

#高斯低通滤波处理转换函数
def gaussian_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    d = np.sqrt((x - crow)**2 + (y - ccol)**2)
    h = np.exp(-(d**2) / (2 * (cutoff**2)))
    return h

def apply_filter(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = np.float32(img) / 255.0  # 归一化图像

    # 进行离散傅里叶变换
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    cutoff = 230
    order = 2
    # 生成滤波器
    ideal_filter = ideal_lowpass_filter(img.shape, cutoff)
    butterworth_filter = butterworth_lowpass_filter(img.shape, cutoff, order)
    gaussian_filter = gaussian_lowpass_filter(img.shape, cutoff)

    # 应用滤波器
    ideal_filtered = dft_shifted * ideal_filter[:, :, np.newaxis]
    butterworth_filtered = dft_shifted * butterworth_filter[:, :, np.newaxis]
    gaussian_filtered = dft_shifted * gaussian_filter[:, :, np.newaxis]

    # 进行逆变换
    img_ideal = cv2.idft(np.fft.ifftshift(ideal_filtered))
    img_butterworth = cv2.idft(np.fft.ifftshift(butterworth_filtered))
    img_gaussian = cv2.idft(np.fft.ifftshift(gaussian_filtered))

    # 取幅值
    img_ideal = cv2.magnitude(img_ideal[:, :, 0], img_ideal[:, :, 1])
    img_butterworth = cv2.magnitude(img_butterworth[:, :, 0], img_butterworth[:, :, 1])
    img_gaussian = cv2.magnitude(img_gaussian[:, :, 0], img_gaussian[:, :, 1])

    # 归一化到0-255范围
    img_ideal = cv2.normalize(img_ideal, None, 0, 255, cv2.NORM_MINMAX)
    img_butterworth = cv2.normalize(img_butterworth, None, 0, 255, cv2.NORM_MINMAX)
    img_gaussian = cv2.normalize(img_gaussian, None, 0, 255, cv2.NORM_MINMAX)

    # 显示结果
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_ideal, cmap='gray')
    plt.title('理想低通滤波')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_butterworth, cmap='gray')
    plt.title(f'巴特沃斯低通滤波 (阶数={order})')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_gaussian, cmap='gray')
    plt.title('高斯低通滤波')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数，输入图像路径
apply_filter('my.jpg')
