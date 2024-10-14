####

#RGB上进行均值滤波以及拉普拉斯变换，仅在HSI的强度分量上进行相同的操作，比较两者的结果。

####
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# RGB空间上进行均值滤波和拉普拉斯变换
def process_rgb(img_rgb):
    # 均值滤波
    img_blur = cv2.blur(img_rgb, (5, 5))

    # 拉普拉斯变换
    img_laplacian = cv2.Laplacian(img_rgb, cv2.CV_64F)
    img_laplacian = np.uint8(np.absolute(img_laplacian))  # 取绝对值后转换为uint8

    return img_blur, img_laplacian


# 将RGB转换为HSI
def rgb_to_hsi(img):
    img = img / 255.0
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 计算亮度（Intensity）
    I = (R + G + B) / 3.0

    # 计算饱和度（Saturation）
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_val / (R + G + B + 1e-6)

    # 计算色调（Hue）
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.zeros_like(R)
    H[B > G] = 2 * np.pi - theta[B > G]
    H[B <= G] = theta[B <= G]
    H = H / (2 * np.pi)  # 归一化到[0, 1]

    return H, S, I


# 将HSI转换回RGB
def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi  # 将H值恢复到[0, 2π]范围
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # H在[0, 2π/3]，对应红色区域
    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # H在[2π/3, 4π/3]，对应绿色区域
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H[idx] -= 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # H在[4π/3, 2π]，对应蓝色区域
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H[idx] -= 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    # 合并通道并转换回[0, 255]范围
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb


# 在HSI的强度（I）分量上进行均值滤波和拉普拉斯变换
def process_hsi(img_rgb):
    # 转换为HSI
    H, S, I = rgb_to_hsi(img_rgb)

    # 对亮度（Intensity）通道进行均值滤波
    I_blur = cv2.blur((I * 255).astype(np.uint8), (5, 5)) / 255.0

    # 对亮度（Intensity）通道进行拉普拉斯变换
    I_laplacian = cv2.Laplacian((I * 255).astype(np.uint8), cv2.CV_64F)
    I_laplacian = np.uint8(np.absolute(I_laplacian)) / 255.0

    # 将均值滤波和拉普拉斯变换后的I通道与原始的H和S通道合并
    img_hsi_blur = hsi_to_rgb(H, S, I_blur)
    img_hsi_laplacian = hsi_to_rgb(H, S, I_laplacian)

    return img_hsi_blur, img_hsi_laplacian



# 读取彩色图像
img = cv2.imread('my.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("请选择处理空间：")
print("1: RGB 空间均值滤波和拉普拉斯变换")
print("2: HSI 空间仅对强度分量均值滤波和拉普拉斯变换")
n = input("请输入选择(1 或 2): ")

if n == '1':
    # 在RGB空间进行处理
    img_blur, img_laplacian = process_rgb(img_rgb)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('原始RGB图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_blur)
    plt.title('RGB空间均值滤波')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_laplacian)
    plt.title('RGB空间拉普拉斯变换')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if n == '2':
    # 在HSI空间进行处理
    img_hsi_blur, img_hsi_laplacian = process_hsi(img_rgb)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('原始RGB图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_hsi_blur)
    plt.title('HSI空间均值滤波 (仅强度分量)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_hsi_laplacian)
    plt.title('HSI空间拉普拉斯变换 (仅强度分量)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
