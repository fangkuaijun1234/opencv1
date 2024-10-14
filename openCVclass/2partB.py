####

#分别在RGB和HSI空间上进行直方图均衡化

####
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# RGB空间直方图均衡化
def equalize_hist_rgb(img_rgb):
    # 分离RGB通道
    R, G, B = cv2.split(img_rgb)

    # 对每个通道分别进行直方图均衡化
    R_eq = cv2.equalizeHist(R)
    G_eq = cv2.equalizeHist(G)
    B_eq = cv2.equalizeHist(B)

    # 合并均衡化后的通道
    img_rgb_eq = cv2.merge((R_eq, G_eq, B_eq))

    return img_rgb_eq


# HSI空间直方图均衡化
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

#转换过程，代码参考csdn
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


def equalize_hist_hsi(img_rgb):
    # 转换为HSI
    H, S, I = rgb_to_hsi(img_rgb)

    # 对亮度（Intensity）通道进行直方图均衡化
    I_eq = cv2.equalizeHist((I * 255).astype(np.uint8)) / 255.0

    # 将均衡化后的HSI转换回RGB
    img_rgb_eq = hsi_to_rgb(H, S, I_eq)

    return img_rgb_eq


#分别在RGB和HSI空间中进行直方图均衡化并显示结果
img = cv2.imread('my.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("请选择要进行直方图均衡化的空间：")
print("1: RGB 空间直方图均衡化")
print("2: HSI 空间直方图均衡化")
n = input("请输入选择(1 或 2): ")
if n == '1':
    img_rgb_eq = equalize_hist_rgb(img_rgb)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('原始RGB图像')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_eq)
    plt.title('RGB空间直方图均衡化')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

if n == '2':
    img_rgb_eq = equalize_hist_hsi(img_rgb)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('原始RGB图像')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_eq)
    plt.title('HSI空间直方图均衡化')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

