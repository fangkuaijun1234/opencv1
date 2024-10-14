####

#给出一张彩色图片的RGB以及HSI分量图

####
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# RGB分量图
def display_rgb_components(img_rgb):
    # 分离RGB通道
    R, G, B = cv2.split(img_rgb)
    # 创建用于显示RGB分量的图像，每个分量只保留该通道的值，其他通道置为0
    # R通道
    R_img = np.zeros_like(img_rgb)
    R_img[:, :, 0] = R  # 红色通道
    # G通道
    G_img = np.zeros_like(img_rgb)
    G_img[:, :, 1] = G  # 绿色通道
    # B通道
    B_img = np.zeros_like(img_rgb)
    B_img[:, :, 2] = B  # 蓝色通道
    # 显示RGB分量
    titles = ['原始彩色图像', '红色通道', '绿色通道', '蓝色通道']
    images = [img_rgb, R_img, G_img, B_img]
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# 添加HSI分量图
def rgb_to_hsi(img):
    # 将RGB图像归一化到[0, 1]范围
    img = img / 255.0
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 计算Intensity分量
    I = (R + G + B) / 3.0

    # 计算Saturation分量
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_val / (R + G + B + 1e-6)

    # 计算Hue分量
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.zeros_like(R)
    H[B > G] = 2 * np.pi - theta[B > G]
    H[B <= G] = theta[B <= G]
    H = H / (2 * np.pi)  # 归一化到[0, 1]

    return H, S, I

def display_hsi_components(img_rgb):
    # 将RGB图像转换为HSI分量
    H, S, I = rgb_to_hsi(img_rgb)

    # 显示HSI分量
    titles = ['原始彩色图像', 'Hue (色调)', 'Saturation (饱和度)', 'Intensity (亮度)']
    images = [img_rgb, H, S, I]

    plt.figure(figsize=(10, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

    # 读取彩色图像
img = cv2.imread('my.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("请选择要显示的图像分量：")
print("1: RGB 分量图")
print("2: HSI 分量图")
n = input("请输入选择(1 或 2): ")

if n == '1':
    display_rgb_components(img_rgb)
if n == '2':
    display_hsi_components(img_rgb)


