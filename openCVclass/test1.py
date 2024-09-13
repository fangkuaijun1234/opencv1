import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

#把灰度图进行切片
def hui_du_ji_qie_pian(img):

    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            #灰度小于230大于190就设置为255
            if img[i, j] <= 230 and img[i, j] >= 190:
                new_img[i, j] = 255
            else:
                new_img[i,j] = img[i,j]

    return new_img
def hua_T_r():
    #灰度0到255
    r = np.arange(0, 256)
    #按照
    T = np.where((r >= 190) & (r <= 230), 255, r)

    # 绘制转换函数图像
    plt.plot(r, T, color='blue')
    plt.title('T(r) - 灰度级切片转换函数')
    plt.xlabel('输入灰度值 r')
    plt.ylabel('输出灰度值 T(r)')
    plt.grid(True)
    plt.show()

#位平面且片
def wei_ping_mian_qie_pian():
    # 读取图像，并转换为灰度图像(防止封装功能后不是灰度图)
    img = cv2.imread('mys.jpg', cv2.IMREAD_GRAYSCALE)
    # # 获取图像的高度和宽度
    # h, w = img.shape
    # 创建一个列表来存储8个位平面的图像
    bit_planes = []

    # 遍历每一位平面，进行位与操作
    for i in range(8):
        # 使用位移操作提取每个位平面
        bit_plane = (img >> i) & 1
        bit_planes.append(bit_plane * 255)  # 将二值图像放大到 0-255 范围

    # 创建一个图像网格展示所有位平面
    plt.figure(figsize=(10, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(bit_planes[i], cmap='gray')
        plt.title(f'Bit Plane {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 函数：计算图像的直方图，并返回直方图图像
def calc_histogram_image(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 创建绘图区域
    fig, ax = plt.subplots()
    ax.plot(hist, color='black')
    ax.set_xlim([0, 256])
    ax.set_title('Histogram')

    # 将绘制的直方图保存为图像
    fig.canvas.draw()

    # 从figure中获取图像数据
    histogram_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    histogram_img = histogram_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 关闭figure，释放内存
    plt.close(fig)

    return histogram_img


# 创建暗图像
def create_dark_image(img):
    dark_img = np.clip(img * 0.5, 0, 255).astype(np.uint8)
    return dark_img


# 创建亮图像
def create_bright_image(img):
    bright_img = np.clip(img * 1.5, 0, 255).astype(np.uint8)
    return bright_img


# 创建低对比度图像
def create_low_contrast_image(img):
    low_contrast_img = np.clip((img - 128) * 0.5 + 128, 0, 255).astype(np.uint8)
    return low_contrast_img


# 创建高对比度图像
def create_high_contrast_image(img):
    high_contrast_img = np.clip((img - 128) * 2 + 128, 0, 255).astype(np.uint8)
    return high_contrast_img


# 主要处理函数
def process_image_and_plot_histograms(image_path):
    # 读取图像，并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 创建四类图像
    dark_img = create_dark_image(img)
    bright_img = create_bright_image(img)
    low_contrast_img = create_low_contrast_image(img)
    high_contrast_img = create_high_contrast_image(img)

    # 计算直方图图像
    original_hist_img = calc_histogram_image(img)
    dark_hist_img = calc_histogram_image(dark_img)
    bright_hist_img = calc_histogram_image(bright_img)
    low_contrast_hist_img = calc_histogram_image(low_contrast_img)
    high_contrast_hist_img = calc_histogram_image(high_contrast_img)

    # 显示结果图像
    plt.figure(figsize=(10, 8))

    # 显示原始图像的直方图图像
    plt.subplot(5, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(5, 2, 2)
    plt.imshow(original_hist_img)
    plt.title('Original Histogram')
    plt.axis('off')

    # 显示暗图像的直方图图像
    plt.subplot(5, 2, 3)
    plt.imshow(dark_img, cmap='gray')
    plt.title('Dark Image')
    plt.axis('off')

    plt.subplot(5, 2, 4)
    plt.imshow(dark_hist_img)
    plt.title('Dark Histogram')
    plt.axis('off')

    # 显示亮图像的直方图图像
    plt.subplot(5, 2, 5)
    plt.imshow(bright_img, cmap='gray')
    plt.title('Bright Image')
    plt.axis('off')

    plt.subplot(5, 2, 6)
    plt.imshow(bright_hist_img)
    plt.title('Bright Histogram')
    plt.axis('off')

    # 显示低对比度图像的直方图图像
    plt.subplot(5, 2, 7)
    plt.imshow(low_contrast_img, cmap='gray')
    plt.title('Low Contrast Image')
    plt.axis('off')

    plt.subplot(5, 2, 8)
    plt.imshow(low_contrast_hist_img)
    plt.title('Low Contrast Histogram')
    plt.axis('off')

    # 显示高对比度图像的直方图图像
    plt.subplot(5, 2, 9)
    plt.imshow(high_contrast_img, cmap='gray')
    plt.title('High Contrast Image')
    plt.axis('off')

    plt.subplot(5, 2, 10)
    plt.imshow(high_contrast_hist_img)
    plt.title('High Contrast Histogram')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


######################

print("输入'1':图像灰度切片,把my.jpg转为灰度图mys.jpg，然后切片")
print("输入'2':图像mys.jpg灰度图位平面切片")
print("输入'3':图像mys.jpg灰度图直方图统计")
print("输入'4':图像mys.jpg灰度图直方图统计,有均衡化过程")
num = input("请选择功能：")

############这是灰度切片。选择了转化函数是一个范围全为白255########
if num == '1':
    image = cv2.imread('my.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('mys.jpg', image)
    img = cv2.imread('mys.jpg',0)
    fig = hui_du_ji_qie_pian(img)
    cv2.imwrite('hui_du_qie_pian.jpg',fig)
    hua_T_r()

###############灰度图的8个bit的位平面
if num == '2':
    wei_ping_mian_qie_pian()

###############灰度图的直方图统计，暗图像，亮图像，低对比和高
if num == '3':
    image = 'mys.jpg'
    process_image_and_plot_histograms(image)