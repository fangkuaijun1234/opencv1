import cv2 as cv
import numpy as np

# 标准霍夫线变换
def line_detection(image, output_filename=None):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换为灰度图
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 使用Canny算子进行边缘检测
    cv.imshow("Edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)   # 使用霍夫变换检测直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   # theta是弧度
        b = np.sin(theta)
        x0 = a * rho    # 代表 x = r * cos（theta）
        y0 = b * rho    # 代表 y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制红色直线
    cv.imshow("HoughLines", image)

    # 如果需要保存图像，保存到指定路径
    if output_filename:
        cv.imwrite(output_filename, image)
        print(f"图像已保存为 {output_filename}")

# 统计概率霍夫线变换
def line_detect_possible_demo(image, output_filename=None):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换为灰度图
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 使用Canny算子进行边缘检测
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制红色直线
    cv.imshow("HoughLinesP", image)

    # 如果需要保存图像，保存到指定路径
    if output_filename:
        cv.imwrite(output_filename, image)
        print(f"图像已保存为 {output_filename}")

# 读取图像
src = cv.imread('mys.jpg')
if src is None:
    print("图片读取失败")
    exit()

# 显示输入图像
cv.imshow('Input Image', src)

# 调用标准霍夫线变换函数，并选择是否保存输出图像
save_path = "output_hough_lines.jpg"  # 设置保存路径
line_detection(src.copy(), output_filename=save_path)

# 重新读取图像，因为函数会修改原图
src = cv.imread('mys.jpg')

# 调用统计概率霍夫线变换函数，并选择是否保存输出图像
save_path_p = "output_hough_lines_p.jpg"  # 设置保存路径
line_detect_possible_demo(src.copy(), output_filename=save_path_p)

cv.waitKey(0)
cv.destroyAllWindows()
