#pragma execution_character_set("utf-8")
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // ===================== 任务1：读取测试图片（绝对路径，100%找到） =====================
    string img_path = "D:/cv-course/lab02/work.png";
    Mat src = imread(img_path, IMREAD_COLOR);

    if (src.empty()) {
        cerr << "错误：无法读取图片 " << img_path << "，请检查路径！" << endl;
        return -1;
    }

    // ===================== 任务2：输出图像基本信息 =====================
    int width = src.cols;
    int height = src.rows;
    int channels = src.channels();
    int depth = src.depth();

    cout << "===== 图像基本信息 =====" << endl;
    cout << "图像宽度: " << width << " 像素" << endl;
    cout << "图像高度: " << height << " 像素" << endl;
    cout << "图像通道数: " << channels << endl;
    cout << "数据类型: ";
    switch (depth) {
        case CV_8U:  cout << "8位无符号整数 (CV_8U)"; break;
        case CV_8S:  cout << "8位有符号整数 (CV_8S)"; break;
        case CV_16U: cout << "16位无符号整数 (CV_16U)"; break;
        case CV_16S: cout << "16位有符号整数 (CV_16S)"; break;
        case CV_32S: cout << "32位有符号整数 (CV_32S)"; break;
        case CV_32F: cout << "32位浮点数 (CV_32F)"; break;
        case CV_64F: cout << "64位浮点数 (CV_64F)"; break;
        default:     cout << "未知类型"; break;
    }
    cout << endl << "=========================" << endl;

    // ===================== 任务3：显示原图 =====================
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", src);

    // ===================== 任务4：转换为灰度图并显示 =====================
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    namedWindow("Grayscale Image", WINDOW_AUTOSIZE);
    imshow("Grayscale Image", gray);

    // ===================== 任务5：保存灰度图 =====================
    string gray_save_path = "D:/cv-course/lab02/test_gray.png";
    bool save_success = imwrite(gray_save_path, gray);
    if (save_success) {
        cout << "灰度图已成功保存为: " << gray_save_path << endl;
    } else {
        cerr << "❌ 灰度图保存失败！" << endl;
    }

    // ===================== 任务6：NumPy等价操作 =====================
    // 1. 输出像素值
    int x = 100, y = 100;
    if (x < width && y < height) {
        Vec3b pixel = src.at<Vec3b>(y, x);
        cout << "\n===== 像素值输出 =====" << endl;
        cout << "坐标 (" << x << "," << y << ") 的BGR像素值: ";
        cout << "B=" << (int)pixel[0] << ", G=" << (int)pixel[1] << ", R=" << (int)pixel[2] << endl;

        uchar gray_pixel = gray.at<uchar>(y, x);
        cout << "对应灰度值: " << (int)gray_pixel << endl;
    }

    // 2. 裁剪左上角200x200区域
    int crop_size = 200;
    if (crop_size <= width && crop_size <= height) {
        Mat crop = src(Rect(0, 0, crop_size, crop_size));
        string crop_save_path = "D:/cv-course/lab02/test_crop.png";
        imwrite(crop_save_path, crop);
        cout << "左上角" << crop_size << "x" << crop_size << "区域已保存为: " << crop_save_path << endl;
        namedWindow("Cropped Image", WINDOW_AUTOSIZE);
        imshow("Cropped Image", crop);
    }

    // 等待按键关闭窗口
    cout << "\n按任意键关闭所有窗口..." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}