#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/videoio.hpp>

int main() {
    std::string url = "http://192.168.0.245:2238/video";

    cv::VideoCapture capture(url);

    if (!capture.isOpened())
    {
        std::cerr << "No capture.\n";
    }

    cv::Mat frame1(1920, 1080, CV_8UC3);
    while (true)
    {
        int threshhold1{30}, threshhold2{30};

        capture >> frame1;
        if (frame1.empty()) break;

        cv::Mat frame2;
        cv::cvtColor(frame1, frame2, cv::COLOR_BGR2GRAY);
        cv::Canny(frame2, frame1, threshhold1, threshhold1);
        cv::dilate(frame1, frame2, cv::getStructuringElement(cv::MorphShapes::MORPH_DIAMOND, cv::Size(4, 4)));

        cv::namedWindow("Ip webcam", cv::WINDOW_NORMAL);
        cv::resizeWindow("Ip webcam", cv::Size(1600, 900));
        cv::imshow("Ip webcam", frame1);

        if (cv::waitKey(1) == 27 )  // 27 for esc
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    
    return 0;
}