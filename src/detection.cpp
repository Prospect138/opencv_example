#include <csetjmp>
#include <limits>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <algorithm>

constexpr float MAX_SCORE = std::numeric_limits<float>::lowest();

struct Detection
{
    cv::Rect    bbox;
    float       score;
    int         class_id;
};

static Detection prev_frame;

std::vector<Detection> parseOutput(cv::Mat out) //[1, 84, 8400]
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    float* raw_data = (float*) out.data;
    std::vector<Detection> parsed_out;

    int num_channels = out.size[1];
    int num_anchors = out.size[2];

    if (num_anchors == 0)
    {
        return parsed_out;
    }
    for (int i = 0; i < num_anchors; ++i)
    {
        float center_x = raw_data[i];
        float center_y = raw_data[i + 1 * num_anchors];
        float width = raw_data[i + 2 * num_anchors];
        float height = raw_data[i + 3 * num_anchors];

        int class_id = -1;
        float max_score = MAX_SCORE;
    
        for (int j = 4; j < num_channels; ++j)
        {
            const float score = raw_data[i + j * num_anchors];
            if (score > max_score)
            {
                max_score = score;
                class_id = j;
            }
        }
        if (class_id == -1 || max_score < 0.7f) continue;
        float left = center_x - width / 2.0f;
        float top = center_y - height / 2.0f;
        boxes.push_back(cv::Rect(left, top, width, height));
        scores.push_back(max_score);
        class_ids.push_back(class_id);
    }
    std::vector<int> indices;
    const float nms_threshold = 0.5f;
    cv::dnn::NMSBoxes(boxes, scores, 0.7f, nms_threshold, indices);

    for (int idx : indices)
    {
        parsed_out.push_back({boxes[idx], scores[idx], class_ids[idx]});
    }
    return parsed_out;
}

void drawRectangles(std::vector<Detection> parsed_data, cv::Mat frame)
{
    for (const auto& detection : parsed_data)
    {
        cv::rectangle(frame, detection.bbox, cv::Scalar(0, 0, 255), 3);
    }
}

void setBackend(cv::dnn::Net& net)
{
    const auto backends = cv::dnn::getAvailableBackends();

    auto it = std::find_if(backends.begin(), backends.end(), [](const std::pair<cv::dnn::Backend, cv::dnn::Target> pair){
        const auto require = std::pair<cv::dnn::Backend, cv::dnn::Target> {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA};
        return pair == require;
    });

    if (it != backends.end()) 
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } 
    else 
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

int main(int argc, char* argv[]) {
    const char* url = argc >=2 ? argv[1] : "http://192.168.0.245:2238/video";

    cv::VideoCapture capture(url);

    if (!capture.isOpened())
    {
        std::cerr << "No capture.\n";
    }

    cv::dnn::Net net = cv::dnn::readNet("../model/yolo11m.onnx");

    setBackend(net);

    while (true)
    {
        cv::Mat frame(1920, 1080, CV_8UC3);
        cv::Mat blob;
        cv::Rect roi(640, 220, 640, 640);
        capture >> frame;
        if (frame.empty()) break;

        cv::Mat display_frame = frame(roi).clone();
        cv::dnn::blobFromImage( display_frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);

        cv::Mat output = net.forward(); // actual inference
        std::vector<Detection> parsed_data = parseOutput(output);
        cv::Mat previous_output(output);
        drawRectangles(parsed_data, display_frame);

        cv::namedWindow("processed", cv::WINDOW_NORMAL);
        cv::resizeWindow("processed", cv::Size(640, 640));
        cv::imshow("processed", display_frame);

        if (cv::waitKey(1) == 27 )  // 27 for esc
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    
    return 0;
}