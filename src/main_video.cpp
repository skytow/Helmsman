#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include "../include/utils/viz_utils.hpp"

int main()
{
    std::string video_path = "../assets/MVI_1551_NIR.avi";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Can't open video: " << video_path << "\n";
        return 1;
    }

    const std::string modelPath = "../checkpoints/best.onnx";
    const std::string onnx_provider = OnnxProviders::CPU;
    const std::string onnx_logid   = "yolov8_inference_video";

    float mask_threshold = 0.5f;
    float conf_threshold = 0.3f;
    float iou_threshold  = 0.45f;
    int conversion_code  = cv::COLOR_BGR2RGB;

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
    std::unordered_map<int, std::string> names = model.getNames();

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video\n";
            break;
        }
        // Convert BGR->RGB for model
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        // Inference
        std::vector<YoloResults> objs = model.predict_once(
            rgb, conf_threshold, iou_threshold, mask_threshold, conversion_code
        );

        // Switch back to BGR in place (or skip if you used a separate Mat)
        cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);

        // Plot
        plot_results(rgb, objs, colors, names, rgb.size());

        // Show
        cv::imshow("video", rgb);
        char c = (char)cv::waitKey(1);
        if (c == 27) { // ESC
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
