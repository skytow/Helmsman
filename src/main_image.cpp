#include "../include/utils/viz_utils.hpp"  // Contains inline visualization utilities
#include "../include/nn/onnx_model_base.h"
#include "../include/nn/autobackend.h"
#include "../include/constants.h"
#include "../include/utils/common.h"

#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

//---------------------------
// Global definitions
//---------------------------
std::vector<std::vector<int>> skeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
    {6, 12},  {7, 13},  {6, 7},   {6, 8},   {7, 9},
    {8, 10},  {9, 11},  {2, 3},   {1, 2},   {1, 3},
    {2, 4},   {3, 5},   {4, 6},   {5, 7}
};

std::vector<cv::Scalar> posePalette = {
    cv::Scalar(255, 128, 0),   cv::Scalar(255, 153, 51),
    cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0),
    cv::Scalar(255, 153, 255), cv::Scalar(153, 204, 255),
    cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102),
    cv::Scalar(255, 51, 51),   cv::Scalar(153, 255, 153),
    cv::Scalar(102, 255, 102), cv::Scalar(51, 255, 51),
    cv::Scalar(0, 255, 0),     cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),     cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
std::vector<int> kptColorIndices  = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};

//---------------------------
// Utility functions
//---------------------------
cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    cv::Scalar color;
    for (int i = 0; i < numChannels; i++) {
        color[i] = dis(gen);
    }
    return color;
}

std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels) {
    std::vector<cv::Scalar> colors;
    colors.reserve(class_names_num);
    for (int i = 0; i < class_names_num; i++) {
        colors.push_back(generateRandomColor(numChannels));
    }
    return colors;
}

// (plot_masks, plot_keypoints, plot_results are defined in viz_utils.hpp as inline functions)
// If you need to update them, do so in the header so that they are shared across translation units.

//---------------------------
// Main function with image/video support
//---------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: Helmsman <image_or_video_path>" << std::endl;
        return 1;
    }

    // Get input file path and extension
    std::string inputPath = argv[1];
    fs::path filePath(inputPath);
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Model settings
    const std::string modelPath = "../checkpoints/best.onnx";  // Adjust as needed
    const std::string onnx_provider = OnnxProviders::CPU;
    const std::string onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;
    float conf_threshold = 0.30f;
    float iou_threshold  = 0.45f;
    int conversion_code = cv::COLOR_BGR2RGB;

    // Initialize the model once
    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());

    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        // Process video input
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video " << inputPath << std::endl;
            return 1;
        }
        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            // Run model inference on the current frame
            std::vector<YoloResults> objs = model.predict_once(frame, conf_threshold, iou_threshold, mask_threshold, conversion_code);
            std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
            std::unordered_map<int, std::string> names = model.getNames();

            // Convert frame to BGR if needed
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            plot_results(frame, objs, colors, names, frame.size());

            cv::imshow("Video Inference", frame);
            if (cv::waitKey(1) == 27) { // Exit on ESC key
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }
    else {
        // Process image input
        cv::Mat img = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Unable to load image " << inputPath << std::endl;
            return 1;
        }

        std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
        std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
        std::unordered_map<int, std::string> names = model.getNames();

        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        plot_results(img, objs, colors, names, img.size());
        cv::imshow("Image Inference", img);
        cv::waitKey(0);
    }

    return 0;
}
