#include "../include/constants.h"
#include "../include/utils/common.h"
#include "../include/nn/onnx_model_base.h"
#include "../include/nn/autobackend.h"
#include "../include/utils/augment.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

namespace fs = std::filesystem;

// Define the skeleton and color mappings (unchanged)
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

std::vector<int> limbColorIndices = {
    9, 9, 9, 9, 7, 7, 7, 0, 0, 0,
    0, 0, 16, 16, 16, 16, 16, 16, 16
};
std::vector<int> kptColorIndices = {
    16, 16, 16, 16, 16, 0, 0, 0,
    0, 0, 0, 9, 9, 9, 9, 9, 9
};

cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3)
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");

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
    for (int i = 0; i < class_names_num; i++) {
        colors.push_back(generateRandomColor(numChannels));
    }
    return colors;
}

/**
 * @brief Overlays detection/mask/labels on the input image.
 *        Removed the blocking waitKey() call so that the video loop won’t freeze.
 */
void plot_results(cv::Mat img,
                  std::vector<YoloResults>& results,
                  std::vector<cv::Scalar> color,
                  const std::unordered_map<int, std::string>& names,
                  const cv::Size& shape)
{
    cv::Mat mask = img.clone();
    for (int i = 0; i < (int)results.size(); i++) {
        float left = results[i].bbox.x;
        float top  = results[i].bbox.y;

        // Draw bounding box
        rectangle(img, results[i].bbox, color[results[i].class_idx], 2);

        // Get class name
        std::string class_name;
        auto it = names.find(results[i].class_idx);
        if (it != names.end()) {
            class_name = it->second;
        } else {
            std::cerr << "Warning: class_idx not found in names for class_idx = "
                      << results[i].class_idx << std::endl;
            class_name = std::to_string(results[i].class_idx);
        }

        // If mask is present, apply it
        if (results[i].mask.rows && results[i].mask.cols > 0) {
            mask(results[i].bbox).setTo(color[results[i].class_idx], results[i].mask);
        }

        // Build label
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2)
                    << results[i].conf;
        std::string label = labelStream.str();

        // Draw label background
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(static_cast<int>(left - 1),
                              static_cast<int>(top - text_size.height - 5),
                              text_size.width + 2,
                              text_size.height + 5);
        rectangle(img, rect_to_fill, color[results[i].class_idx], -1);

        // Put text
        putText(img, label,
                cv::Point(static_cast<int>(left - 1.5), static_cast<int>(top - 2.5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
    }

    // Combine mask & image
    addWeighted(img, 0.6, mask, 0.4, 0, img);

    // Removed blocking waitKey(),
    // so this function no longer blocks the video loop.
    // If you want to see it here for debugging, you could do:
    // cv::imshow("Debug Plot", img);
    // cv::waitKey(1); // or a small delay
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: Helmsman <image_or_video_path>\n";
        return 1;
    }

    std::string inputPath = argv[1];
    fs::path filePath(inputPath);
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Model settings (adjust paths as needed)
    const std::string modelPath   = "../checkpoints/best.onnx";
    // Use CPUExecutionProvider (this must match what your onnx_model_base expects)
    const std::string onnx_provider = OnnxProviders::CPUExecutionProvider;
    const std::string onnx_logid  = "yolov8_inference";
    float mask_threshold = 0.5f;
    float conf_threshold = 0.3f;
    float iou_threshold  = 0.45f;
    int conversion_code  = cv::COLOR_BGR2RGB;

    // Initialize model
    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());

    // Generate random colors for bounding boxes/masks
    std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());

    if (ext == ".avi" || ext == ".mp4" || ext == ".mov" || ext == ".mkv") {
        // --- Video ---
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video: " << inputPath << "\n";
            return 1;
        }
        std::cout << "Processing video: " << inputPath << std::endl;

        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "End of video\n";
                break;
            }
            // Convert BGR->RGB for inference
            cv::Mat frame_rgb;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);

            // Inference
            std::vector<YoloResults> results = model.predict_once(
                frame_rgb, conf_threshold, iou_threshold, mask_threshold, conversion_code);

            // Draw results
            plot_results(frame, results, colors, model.getNames(), frame.size());

            // Now display the processed frame (BGR)
            cv::imshow("Video Inference", frame);

            // Small waitKey so it doesn't block. Press ESC to break
            if (cv::waitKey(1) == 27) { // ESC
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();

    } else {
        // --- Single Image ---
        cv::Mat img = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Unable to load image: " << inputPath << "\n";
            return 1;
        }
        std::cout << "Processing image: " << inputPath << std::endl;

        // Inference
        std::vector<YoloResults> results = model.predict_once(
            img, conf_threshold, iou_threshold, mask_threshold, conversion_code);

        // Draw results
        plot_results(img, results, colors, model.getNames(), img.size());

        // Show final result
        cv::imshow("Image Inference", img);
        cv::waitKey(0);
    }

    return 0;
}
