#pragma once
#include "../nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>
#include <vector>

// Forward declare your YoloResults struct if needed
// (adjust if you include another header for that):
/*struct YoloResults {
    cv::Rect bbox;
    float conf = 0.f;
    int class_idx = -1;
    cv::Mat mask;
    std::vector<float> keypoints;
};*/

// Declarations of utility vectors
extern std::vector<std::vector<int>> skeleton;
extern std::vector<cv::Scalar> posePalette;
extern std::vector<int> limbColorIndices;
extern std::vector<int> kptColorIndices;

// Declaration of the utility functions
cv::Scalar generateRandomColor(int numChannels);
std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels);

void plot_masks(cv::Mat img,
                std::vector<YoloResults>& result,
                std::vector<cv::Scalar> color,
                std::unordered_map<int, std::string>& names);

void plot_keypoints(cv::Mat& image,
                    const std::vector<YoloResults>& results,
                    const cv::Size& shape);

void plot_results(cv::Mat img,
                  std::vector<YoloResults>& results,
                  std::vector<cv::Scalar> color,
                  std::unordered_map<int, std::string>& names,
                  const cv::Size& shape);
