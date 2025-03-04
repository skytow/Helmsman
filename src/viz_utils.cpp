#include "../include/utils/viz_utils.hpp"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>

// Define the skeleton and color mappings (now as actual definitions)
std::vector<std::vector<int>> skeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
    {6, 12}, {7, 13},  {6, 7},    {6, 8},   {7, 9},
    {8, 10}, {9, 11},  {2, 3},    {1, 2},   {1, 3},
    {2, 4},  {3, 5},   {4, 6},    {5, 7}
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
    9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
};
std::vector<int> kptColorIndices = {
    16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9
};

cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be 1..3.");
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

void plot_masks(cv::Mat img,
                std::vector<YoloResults>& result,
                std::vector<cv::Scalar> color,
                std::unordered_map<int, std::string>& names)
{
    cv::Mat mask = img.clone();
    for (int i = 0; i < (int)result.size(); i++)
    {
        float left = result[i].bbox.x;
        float top  = result[i].bbox.y;
        int color_num = i;
        int& class_idx = result[i].class_idx;

        // bounding box
        cv::rectangle(img, result[i].bbox, color[class_idx], 2);

        // label
        std::string class_name;
        auto it = names.find(class_idx);
        if (it != names.end()) {
            class_name = it->second;
        } else {
            std::cerr << "Warning: class_idx not found in names for class_idx = "
                      << class_idx << std::endl;
            class_name = std::to_string(class_idx);
        }

        // fill mask if present
        if (!result[i].mask.empty()) {
            mask(result[i].bbox).setTo(color[class_idx], result[i].mask);
        }

        // build label text
        std::stringstream labelStream;
        labelStream << class_name << " "
                    << std::fixed << std::setprecision(2) << result[i].conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5,
                              text_size.width + 2, text_size.height + 5);

        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        cv::rectangle(img, rect_to_fill, color[class_idx], -1);

        cv::putText(img, label, cv::Point(left - 1.5f, top - 2.5f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }
    addWeighted(img, 0.6, mask, 0.4, 0, img);
    // Show
    cv::imshow("img", img);
    cv::waitKey();
}

void plot_keypoints(cv::Mat& image,
                    const std::vector<YoloResults>& results,
                    const cv::Size& shape)
{
    int radius = 5;
    bool drawLines = true;
    if (results.empty()) return;

    // Build color palettes
    std::vector<cv::Scalar> limbColorPalette, kptColorPalette;
    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }
    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& res : results) {
        auto keypoint = res.keypoints;
        bool isPose = (keypoint.size() == 51); // 17*3
        drawLines &= isPose;

        // Draw each point
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if ((x_coord >= 0 && x_coord < shape.width) &&
                (y_coord >= 0 && y_coord < shape.height))
            {
                float conf = keypoint[idx + 2];
                if (conf < 0.5) continue; // skip low conf

                cv::Scalar color_k = isPose ? kptColorPalette[i]
                                            : cv::Scalar(0, 0, 255);
                cv::circle(image, cv::Point(x_coord, y_coord),
                           radius, color_k, -1, cv::LINE_AA);
            }
        }

        // Possibly draw lines between keypoints
        if (drawLines) {
            for (int i = 0; i < (int)skeleton.size(); i++) {
                const std::vector<int> &sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int x1 = static_cast<int>(keypoint[idx1 * 3]);
                int y1 = static_cast<int>(keypoint[idx1 * 3 + 1]);
                int x2 = static_cast<int>(keypoint[idx2 * 3]);
                int y2 = static_cast<int>(keypoint[idx2 * 3 + 1]);

                float conf1 = keypoint[idx1 * 3 + 2];
                float conf2 = keypoint[idx2 * 3 + 2];
                if (conf1 < 0.5f || conf2 < 0.5f) continue;

                if (x1<0||y1<0||x2<0||y2<0 ||
                    x1>=shape.width||x2>=shape.width ||
                    y1>=shape.height||y2>=shape.height) {
                    continue;
                }
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2),
                         color_limb, 2, cv::LINE_AA);
            }
        }
    }
}

void plot_results(cv::Mat img,
                  std::vector<YoloResults>& results,
                  std::vector<cv::Scalar> color,
                  std::unordered_map<int, std::string>& names,
                  const cv::Size& shape)
{
    cv::Mat mask = img.clone();
    bool drawLines = true;
    auto raw_image_shape = img.size();
    int radius = 5;

    // Build color palettes
    std::vector<cv::Scalar> limbColorPalette, kptColorPalette;
    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }
    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (auto & res : results) {
        float left = res.bbox.x;
        float top  = res.bbox.y;

        // bounding box
        cv::rectangle(img, res.bbox, color[res.class_idx], 2);

        // label text
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end()) {
            class_name = it->second;
        } else {
            class_name = std::to_string(res.class_idx);
        }

        // optional mask
        if (!res.mask.empty()) {
            mask(res.bbox).setTo(color[res.class_idx], res.mask);
        }

        // label
        std::stringstream labelStream;
        labelStream << class_name << " "
                    << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5,
                              text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255, 255, 255);
        cv::rectangle(img, rect_to_fill, color[res.class_idx], -1);
        cv::putText(img, label, cv::Point(left - 1.5f, top - 2.5f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

        // if keypoints exist, draw them
        auto &keypoint = res.keypoints;
        if (!keypoint.empty()) {
            bool isPose = (keypoint.size() == 51);
            drawLines &= isPose;

            // draw points
            for (int i = 0; i < 17; i++) {
                int idx = i * 3;
                int x_coord = static_cast<int>(keypoint[idx]);
                int y_coord = static_cast<int>(keypoint[idx + 1]);
                float conf = keypoint[idx + 2];

                if (conf < 0.5f) continue;
                if (x_coord<0 || y_coord<0 ||
                    x_coord>=raw_image_shape.width ||
                    y_coord>=raw_image_shape.height) {
                    continue;
                }

                cv::Scalar color_k = isPose ? kptColorPalette[i]
                                            : cv::Scalar(0,0,255);
                cv::circle(img, cv::Point(x_coord, y_coord),
                           radius, color_k, -1, cv::LINE_AA);
            }

            // possibly draw lines
            if (drawLines) {
                for (int i = 0; i < (int)skeleton.size(); i++) {
                    int idx1 = skeleton[i][0] - 1;
                    int idx2 = skeleton[i][1] - 1;

                    int x1 = static_cast<int>(keypoint[idx1 * 3]);
                    int y1 = static_cast<int>(keypoint[idx1 * 3 + 1]);
                    float conf1 = keypoint[idx1 * 3 + 2];

                    int x2 = static_cast<int>(keypoint[idx2 * 3]);
                    int y2 = static_cast<int>(keypoint[idx2 * 3 + 1]);
                    float conf2 = keypoint[idx2 * 3 + 2];

                    if (conf1<0.5f || conf2<0.5f) continue;
                    if (x1<0 || y1<0 || x2<0 || y2<0 ||
                        x1>=raw_image_shape.width ||
                        x2>=raw_image_shape.width ||
                        y1>=raw_image_shape.height||
                        y2>=raw_image_shape.height) {
                        continue;
                    }
                    cv::Scalar color_limb = limbColorPalette[i];
                    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2),
                             color_limb, 2, cv::LINE_AA);
                }
            }
        }
    }

    addWeighted(img, 0.6, mask, 0.4, 0, img);
    // optionally display / or let caller show
    // e.g. cv::imshow("img", img); cv::waitKey();
}
