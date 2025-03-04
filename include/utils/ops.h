#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <tuple>

// Clip boxes: integer version.
inline void clip_boxes(cv::Rect &box, const cv::Size &shape) {
    box.x = std::max(0, std::min(box.x, shape.width));
    box.y = std::max(0, std::min(box.y, shape.height));
    box.width = std::max(0, std::min(box.width, shape.width - box.x));
    box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

// Clip boxes: floating-point version.
inline void clip_boxes(cv::Rect_<float> &box, const cv::Size &shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}

// Scale boxes from one image shape to another.
inline cv::Rect_<float> scale_boxes(const cv::Size &img1_shape, cv::Rect_<float> &box, const cv::Size &img0_shape,
                                     std::pair<float, cv::Point2f> ratio_pad = {-1.0f, cv::Point2f(-1.0f, -1.0f)},
                                     bool padding = true) {
    float gain, pad_x, pad_y;
    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(img1_shape.height) / img0_shape.height,
                        static_cast<float>(img1_shape.width) / img0_shape.width);
        pad_x = std::round((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
        pad_y = std::round((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
    } else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }
    cv::Rect_<float> scaledCoords = box;
    if (padding) {
        scaledCoords.x -= pad_x;
        scaledCoords.y -= pad_y;
    }
    scaledCoords.x /= gain;
    scaledCoords.y /= gain;
    scaledCoords.width /= gain;
    scaledCoords.height /= gain;
    clip_boxes(scaledCoords, img0_shape);
    return scaledCoords;
}

// Scale coordinates from one image shape to another.
inline std::vector<float> scale_coords(const cv::Size &img1_shape, std::vector<float> &coords, const cv::Size &img0_shape) {
    std::vector<float> scaledCoords = coords;
    double gain = std::min(static_cast<double>(img1_shape.width) / img0_shape.width,
                           static_cast<double>(img1_shape.height) / img0_shape.height);
    cv::Point2d pad((img1_shape.width - img0_shape.width * gain) / 2,
                    (img1_shape.height - img0_shape.height * gain) / 2);
    for (size_t i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i]   = (scaledCoords[i]   - pad.x) / gain;
        scaledCoords[i+1] = (scaledCoords[i+1] - pad.y) / gain;
    }
    // Optionally clip coordinates to image bounds (if desired)
    for (size_t i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i]   = std::min(std::max(scaledCoords[i], 0.0f), static_cast<float>(img0_shape.width - 1));
        scaledCoords[i+1] = std::min(std::max(scaledCoords[i+1], 0.0f), static_cast<float>(img0_shape.height - 1));
    }
    return scaledCoords;
}

// Non-Maximum Suppression (NMS)
inline std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat &output0, int class_names_num, int data_width, double conf_threshold, float iou_threshold) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> rest;

    int rest_start_pos = class_names_num + 4;
    int rest_features = data_width - rest_start_pos;
    int rows = output0.rows;
    float* pdata = reinterpret_cast<float*>(output0.data);

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);
        if (max_conf > conf_threshold) {
            std::vector<float> mask_data(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back(static_cast<float>(max_conf));
            float out_w = pdata[2], out_h = pdata[3];
            float out_left = std::max((pdata[0] - 0.5f * out_w + 0.5f), 0.0f);
            float out_top  = std::max((pdata[1] - 0.5f * out_h + 0.5f), 0.0f);
            cv::Rect_<float> bbox(out_left, out_top, (out_w + 0.5f), (out_h + 0.5f));
            boxes.push_back(bbox);
            if (rest_features > 0) {
                std::vector<float> rest_data(pdata + rest_start_pos, pdata + data_width);
                rest.push_back(rest_data);
            }
        }
        pdata += data_width;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result);
    std::vector<int> nms_class_ids;
    std::vector<float> nms_confidences;
    std::vector<cv::Rect> nms_boxes;
    std::vector<std::vector<float>> nms_rest;
    for (int idx : nms_result) {
        nms_class_ids.push_back(class_ids[idx]);
        nms_confidences.push_back(confidences[idx]);
        nms_boxes.push_back(boxes[idx]);
        nms_rest.push_back(rest[idx]);
    }
    return std::make_tuple(nms_boxes, nms_confidences, nms_class_ids, nms_rest);
}
