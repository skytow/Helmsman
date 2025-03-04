#pragma once

#include <opencv2/opencv.hpp>
#include <utility>

// Letterbox resize: resize image while keeping aspect ratio and add border padding.
inline void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape = cv::Size(640, 640),
                      cv::Scalar color = cv::Scalar(114, 114, 114), bool auto_ = true,
                      bool scaleFill = false, bool scaleUp = true, int stride = 32) {
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / shape.height,
                       static_cast<float>(newShape.width) / shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);
    int newUnpadW = std::round(shape.width * r);
    int newUnpadH = std::round(shape.height * r);
    float dw = newShape.width - newUnpadW;
    float dh = newShape.height - newUnpadH;
    if (auto_) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0;
        dh = 0;
        newUnpadW = newShape.width;
        newUnpadH = newShape.height;
        r = static_cast<float>(newShape.width) / shape.width;
    }
    dw /= 2.0f;
    dh /= 2.0f;
    cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH));
    int top = std::round(dh - 0.1f);
    int bottom = std::round(dh + 0.1f);
    int left = std::round(dw - 0.1f);
    int right = std::round(dw + 0.1f);
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

// Scale image: resize a mask (or other image) to the target shape.
inline cv::Mat scale_image(const cv::Mat &resized_mask, const cv::Size &im0_shape,
                           const std::pair<float, cv::Point2f> &ratio_pad = {-1.0f, cv::Point2f(-1.0f, -1.0f)}) {
    if (resized_mask.size() == im0_shape)
        return resized_mask.clone();
    float gain, pad_x, pad_y;
    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(resized_mask.rows) / im0_shape.height,
                        static_cast<float>(resized_mask.cols) / im0_shape.width);
        pad_x = (resized_mask.cols - im0_shape.width * gain) / 2.0f;
        pad_y = (resized_mask.rows - im0_shape.height * gain) / 2.0f;
    } else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }
    cv::Rect clipped_rect(static_cast<int>(pad_x), static_cast<int>(pad_y),
                            resized_mask.cols - static_cast<int>(pad_x * 2),
                            resized_mask.rows - static_cast<int>(pad_y * 2));
    cv::Mat clipped_mask = resized_mask(clipped_rect);
    cv::Mat scaled_mask;
    cv::resize(clipped_mask, scaled_mask, im0_shape);
    return scaled_mask;
}

// Scale image 2: convenience wrapper that writes result to scaled_mask.
inline void scale_image2(cv::Mat &scaled_mask, const cv::Mat &resized_mask, const cv::Size &im0_shape,
                         const std::pair<float, cv::Point2f> &ratio_pad = {-1.0f, cv::Point2f(-1.0f, -1.0f)}) {
    scaled_mask = scale_image(resized_mask, im0_shape, ratio_pad);
}
