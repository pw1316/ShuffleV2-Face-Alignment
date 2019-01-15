#include <numeric>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tensorflow_wrapper.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#ifndef _DEBUG
#pragma comment(lib, "opencv_world340.lib")
#else
#pragma comment(lib, "opencv_world340d.lib")
#endif
#pragma comment(lib, "tensorflow-lite.lib")

#define LIVE

#ifdef LIVE
constexpr int CAP_NAME = 0;
constexpr bool IS_FLIP = true;
#else
constexpr const char* CAP_NAME = "a";
constexpr bool IS_FLIP = false;
#endif

bool detect(cv::CascadeClassifier& classifier, const cv::Mat& image, std::vector<int>& ret)
{
    std::vector<cv::Rect> det;
    classifier.detectMultiScale(image, det, 1.2, 3, 0, cv::Size(60, 60));
    if (!det.empty())
    {
        ret = { det[0].y, det[0].x, det[0].y + det[0].height, det[0].x + det[0].width };
        return true;
    }
    return false;
}

int main()
{
    auto model = tfLoadLiteModelFromFile("../../shuffle.tflite", 4);

    auto video = cv::VideoCapture(CAP_NAME);
    auto size = cv::Size(static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));
    auto out_video = cv::VideoWriter("out.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 23.97, size);

    auto classifier = cv::CascadeClassifier("./haarcascade_frontalface_alt2.xml");
    cv::Mat frame;
    auto success = video.read(frame);

#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER totalTime;
    LARGE_INTEGER inferenceTime;
    QueryPerformanceFrequency(&freq);
    totalTime.QuadPart = 0;
    inferenceTime.QuadPart = 0;
#endif
    auto key = 1;
    auto isTracking = false;

    std::vector<int> bbox;
    while (success)
    {
#ifdef _WIN32
        LARGE_INTEGER frame_start;
        LARGE_INTEGER inference_start;
        LARGE_INTEGER inference_end;
        LARGE_INTEGER frame_end;
        QueryPerformanceCounter(&frame_start);
#endif
        if (IS_FLIP)
        {
            cv::flip(frame, frame, 1);
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2RGB);

        if (isTracking)
        {
            auto bbsize = std::max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 4 / 6.0;
            auto cy = (bbox[0] + bbox[2])*0.5;
            auto cx = (bbox[1] + bbox[3])*0.5;
            bbox[0] = static_cast<int>(cy - bbsize);
            bbox[1] = static_cast<int>(cx - bbsize);
            bbox[2] = static_cast<int>(cy + bbsize);
            bbox[3] = static_cast<int>(cx + bbsize);
        }
        else
        {
            if (detect(classifier, gray, bbox))
            {
                auto bbsize = std::max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 4 / 6.0;
                auto cy = (bbox[0] + bbox[2])*0.5 + (bbox[2] - bbox[0])*0.1;
                auto cx = (bbox[1] + bbox[3])*0.5;
                bbox[0] = static_cast<int>(cy - bbsize);
                bbox[1] = static_cast<int>(cx - bbsize);
                bbox[2] = static_cast<int>(cy + bbsize);
                bbox[3] = static_cast<int>(cx + bbsize);
                isTracking = true;
            }
            else
            {
                std::cout << "Frame" << key << " no face detected!\n";
                success = video.read(frame);
                ++key;
                continue;
            }
        }

        std::vector<int> borders = {
            std::max(0, -bbox[0]), std::max(0, -bbox[1]),
            std::max(0, bbox[2] + 1 - size.height), std::max(0, bbox[3] + 1 - size.width),
        };
        auto border = std::accumulate(borders.begin(), borders.end(), 0, [](auto&& x, auto&& y) {return std::max(x, y); });
        for (auto&& i : bbox)
        {
            i += border;
        }
        cv::copyMakeBorder(gray, gray, border, border, border, border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::Mat window = gray(cv::Rect(bbox[1], bbox[0], bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1));
        cv::resize(window, window, cv::Size(112, 112));
        auto fy = (bbox[2] - bbox[0]) / 112.;
        auto fx = (bbox[3] - bbox[1]) / 112.;
        window.convertTo(window, CV_32F);
        window = window / 255.;
        std::cout << *(float*)window.data;

#ifdef _WIN32
        QueryPerformanceCounter(&inference_start);
#endif
        tfSetLiteModelInputTensorFloat(model, 0, reinterpret_cast<float*>(window.data), 112 * 112 * 3);
        tfRunLiteModel(model);
        auto dx = tfGetLiteModelOutputTensorFloat(model, 0);
        auto mean = tfGetLiteModelOutputTensorFloat(model, 1);
#ifdef _WIN32
        QueryPerformanceCounter(&inference_end);
        LARGE_INTEGER duration;
        duration.QuadPart = inference_end.QuadPart - inference_start.QuadPart;
        inferenceTime.QuadPart += duration.QuadPart;
        std::cout << "Frame" << key << " forward time:" << 1000.0 * duration.QuadPart / freq.QuadPart << "\n";
#endif

        std::vector<int> nbbox = { INT_MAX , INT_MAX , -INT_MAX , -INT_MAX };
        for (int i = 0; i < 73; ++i)
        {
            auto y = (mean[2 * i] + dx[2 * i]) * fy + bbox[0] - border;
            auto x = (mean[2 * i + 1] + dx[2 * i + 1]) * fx + bbox[1] - border;
            auto iy = static_cast<int>(y);
            auto ix = static_cast<int>(x);
            nbbox[0] = std::min(nbbox[0], iy);
            nbbox[1] = std::min(nbbox[1], ix);
            nbbox[2] = std::max(nbbox[2], iy);
            nbbox[3] = std::max(nbbox[3], ix);
            cv::circle(frame, cv::Point(ix, iy), 2, cv::Scalar(255, 255, 255), 1);
        }
        bbox = std::move(nbbox);
        cv::rectangle(frame, cv::Rect(bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]), cv::Scalar(0, 0, 255));
        cv::imshow("src", frame);
        auto k = cv::waitKey(1);
        if (k == 114)
        {
            isTracking = false;
        }
        out_video << frame;
        ++key;
#ifdef _WIN32
        QueryPerformanceCounter(&frame_end);
        duration.QuadPart = frame_end.QuadPart - frame_start.QuadPart;
        totalTime.QuadPart += duration.QuadPart;
        std::cout << "Frame" << key << " time:" << 1000.0 * duration.QuadPart / freq.QuadPart << "\n";
#endif
        success = video.read(frame);
    }

#ifdef _WIN32
    std::cout << "mean time:" << 1000.0 * totalTime.QuadPart / freq.QuadPart / key << "\n";
    std::cout << "mean infer time:" << 1000.0 * inferenceTime.QuadPart / freq.QuadPart / key << "\n";
#endif
}
