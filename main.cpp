#include <opencv2/opencv.hpp>
#include <fmt/printf.h>
#include "src/timer.hpp"

using namespace cv;

std::vector<std::pair<Rect2i, UMat>> getSubregionCandidates(InputArray gray);


int main(const int argc, const char *argv[])
{
    const cv::CommandLineParser parser(
        argc, argv,
        "{help ? h ||}"
        "{@input | /home/afterburner/CLionProjects/ANPR/test/2715DTZ.jpg | input image}"
        "{scale_factor| 0.5| image scale factor}"
        "{rotate | false | true when dealing with things like book pages }");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    const auto image_path = parser.get<std::string>("@input");
    Mat src = cv::imread(image_path, cv::IMREAD_COLOR);
    if (src.empty())
    {
        fmt::print("Error reading image <{}>\n", image_path);
        return -1;
    }
    UMat cl_src = src.getUMat(ACCESS_READ), cl_gray;
    cvtColor(cl_src, cl_gray, COLOR_BGR2GRAY);

    std::vector<std::pair<Rect2i, UMat>> candidates = getSubregionCandidates(cl_gray);
//    show(candidates);
//    std::vector<std::pair<Rect2i, Mat>> verified = verifyCandidates(std::move(candidates));
//    show(verified);
//    String id = extractId(verified);
//    showResult(id, verified);

}

std::vector<RotatedRect> get_rotatedRects(const std::vector<std::vector<Point2i>> &contours) {
    std::vector<RotatedRect> rotatedRects;
    rotatedRects.reserve(contours.size());
    for (const auto &c : contours) {
        rotatedRects.emplace_back(minAreaRect(c));
    }
    return rotatedRects;
}

std::vector<RotatedRect> filter_rotatedRects_bySize(const std::vector<RotatedRect> &rotatedRects) {
    constexpr float aspectRatio = 4.7272f, fluctuation = 1.5f;
    constexpr float low_AR = aspectRatio - fluctuation;
    constexpr float high_AR = aspectRatio + fluctuation;
    std::vector<RotatedRect> ret;
    for (const auto &r : rotatedRects) {
        float ar = r.size.width / r.size.height;
        if(ar < 1.f) ar = 1.f / ar;
        const float minLength = min(r.size.width, r.size.height);
        if(ar>low_AR and ar<high_AR and minLength > 15 and minLength < 100) ret.push_back(r);
    }
    return ret;
}


UMat get_upRight_image(InputArray image, const RotatedRect &rt) {
    float angle;
    if (rt.size.width > rt.size.height) {
        angle = rt.angle;
    }else{
        angle = rt.angle + 90.f;
    }

    Mat rotationMatrix = getRotationMatrix2D(rt.center, angle, 1);
    UMat ret;
    warpAffine(image, ret, rotationMatrix, {});
    return ret;
}


std::vector<std::pair<Rect2i, UMat>> getSubregionCandidates(InputArray gray) {
    UMat edgeImage, blurred;
    GaussianBlur(gray, blurred, {5, 5}, 1.5);
    Sobel(blurred, edgeImage, CV_8U, 1, 0);
    threshold(edgeImage, edgeImage, 100, 255, THRESH_BINARY);
    Mat structuringElement = getStructuringElement(MORPH_RECT, {17, 3});
    morphologyEx(edgeImage, edgeImage, MORPH_CLOSE, structuringElement);
    std::vector<std::vector<Point2i>> allContours;
    findContours(edgeImage, allContours, {}, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::vector<RotatedRect> all_rotatedRects = get_rotatedRects(allContours);
    std::vector<RotatedRect> rotatedRects_sizeQualified = filter_rotatedRects_bySize(all_rotatedRects); //TODO: Return noopt if size = 0

    std::vector<std::pair<Rect2i, UMat>> upRightRects_AND_rotationRectifiedImages;
    upRightRects_AND_rotationRectifiedImages.reserve(rotatedRects_sizeQualified.size());
    for (const auto &rt : rotatedRects_sizeQualified) {
        upRightRects_AND_rotationRectifiedImages.emplace_back(rt.boundingRect(), get_upRight_image(blurred, rt));
    }

    for (const auto &p : upRightRects_AND_rotationRectifiedImages) {
        imshow("debug", p.second);
        waitKey(0);
    }


    return std::vector<std::pair<Rect2i, UMat>>();
}
