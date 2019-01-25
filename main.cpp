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
    equalizeHist(cl_gray, cl_gray);

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
        if(ar>low_AR and ar<high_AR and minLength > 15 and minLength < 50) ret.push_back(r);
    }
    return ret;
}


std::vector<UMat> get_upright_plates(InputArray image, const std::vector<RotatedRect>& rts) {
    Mat bw;
    std::vector<UMat> ret;
    ret.reserve(rts.size());
    for (const auto &rt: rts) {
        float angle;
        int height;
        if (rt.size.width > rt.size.height) {
            angle = rt.angle;
            height = static_cast<int>(rt.size.height);
        }else{
            angle = rt.angle + 90.f;
            height = static_cast<int>(rt.size.width);
        }
        Mat rotationMatrix = getRotationMatrix2D(rt.center, angle, 1);
        UMat rotationRetified;
        warpAffine(image, rotationRetified, rotationMatrix, image.size());
        threshold(rotationRetified, bw, 100, 255, THRESH_BINARY);
        imshow("bw", bw);
        waitKey(0);
        const int interval = height / 3;
        const int y = static_cast<const int>(rt.center.y);
        const int x = static_cast<const int>(rt.center.x);
        std::vector<Point2i> seeds;
        seeds.reserve(8);
        for (int i = 1; i < 5; i++) {
            seeds.emplace_back(x - i * interval, y);
            seeds.emplace_back(x + i * interval, y);
        }
        UMat mask= UMat::zeros(rotationRetified.rows+2, rotationRetified.cols+2, CV_8UC1);
        for (const auto &s : seeds) {
                floodFill(bw, mask, s, {}, nullptr, 5, 5,
                          4 | (255 << 8) | FLOODFILL_MASK_ONLY);
        }
        UMat mask_fitted = UMat(mask, Rect2i{1,1,rotationRetified.cols, rotationRetified.rows});
//    threshold(mask_fitted, mask_fitted, 100, 255, THRESH_BINARY);
        line(mask_fitted, seeds[6], seeds[7], 0, 5);
        imshow("mask_fitted", mask_fitted);
        waitKey(0);
        //    std::vector<std::vector<Point2i>> contours;
//    findContours(mask_fitted, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        ret.emplace_back(rotationRetified, boundingRect(mask_fitted));
    }


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
    std::vector<RotatedRect> rotatedRects_sizeQualified = filter_rotatedRects_bySize(all_rotatedRects); //TODO: Return noopt if size = 0, ret.first = rt.boundingBox

    std::cout << rotatedRects_sizeQualified.size() << '\n';
    std::vector<UMat> upright_plates = get_upright_plates(blurred, rotatedRects_sizeQualified);
    for (const auto &img : upright_plates) {
        imshow("debug", img);
        waitKey(0);
    }



    return std::vector<std::pair<Rect2i, UMat>>();
}
