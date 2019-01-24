#include <opencv2/opencv.hpp>
#include <fmt/printf.h>
#include "src/timer.hpp"

namespace db
{
void imshow(const cv::String &winName_, const cv::Mat &img_)
{
#ifdef DEBUG
    if (img_.depth() == CV_8U)
    {
        cv::imshow(winName_, img_);
    }
    else
    {
        double min_ = 0, max_ = 0;
        cv::Point2i min_pt, max_pt;
        cv::Mat temp;
        cv::minMaxLoc(img_, &min_, &max_, &min_pt, &max_pt, cv::noArray());
        cv::normalize(img_, temp, 0, 255, cv::NORM_MINMAX, CV_8U, {});

        cv::imshow(winName_, temp);
        if (min_ < 0 || max_ > 1)
        {
            fmt::print("[DEBUG] {} is not of any showing formats, a makeshift image "
                       "is create.\nOriginal states:\n",
                       winName_);
            fmt::print("minVal: {} at point ({},{})\n", min_, min_pt.x, min_pt.y);
            fmt::print("maxVal: {} at point ({},{})\n\n", max_, max_pt.x, max_pt.y);
        }
    }
#endif // DEBUG
}
}; // namespace db

using namespace cv;
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

    std::vector<std::pair<Rect2i, Mat>> candidates = getSubregionCandidates(cl_gray);
    show(candidates);
    std::vector<std::pair<Rect2i, Mat>> verified = verifyCandidates(std::move(candidates));
    show(verified);
    String id = extractId(verified);
    showResult(id, verified);

}
