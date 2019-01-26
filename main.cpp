#include "src/timer.hpp"
#include <fmt/printf.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
String track_name;

struct ConvolutionClassifier{
    explicit ConvolutionClassifier(const String &model_path) : classifier(dnn::readNetFromTensorflow(model_path)) {}



    std::vector<std::string>
    extract_IDs(const std::vector<std::pair<UMat, Rect2i>> &plates_and_theirLocations) {
        std::vector<std::string> ret;
        for (const auto &p : plates_and_theirLocations) {
            UMat bw;
            threshold(p.first, bw, 80, 255, THRESH_BINARY_INV);
//            Mat structuringElement = getStructuringElement(MORPH_RECT, {3, 1});
//            erode(bw, bw, structuringElement);
            imshow(track_name, bw);
            waitKey(0);

            std::vector<std::vector<Point2i>> contours;
            findContours(bw, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            std::vector<Rect2i> boxes = extract_boxes_from_contours(contours);
            boxes = filterBoxes_bySize(std::move(boxes), bw);
            fmt::print("boxes.size: {}\n", boxes.size());

            std::string id = extract_ID(bw, std::move(boxes));
            ret.push_back(std::move(id));
        }

        return ret;
    }

private:
    dnn::Net classifier;
    std::array<char, 30> classes{'0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
    const size_t numCharacters = 7;
    const char nullChar = '~';


    std::vector<Rect2i> extract_boxes_from_contours(const std::vector<std::vector<Point2i>> &contours) {
        std::vector<Rect2i> ret; ret.reserve(contours.size());
        for (auto &c : contours) ret.push_back(boundingRect(c));
        return ret;
    }

    std::vector<Rect2i> filterBoxes_bySize(std::vector<Rect2i> &&allBoxes, const UMat &bw) {
        std::vector<Rect2i> ret;
        constexpr int maxWidth = 30, maxHeight = 33, minHeight = 20;
        constexpr float maxAreaRatio = 0.8f;
        for (auto &box: allBoxes) {
            const bool sizeQualified = box.width < maxWidth and box.height > minHeight and box.height < maxHeight;
            const bool areaQualified = (countNonZero(UMat(bw, box)) / static_cast<float>(box.area())) < maxAreaRatio;
            if(sizeQualified and areaQualified) ret.push_back(std::move(box));
        }
        return ret;
    }

    UMat adjust_size(const UMat &input) {
        constexpr int networkInput_width = 20;
        const int boarderThickness = (input.rows - input.cols) / 2;
        UMat padded;
        copyMakeBorder(input, padded, 0, 0, boarderThickness, boarderThickness, BORDER_CONSTANT);
        UMat resized;
        resize(padded, resized, {networkInput_width, networkInput_width});
        return resized;
    }

    char classify_character(const UMat &mat) {
        constexpr double confidenceThreshold = 0.3;
        Mat net_in = dnn::blobFromImage(mat);
        classifier.setInput(net_in);
        Mat net_out;
        classifier.forward(net_out);
        double maxVal;
        Point2i maxVal_location;
        minMaxLoc(net_out, nullptr, &maxVal, nullptr, &maxVal_location);
        if (maxVal < confidenceThreshold) {
            return nullChar;
        }else{
            return classes[maxVal_location.x];
        }
    }

    String  extract_ID(const UMat &bw, std::vector<Rect2i> &&boxes) {
        String id{};
        std::vector<std::pair<char,int>> characters_and_theirOrders;
        for (const auto &b : boxes) {
            UMat characterImage(bw, b);
            characterImage = adjust_size(characterImage);
            const char result = classify_character(characterImage);
            if (result != nullChar) {
                characters_and_theirOrders.emplace_back(result, b.x);
                fmt::print("{}-", result);
            }
        }
        if(characters_and_theirOrders.size() != numCharacters) return {"Number of characters doesn't match"};

        std::sort(characters_and_theirOrders.begin(), characters_and_theirOrders.end(),
                [](const auto& a, const auto& b){return a.second< b.second;});
        for (const auto &pair : characters_and_theirOrders) id.push_back(pair.first);
        fmt::print("\n");
        return id;
    }

};

std::vector<std::pair<UMat, Rect2i>> getSubregionCandidates(const UMat &gray);

void show(const std::vector<std::pair<UMat, Rect2i>> &input, const UMat &image);

std::vector<std::pair<UMat, Rect2i>> verifyCandidates(std::vector<std::pair<UMat, Rect2i>> &candidates, const Ptr<ml::SVM> &SVM_ptr);


int main(const int argc, const char *argv[]) {
    Ptr<ml::SVM> SVM_ptr = ml::SVM::load("../SVM-weights.yaml");
    ConvolutionClassifier classifier("../model.pb");
    std::vector<String> filenames;
    const String image_dir = "/home/afterburner/CLionProjects/ANPR/test";
    utils::fs::glob(image_dir, "*.jpg", filenames);
    utils::fs::glob(image_dir, "*.JPG", filenames);

    for (const auto &name : filenames) {
        fmt::print("Processing image: {}\n", name);
        track_name = name;
        Mat src = cv::imread(name, cv::IMREAD_COLOR);
        if (src.empty()) {
            fmt::print("Error reading image <{}>\n", name);
            return -1;
        }
        UMat cl_src = src.getUMat(ACCESS_READ), cl_gray;
        cvtColor(cl_src, cl_gray, COLOR_BGR2GRAY);
        equalizeHist(cl_gray, cl_gray);

        std::vector<std::pair<UMat, Rect2i>> candidates = getSubregionCandidates(cl_gray);
        if (candidates.empty()) {
            fmt::print("Candidates for plates don't exist in image <{}>\n", name);
            continue;
        }
//        show(candidates, cl_src);
        std::vector<std::pair<UMat, Rect2i>> verified = verifyCandidates(candidates, SVM_ptr);
//        show(verified, cl_src);
        std::vector<std::string> potential_IDs = classifier.extract_IDs(verified);
        if (!potential_IDs.empty()) {
            for (const auto &id : potential_IDs) {
                fmt::print("Plate ID: {}\n", id);
            }
        }
        //    showResult(potential_IDs, verified);
        fmt::print("\n\n");
    }
}



std::vector<std::pair<UMat, Rect2i>> verifyCandidates(std::vector<std::pair<UMat, Rect2i>> &candidates, const Ptr<ml::SVM> &SVM_ptr) {
    std::vector<std::pair<UMat, Rect2i>> ret;
    for (auto &p : candidates) {
        equalizeHist(p.first, p.first);
        resize(p.first, p.first, {144, 33});
        UMat SVM_input;
        p.first.reshape(1, 1).convertTo(SVM_input, CV_32F);
        if (SVM_ptr->predict(SVM_input) > 0) {
            ret.push_back(std::move(p));
        }
    }
    fmt::print("SVM: {} in {} out\n", candidates.size(), ret.size());
    return ret;
}

void show(const std::vector<std::pair<UMat, Rect2i>> &input, const UMat &image) {
    for (const auto &p : input) {
        auto canvas = image.clone();
        rectangle(canvas, p.second, {255, 255, 0}, 2);
        imshow("Whole", canvas);
        imshow("Part", p.first);
        waitKey(0);
    }
}

std::vector<RotatedRect> get_rotatedRects(const std::vector<std::vector<Point2i>> &contours) {
    std::vector<RotatedRect> rotatedRects;
    rotatedRects.reserve(contours.size());
    for (const auto &c : contours) {
        rotatedRects.emplace_back(minAreaRect(c));
    }
    return rotatedRects;
}
template <typename T> bool is_ratioQualified(const T &r) {
    float ar;
    if constexpr (std::is_same_v<T, RotatedRect>) {
        ar = r.size.width / r.size.height;
    } else {
        ar = r.width / r.height;
    }
    if (ar < 1.f)
        ar = 1.f / ar;

    constexpr float aspectRatio = 4.7272f, fluctuation = 1.f;
    constexpr float low_AR = aspectRatio - fluctuation;
    constexpr float high_AR = aspectRatio + fluctuation;

    return (ar > low_AR and ar < high_AR);
}

std::vector<RotatedRect> filter_rotatedRects_bySize(const std::vector<RotatedRect> &rotatedRects) {
    std::vector<RotatedRect> ret;
    for (const auto &r : rotatedRects) {
        const float minLength = min(r.size.width, r.size.height);
        if (is_ratioQualified(r) and minLength > 15 and minLength < 50)
            ret.push_back(r);
    }
    return ret;
}

std::vector<std::pair<UMat, Rect2i>> get_uprightPlates_and_theirLocations(const UMat &image, const std::vector<RotatedRect> &rts) {
    Mat bw;
    std::vector<std::pair<UMat, Rect2i>> ret;
    ret.reserve(rts.size());
    for (const auto &rt : rts) {
        float angle;
        int height, width;
        if (rt.size.width > rt.size.height) {
            angle = rt.angle;
            height = static_cast<int>(rt.size.height);
            width = static_cast<int>(rt.size.width);
        } else {
            angle = rt.angle + 90.f;
            height = static_cast<int>(rt.size.width);
            width = static_cast<int>(rt.size.height);
        }
        const int halfWidth_cropped = static_cast<const int>(0.75f * width);
        const int halfHeight_cropped = static_cast<const int>(0.75f * height);
        const Rect2i rect_cropped{static_cast<int>(rt.center.x - halfWidth_cropped), static_cast<int>(rt.center.y - halfHeight_cropped), 2 * halfWidth_cropped, 2 * halfHeight_cropped};
        if (rect_cropped.x < 0 or rect_cropped.y < 0 or (rect_cropped.x + rect_cropped.width) > image.cols or (rect_cropped.y + rect_cropped.height) > image.rows) {
            continue;
        }
        Mat rotationMatrix = getRotationMatrix2D(Point2f(halfWidth_cropped, halfHeight_cropped), angle, 1);
        UMat rotationRetified, cropped{image, rect_cropped};
        warpAffine(cropped, rotationRetified, rotationMatrix, {}, INTER_AREA);
        threshold(rotationRetified, bw, 100, 255, THRESH_BINARY);

        const int interval = height / 3;
        std::vector<Point2i> seeds;
        seeds.reserve(8);
        for (int i = 1; i < 5; i++) {
            seeds.emplace_back(halfWidth_cropped - i * interval, halfHeight_cropped);
            seeds.emplace_back(halfWidth_cropped + i * interval, halfHeight_cropped);
        }
        UMat mask = UMat::zeros(rotationRetified.rows + 2, rotationRetified.cols + 2, CV_8UC1);
        for (const auto &s : seeds) {
            floodFill(bw, mask, s, {}, nullptr, 5, 5, 4 | (255 << 8) | FLOODFILL_MASK_ONLY);
        }
        UMat mask_fitted = UMat(mask, Rect2i{1, 1, rotationRetified.cols, rotationRetified.rows});
        Mat kernel = getStructuringElement(MORPH_RECT, {15, 3});
        morphologyEx(mask_fitted, mask_fitted, MORPH_OPEN, kernel);
        threshold(mask_fitted, mask_fitted, 100, 255, THRESH_BINARY);
        ret.emplace_back(UMat{rotationRetified, boundingRect(mask_fitted)}, rt.boundingRect());
    }

    return ret;
}

std::vector<std::pair<UMat, Rect2i>> getSubregionCandidates(const UMat &gray) {
    UMat edgeImage, blurred;
    GaussianBlur(gray, blurred, {5, 5}, 1.5);
    Sobel(blurred, edgeImage, CV_8U, 1, 0);
    threshold(edgeImage, edgeImage, 100, 255, THRESH_BINARY);
    Mat structuringElement = getStructuringElement(MORPH_RECT, {17, 3});
    morphologyEx(edgeImage, edgeImage, MORPH_CLOSE, structuringElement);

    std::vector<std::vector<Point2i>> allContours;
    findContours(edgeImage, allContours, {}, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::vector<RotatedRect> all_rotatedRects = get_rotatedRects(allContours);

    std::vector<RotatedRect> rotatedRects_sizeQualified = filter_rotatedRects_bySize(all_rotatedRects);
    if (rotatedRects_sizeQualified.empty())
        return {};

    std::vector<std::pair<UMat, Rect2i>> uprightPlates_and_theirLocations = get_uprightPlates_and_theirLocations(blurred, rotatedRects_sizeQualified);

    return uprightPlates_and_theirLocations;
}
