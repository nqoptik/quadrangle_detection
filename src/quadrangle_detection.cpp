#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void detect_quadrangles(cv::Mat image, std::vector<cv::Vec8i>& quadrangles);
void simplify_contours(int minArea, std::vector<std::vector<cv::Point>>& contours);
bool get_quadrangle_from_contours(cv::Mat image, std::vector<cv::Point> contour, cv::Vec8i& edges);
void simplify_lines(cv::Mat image, std::vector<cv::Vec4i>& lines);
void sort_lines(std::vector<cv::Vec4i>& lines);
void pick_lines(std::vector<cv::Vec4i>& lines);
bool get_quadrangle_corners(std::vector<cv::Vec4i> lines, cv::Vec8i& intersects);
int get_scalar_product(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2);
float get_euclid_distance_float(cv::Point p1, cv::Point p2);
int get_euclid_distance_int(cv::Point p1, cv::Point p2);
float get_cos_two_lines(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2);
int get_triagle_area(cv::Point p1, cv::Point p2, cv::Point p3);
int get_quadrangle_area(cv::Point p1, cv::Point p2, cv::Point p3, cv::Point p4);
cv::Point get_intersection(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2);

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("To run quadrangle_detection, type ./quadrangle_detection <video_file>\n");
        return 1;
    }
    cv::VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        std::cout << "Video not found." << std::endl;
        return 0;
    }

    for (;;)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        cv::Mat orgImg = frame.clone();

        cv::Mat resizeImg;
        float rate = orgImg.cols / 400.0f;
        cv::resize(orgImg, resizeImg, cv::Size(400, round(orgImg.rows / rate)));
        cv::resize(orgImg, orgImg, cv::Size(400, round(orgImg.rows / rate)));

        std::vector<cv::Vec8i> quadrangles;
        detect_quadrangles(resizeImg, quadrangles);

        cv::Mat outImg = orgImg.clone();

        cv::resize(outImg, outImg, cv::Size(outImg.cols * 2, outImg.rows * 2));
        for (unsigned int i = 0; i < quadrangles.size(); i++)
        {
            cv::line(outImg, cv::Point(2 * quadrangles[i][0], 2 * quadrangles[i][1]), cv::Point(2 * quadrangles[i][2], 2 * quadrangles[i][3]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(outImg, cv::Point(2 * quadrangles[i][2], 2 * quadrangles[i][3]), cv::Point(2 * quadrangles[i][4], 2 * quadrangles[i][5]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(outImg, cv::Point(2 * quadrangles[i][4], 2 * quadrangles[i][5]), cv::Point(2 * quadrangles[i][6], 2 * quadrangles[i][7]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(outImg, cv::Point(2 * quadrangles[i][6], 2 * quadrangles[i][7]), cv::Point(2 * quadrangles[i][0], 2 * quadrangles[i][1]), cv::Scalar(0, 0, 255), 2, 8, 0);
        }

        cv::imshow("outImg", outImg);

        cv::waitKey(1);
    }
    cv::waitKey(0);

    return 0;
}

void detect_quadrangles(cv::Mat image, std::vector<cv::Vec8i>& quadrangles)
{
    cv::Mat hsvImg;
    cv::cvtColor(image, hsvImg, CV_BGR2HSV);

    ///Use hue channel
    cv::Mat hueImg;
    hueImg.create(hsvImg.size(), hsvImg.depth());
    int from_To[] = {0, 0};
    cv::mixChannels(&hsvImg, 1, &hueImg, 1, from_To, 1);

    ///threshold and close hue channel
    cv::threshold(hueImg, hueImg, 70, 255, CV_THRESH_BINARY);
    cv::dilate(hueImg, hueImg, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    cv::erode(hueImg, hueImg, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

    ///Find contours of hue channel
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(hueImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    ///Remove noise contours
    simplify_contours(2000, contours);

    cv::Mat contoursImg = cv::Mat::zeros(hueImg.size(), CV_8UC1);

    ///find quadrangle in each contour
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        cv::Vec8i edges;
        if (get_quadrangle_from_contours(contoursImg, contours[i], edges))
        {
            quadrangles.push_back(edges);
        }
    }
}

void simplify_contours(int minArea, std::vector<std::vector<cv::Point>>& contours)
{
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        if (cv::contourArea(contours[i]) < minArea)
        {
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i--;
        }
    }
}

bool get_quadrangle_from_contours(cv::Mat image, std::vector<cv::Point> contour, cv::Vec8i& edges)
{
    cv::Mat contourImg = cv::Mat::zeros(image.size(), CV_8UC1);

    for (unsigned int i = 0; i < contour.size() - 1; i++)
    {
        cv::line(contourImg, contour[i], contour[i + 1], 255, 1, 8, 0);
    }
    cv::line(contourImg, contour[0], contour[contour.size() - 1], 255, 1, 8, 0);

    ///find edge lines
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(contourImg, lines, 1, CV_PI / 180, 20, 20, 15);

    simplify_lines(contourImg, lines);

    ///sort lines by length
    sort_lines(lines);

    ///pick 4 best lines
    pick_lines(lines);

    if (lines.size() != 4)
    {
        return false;
    }

    cv::Vec8i intersects;

    if (get_quadrangle_corners(lines, intersects))
    {
        edges = intersects;
    }
    else
    {
        return false;
    }

    return true;
}

void simplify_lines(cv::Mat image, std::vector<cv::Vec4i>& lines)
{
    for (unsigned int i = 0; i < lines.size(); i++)
    {
        if (lines[i][0] < 2 && lines[i][2] < 2)
        {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        }
        else if (lines[i][1] < 2 && lines[i][3] < 2)
        {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        }
        else if (lines[i][0] > image.cols - 3 && lines[i][2] > image.cols - 3)
        {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        }
        else if (lines[i][1] > image.rows - 3 && lines[i][3] > image.rows - 3)
        {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        }
    }

    if (lines.size() < 4)
    {
        lines.erase(lines.begin(), lines.end());
    }
}

void sort_lines(std::vector<cv::Vec4i>& lines)
{
    std::vector<int> dLines;

    for (unsigned int i = 0; i < lines.size(); i++)
    {
        dLines.push_back(get_euclid_distance_int(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3])));
    }
    std::vector<cv::Vec4i> sortedLines;

    unsigned int linesSize = lines.size();

    ///sort lines by length
    for (unsigned int i = 0; i < linesSize; i++)
    {
        int dMax = 0;
        int indexMax = -1;
        for (unsigned int j = 0; j < lines.size(); j++)
        {
            if (dLines[j] > dMax)
            {
                dMax = dLines[j];
                indexMax = j;
            }
        }

        if (indexMax != -1)
        {
            sortedLines.push_back(lines[indexMax]);
            lines.erase(lines.begin() + indexMax, lines.begin() + indexMax + 1);
            dLines.erase(dLines.begin() + indexMax, dLines.begin() + indexMax + 1);
        }
    }
    lines = sortedLines;
}

void pick_lines(std::vector<cv::Vec4i>& lines)
{
    for (unsigned int i = 0; i < lines.size(); i++)
    {
        bool have = false;
        for (unsigned int j = i + 1; j < lines.size(); j++)
        {
            float cosTL = get_cos_two_lines(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]));

            if (cosTL > 0.8f || cosTL < -0.8f)
            {
                if (!have)
                {
                    int area = get_quadrangle_area(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]));

                    if (area > 2000)
                    {
                        have = true;
                    }
                    else
                    {
                        lines.erase(lines.begin() + j, lines.begin() + j + 1);
                        j--;
                    }
                }
                else
                {
                    lines.erase(lines.begin() + j, lines.begin() + j + 1);
                    j--;
                }
            }
        }
    }
}

bool get_quadrangle_corners(std::vector<cv::Vec4i> lines, cv::Vec8i& intersects)
{
    cv::Point line1_p1 = cv::Point(lines[0][0], lines[0][1]);
    cv::Point line1_p2 = cv::Point(lines[0][2], lines[0][3]);
    cv::Point line2_p1 = cv::Point(lines[1][0], lines[1][1]);
    cv::Point line2_p2 = cv::Point(lines[1][2], lines[1][3]);
    cv::Point line3_p1 = cv::Point(lines[2][0], lines[2][1]);
    cv::Point line3_p2 = cv::Point(lines[2][2], lines[2][3]);
    cv::Point line4_p1 = cv::Point(lines[3][0], lines[3][1]);
    cv::Point line4_p2 = cv::Point(lines[3][2], lines[3][3]);

    float cosTL12 = get_cos_two_lines(line1_p1, line1_p2, line2_p1, line2_p2);
    float cosTL13 = get_cos_two_lines(line1_p1, line1_p2, line3_p1, line3_p2);
    float cosTL14 = get_cos_two_lines(line1_p1, line1_p2, line4_p1, line4_p2);

    std::vector<cv::Point> inters;

    if (cosTL12 > 0.8f || cosTL12 < -0.8f)
    {
        inters.push_back(get_intersection(line1_p1, line1_p2, line3_p1, line3_p2));
        inters.push_back(get_intersection(line1_p1, line1_p2, line4_p1, line4_p2));
        inters.push_back(get_intersection(line2_p1, line2_p2, line3_p1, line3_p2));
        inters.push_back(get_intersection(line2_p1, line2_p2, line4_p1, line4_p2));
    }
    else if (cosTL13 > 0.8f || cosTL13 < -0.8f)
    {
        inters.push_back(get_intersection(line1_p1, line1_p2, line2_p1, line2_p2));
        inters.push_back(get_intersection(line1_p1, line1_p2, line4_p1, line4_p2));
        inters.push_back(get_intersection(line3_p1, line3_p2, line2_p1, line2_p2));
        inters.push_back(get_intersection(line3_p1, line3_p2, line4_p1, line4_p2));
    }
    else if (cosTL14 > 0.8f || cosTL14 < -0.8f)
    {
        inters.push_back(get_intersection(line1_p1, line1_p2, line2_p1, line2_p2));
        inters.push_back(get_intersection(line1_p1, line1_p2, line3_p1, line3_p2));
        inters.push_back(get_intersection(line4_p1, line4_p2, line2_p1, line2_p2));
        inters.push_back(get_intersection(line4_p1, line4_p2, line3_p1, line3_p2));
    }
    else
    {
        return false;
    }

    cv::convexHull(cv::Mat(inters).clone(), inters);

    intersects[0] = inters[0].x;
    intersects[1] = inters[0].y;
    intersects[2] = inters[1].x;
    intersects[3] = inters[1].y;
    intersects[4] = inters[2].x;
    intersects[5] = inters[2].y;
    intersects[6] = inters[3].x;
    intersects[7] = inters[3].y;
    return true;
}

int get_scalar_product(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2)
{
    cv::Point v1 = line1_p2 - line1_p1;
    cv::Point v2 = line2_p2 - line2_p1;
    return v1.x * v2.x + v1.y * v2.y;
}

float get_euclid_distance_float(cv::Point p1, cv::Point p2)
{
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    return sqrtf((float)(dx * dx + dy * dy));
}

float get_cos_two_lines(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2)
{
    float upper = (float)get_scalar_product(line1_p1, line1_p2, line2_p1, line2_p2);
    float under = get_euclid_distance_float(line1_p1, line1_p2) * get_euclid_distance_float(line2_p1, line2_p2);
    return upper / under;
}

int get_euclid_distance_int(cv::Point p1, cv::Point p2)
{
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    return round(sqrtf((float)(dx * dx + dy * dy)));
}

int get_triagle_area(cv::Point p1, cv::Point p2, cv::Point p3)
{
    int a = get_euclid_distance_int(p1, p2);
    int b = get_euclid_distance_int(p2, p3);
    int c = get_euclid_distance_int(p3, p1);
    int upper = round(sqrtf((float)(a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)));
    return upper / 4;
}

int get_quadrangle_area(cv::Point p1, cv::Point p2, cv::Point p3, cv::Point p4)
{
    int s1 = get_triagle_area(p1, p2, p3);
    int s2 = get_triagle_area(p1, p3, p4);
    return s1 + s2;
}

cv::Point get_intersection(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2)
{
    cv::Point v1 = line1_p2 - line1_p1;
    cv::Point v2 = line2_p2 - line2_p1;
    float a1 = (float)-v1.y;
    float b1 = (float)v1.x;
    float c1 = a1 * line1_p1.x + b1 * line1_p1.y;
    float a2 = (float)-v2.y;
    float b2 = (float)v2.x;
    float c2 = a2 * line2_p1.x + b2 * line2_p1.y;
    float delta = a1 * b2 - a2 * b1;
    float deltaX = c1 * b2 - c2 * b1;
    float deltaY = a1 * c2 - a2 * c1;

    if (delta != 0)
    {
        cv::Point p;
        p.x = (int)floor(deltaX / delta + 0.5f);
        p.y = (int)floor(deltaY / delta + 0.5f);
        return p;
    }
    else
    {
        return cv::Point(0, 0);
    }
}
