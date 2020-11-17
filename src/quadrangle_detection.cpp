#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void detect_quadrangles(cv::Mat image, std::vector<cv::Vec8i>& quadrangles);
void simplify_contours(int mininium_area, std::vector<std::vector<cv::Point>>& contours);
bool get_quadrangle_from_contours(cv::Mat image, std::vector<cv::Point> contour, cv::Vec8i& edges);
void simplify_lines(cv::Mat image, std::vector<cv::Vec4i>& lines);
void pick_lines(std::vector<cv::Vec4i>& lines);
bool get_quadrangle_corners(std::vector<cv::Vec4i> lines, cv::Vec8i& intersections);
int get_scalar_product(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2);
float get_euclid_distance_float(cv::Point point_1, cv::Point point_2);
int get_euclid_distance_int(cv::Point point_1, cv::Point point_2);
float get_cos_two_lines(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2);
int get_triagle_area(cv::Point point_1, cv::Point point_2, cv::Point point_3);
int get_quadrangle_area(cv::Point point_1, cv::Point point_2, cv::Point point_3, cv::Point point_4);
cv::Point get_intersection(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2);

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

    while (true)
    {
        cv::Mat original_image;
        cap >> original_image;

        if (original_image.empty())
        {
            break;
        }

        cv::Mat resized_image;
        float rate = original_image.cols / 400.0f;
        cv::resize(original_image, resized_image, cv::Size(400, round(original_image.rows / rate)));
        cv::resize(original_image, original_image, cv::Size(400, round(original_image.rows / rate)));

        std::vector<cv::Vec8i> quadrangles;
        detect_quadrangles(resized_image, quadrangles);

        cv::Mat output_image = original_image.clone();
        cv::resize(output_image, output_image, cv::Size(output_image.cols * 2, output_image.rows * 2));
        for (unsigned int i = 0; i < quadrangles.size(); ++i)
        {
            cv::line(output_image, cv::Point(2 * quadrangles[i][0], 2 * quadrangles[i][1]), cv::Point(2 * quadrangles[i][2], 2 * quadrangles[i][3]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(output_image, cv::Point(2 * quadrangles[i][2], 2 * quadrangles[i][3]), cv::Point(2 * quadrangles[i][4], 2 * quadrangles[i][5]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(output_image, cv::Point(2 * quadrangles[i][4], 2 * quadrangles[i][5]), cv::Point(2 * quadrangles[i][6], 2 * quadrangles[i][7]), cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::line(output_image, cv::Point(2 * quadrangles[i][6], 2 * quadrangles[i][7]), cv::Point(2 * quadrangles[i][0], 2 * quadrangles[i][1]), cv::Scalar(0, 0, 255), 2, 8, 0);
        }

        cv::imshow("output_image", output_image);
        cv::waitKey(1);
    }
    cv::waitKey(0);

    return 0;
}

void detect_quadrangles(cv::Mat image, std::vector<cv::Vec8i>& quadrangles)
{
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    ///Use hue channel
    cv::Mat hue_image;
    hue_image.create(hsv_image.size(), hsv_image.depth());
    int from_to[] = {0, 0};
    cv::mixChannels(&hsv_image, 1, &hue_image, 1, from_to, 1);

    ///threshold and close hue channel
    cv::threshold(hue_image, hue_image, 70, 255, cv::THRESH_BINARY);
    cv::dilate(hue_image, hue_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    cv::erode(hue_image, hue_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

    ///Find contours of hue channel
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(hue_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    ///Remove noise contours
    simplify_contours(2000, contours);

    ///find quadrangle in each contour
    cv::Mat contours_image = cv::Mat::zeros(hue_image.size(), CV_8UC1);
    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        cv::Vec8i edges;
        if (get_quadrangle_from_contours(contours_image, contours[i], edges))
        {
            quadrangles.push_back(edges);
        }
    }
}

void simplify_contours(int mininium_area, std::vector<std::vector<cv::Point>>& contours)
{
    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        if (cv::contourArea(contours[i]) < mininium_area)
        {
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i--;
        }
    }
}

bool get_quadrangle_from_contours(cv::Mat image, std::vector<cv::Point> contour, cv::Vec8i& edges)
{
    cv::Mat contours_image = cv::Mat::zeros(image.size(), CV_8UC1);

    for (unsigned int i = 0; i < contour.size() - 1; ++i)
    {
        cv::line(contours_image, contour[i], contour[i + 1], 255, 1, 8, 0);
    }
    cv::line(contours_image, contour[0], contour[contour.size() - 1], 255, 1, 8, 0);

    ///find edge lines
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(contours_image, lines, 1, CV_PI / 180, 20, 20, 15);

    simplify_lines(contours_image, lines);

    ///sort lines by length
    std::sort(lines.begin(), lines.end(), [](const cv::Vec4i& line_1, const cv::Vec4i& line_2) {
        int length_1 = get_euclid_distance_int(cv::Point(line_1[0], line_1[1]), cv::Point(line_1[2], line_1[3]));
        int length_2 = get_euclid_distance_int(cv::Point(line_2[0], line_2[1]), cv::Point(line_2[2], line_2[3]));
        return (length_1 < length_2);
    });

    ///pick 4 best lines
    pick_lines(lines);
    if (lines.size() != 4)
    {
        return false;
    }

    cv::Vec8i intersections;
    if (get_quadrangle_corners(lines, intersections))
    {
        edges = intersections;
    }
    else
    {
        return false;
    }

    return true;
}

void simplify_lines(cv::Mat image, std::vector<cv::Vec4i>& lines)
{
    for (unsigned int i = 0; i < lines.size(); ++i)
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

void pick_lines(std::vector<cv::Vec4i>& lines)
{
    for (unsigned int i = 0; i < lines.size(); ++i)
    {
        bool have = false;
        for (unsigned int j = i + 1; j < lines.size(); ++j)
        {
            float cos_two_lines = get_cos_two_lines(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]));

            if (cos_two_lines > 0.8f || cos_two_lines < -0.8f)
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

bool get_quadrangle_corners(std::vector<cv::Vec4i> lines, cv::Vec8i& intersections)
{
    cv::Point line_1_point_1 = cv::Point(lines[0][0], lines[0][1]);
    cv::Point line_1_point_2 = cv::Point(lines[0][2], lines[0][3]);
    cv::Point line_2_point_1 = cv::Point(lines[1][0], lines[1][1]);
    cv::Point line_2_point_2 = cv::Point(lines[1][2], lines[1][3]);
    cv::Point line_3_point_1 = cv::Point(lines[2][0], lines[2][1]);
    cv::Point line_3_point_2 = cv::Point(lines[2][2], lines[2][3]);
    cv::Point line_4_point_1 = cv::Point(lines[3][0], lines[3][1]);
    cv::Point line_4_point_2 = cv::Point(lines[3][2], lines[3][3]);

    float cos_two_lines_1_2 = get_cos_two_lines(line_1_point_1, line_1_point_2, line_2_point_1, line_2_point_2);
    float cos_two_lines_1_3 = get_cos_two_lines(line_1_point_1, line_1_point_2, line_3_point_1, line_3_point_2);
    float cos_two_lines_1_4 = get_cos_two_lines(line_1_point_1, line_1_point_2, line_4_point_1, line_4_point_2);

    std::vector<cv::Point> this_intersections;
    if (cos_two_lines_1_2 > 0.8f || cos_two_lines_1_2 < -0.8f)
    {
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_3_point_1, line_3_point_2));
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_4_point_1, line_4_point_2));
        this_intersections.push_back(get_intersection(line_2_point_1, line_2_point_2, line_3_point_1, line_3_point_2));
        this_intersections.push_back(get_intersection(line_2_point_1, line_2_point_2, line_4_point_1, line_4_point_2));
    }
    else if (cos_two_lines_1_3 > 0.8f || cos_two_lines_1_3 < -0.8f)
    {
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_2_point_1, line_2_point_2));
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_4_point_1, line_4_point_2));
        this_intersections.push_back(get_intersection(line_3_point_1, line_3_point_2, line_2_point_1, line_2_point_2));
        this_intersections.push_back(get_intersection(line_3_point_1, line_3_point_2, line_4_point_1, line_4_point_2));
    }
    else if (cos_two_lines_1_4 > 0.8f || cos_two_lines_1_4 < -0.8f)
    {
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_2_point_1, line_2_point_2));
        this_intersections.push_back(get_intersection(line_1_point_1, line_1_point_2, line_3_point_1, line_3_point_2));
        this_intersections.push_back(get_intersection(line_4_point_1, line_4_point_2, line_2_point_1, line_2_point_2));
        this_intersections.push_back(get_intersection(line_4_point_1, line_4_point_2, line_3_point_1, line_3_point_2));
    }
    else
    {
        return false;
    }

    cv::convexHull(cv::Mat(this_intersections).clone(), this_intersections);
    intersections[0] = this_intersections[0].x;
    intersections[1] = this_intersections[0].y;
    intersections[2] = this_intersections[1].x;
    intersections[3] = this_intersections[1].y;
    intersections[4] = this_intersections[2].x;
    intersections[5] = this_intersections[2].y;
    intersections[6] = this_intersections[3].x;
    intersections[7] = this_intersections[3].y;
    return true;
}

int get_scalar_product(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2)
{
    cv::Point vector_1 = line_1_point_2 - line_1_point_1;
    cv::Point vector_2 = line_2_point_2 - line_2_point_1;
    return vector_1.x * vector_2.x + vector_1.y * vector_2.y;
}

float get_euclid_distance_float(cv::Point point_1, cv::Point point_2)
{
    int d_x = point_2.x - point_1.x;
    int d_y = point_2.y - point_1.y;
    return sqrtf((float)(d_x * d_x + d_y * d_y));
}

float get_cos_two_lines(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2)
{
    float upper = (float)get_scalar_product(line_1_point_1, line_1_point_2, line_2_point_1, line_2_point_2);
    float under = get_euclid_distance_float(line_1_point_1, line_1_point_2) * get_euclid_distance_float(line_2_point_1, line_2_point_2);
    return upper / under;
}

int get_euclid_distance_int(cv::Point point_1, cv::Point point_2)
{
    int d_x = point_2.x - point_1.x;
    int d_y = point_2.y - point_1.y;
    return round(sqrtf((float)(d_x * d_x + d_y * d_y)));
}

int get_triagle_area(cv::Point point_1, cv::Point point_2, cv::Point point_3)
{
    int a = get_euclid_distance_int(point_1, point_2);
    int b = get_euclid_distance_int(point_2, point_3);
    int c = get_euclid_distance_int(point_3, point_1);
    int upper = round(sqrtf((float)(a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)));
    return upper / 4;
}

int get_quadrangle_area(cv::Point point_1, cv::Point point_2, cv::Point point_3, cv::Point point_4)
{
    int area_1 = get_triagle_area(point_1, point_2, point_3);
    int area_2 = get_triagle_area(point_1, point_3, point_4);
    return area_1 + area_2;
}

cv::Point get_intersection(cv::Point line_1_point_1, cv::Point line_1_point_2, cv::Point line_2_point_1, cv::Point line_2_point_2)
{
    cv::Point vector_1 = line_1_point_2 - line_1_point_1;
    cv::Point vector_2 = line_2_point_2 - line_2_point_1;
    float a_1 = (float)(-vector_1.y);
    float b_1 = (float)(vector_1.x);
    float c_1 = a_1 * line_1_point_1.x + b_1 * line_1_point_1.y;
    float a_2 = (float)(-vector_2.y);
    float b_2 = (float)(vector_2.x);
    float c_2 = a_2 * line_2_point_1.x + b_2 * line_2_point_1.y;
    float delta = a_1 * b_2 - a_2 * b_1;
    float delta_x = c_1 * b_2 - c_2 * b_1;
    float delta_y = a_1 * c_2 - a_2 * c_1;

    if (delta != 0)
    {
        cv::Point point;
        point.x = (int)floor(delta_x / delta + 0.5f);
        point.y = (int)floor(delta_y / delta + 0.5f);
        return point;
    }
    else
    {
        return cv::Point(0, 0);
    }
}
