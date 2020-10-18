#ifndef hunga_H_
#define hunga_H_
#include "utils.h"
#include "Hungarian.h"

struct r {
    Mat m;
    vector<Point2f> centers;
    Mat pc;
    vector<vector<float>> stereo_correspondences;
    vector<vector<float>> bboxes;
};

void find_left_score(unordered_map<int, float>*left_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k);
void find_right_score(unordered_map<int, float>*right_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k);
vector<vector<float>> run_hungarian(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) ;
void write_correspondences(Mat img_l, Mat img_r, int frame_id, vector<vector<float>> stereo_correspondences);
void write_correspondences_temporal(Mat img_l, Mat img_r, int frame_id, vector<vector<float>> stereo_correspondences);



#endif