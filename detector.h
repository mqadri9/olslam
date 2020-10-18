#ifndef DETECTOR_H_
#define DETECTOR_H_
#include "utils.h"
#include "hungarian.h"


vector<vector<float>> find_correspondences_surf(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype);
vector<vector<float>> find_correspondences_sift(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype);


#endif