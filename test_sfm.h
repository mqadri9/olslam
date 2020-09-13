#ifndef TESTSFM_H_
#define TESTSFM_H_

#include "utils.h"

void test_sfm(gtsam::Values, Cal3_S2::shared_ptr, vector<int>);
void reconstruct_pointcloud(gtsam::Values, Cal3_S2::shared_ptr, vector<int>);
bool check_bound(float, float, vector<vector<float>> );
vector<vector<float>> get_3d_bounds(int, Mat);
int NumDigits(int);  
void result_to_vtk(gtsam::Values, int);

#endif