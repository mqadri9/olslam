#ifndef UTILS_H_
#define UTILS_H_


#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/StereoFactor.h>

// Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).
#include <gtsam/geometry/Point2.h>

// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Projection factors to model the camera's landmark observations.
// Also, we will initialize the robot at some location using a Prior factor.
#include <gtsam/slam/ProjectionFactor.h>

// We want to use iSAM to solve the structure-from-motion problem incrementally, so
// include iSAM here
#include <gtsam/nonlinear/NonlinearISAM.h>

// iSAM requires as input a set set of new factors to be added stored in a factor graph,
// and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/DoglegOptimizer.h>

#include <gtsam/nonlinear/ISAM2.h>


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <utility> 
#include <stdexcept> 
#include <sstream> 

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "pointmatcher/IO.h"
#include <cassert>
#include <typeinfo>
#include <fstream>
//#include "SFMdata.h"
#include "Procrustes.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>


using namespace boost::accumulators;

using namespace std;
using namespace gtsam;
//#include "boost/filesystem.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

/*
//  DEFINE GLOBAL VARIABLES
static int THRESHOLD_NUMBER_MATCHES = 100;
static float RESIZE_FACTOR = 0.2;
static float focal_length = 2869.381763767118*RESIZE_FACTOR;
static float fy = 2868.360919010879*RESIZE_FACTOR;

static float baseline = 0.089;
static float cx = 2120.291162136175*RESIZE_FACTOR;
static float cxprime = 2120.291162136175*RESIZE_FACTOR;
static float cy = 1401.17755609316*RESIZE_FACTOR;
static string image_folder = "/home/remote_user2/olslam/clusters/clusters2/151_5_30/rect1";
static string data_folder = "/home/remote_user2/olslam/clusters/clusters2/151_5_30/disparities" ;
static float ellipse_resize_factor = 1;
*/

static int THRESHOLD_NUMBER_MATCHES = 100;
static float RESIZE_FACTOR = 0.15;
static float focal_length = 2583.002890739886;
static float fy = 2577.953120526253;
static float MIN_DISPARITY = 1;
static float baseline = 0.11;
static float cx = 2012.978267793323;
static float cxprime = 2012.978267793323;
static float cy = 1525.658605179268;
static string image_folder = "/home/remote_user2/olslam/sorghum_dataset/row_2066_2116/stereo_tmp_seed/rect1_fullres";
static string image_folder_right = "/home/remote_user2/olslam/sorghum_dataset/row_2066_2116/stereo_tmp_seed/rect0_fullres";
static string data_folder = "/home/remote_user2/olslam/sorghum_dataset/row_2066_2116/stereo_tmp_seed/disparities" ;
static string csv_folder = "/home/remote_user2/olslam/sorghum_dataset/row_2066_2116/final_op_rows_2066_2116_left";
static string csv_folder_right = "/home/remote_user2/olslam/sorghum_dataset/row_2066_2116/final_op_rows_2066_2116_right";
static float ellipse_resize_factor = 0.15;

#define PI   3.1415926535897932384626433832795

// FUNCTION HEADERS


vector<vector<float>> read_csv(std::string);
void printKeypointMapper(map<int, map<int, Point2f>>);
void printKeypointIndexer(map<string, int>);
string getKpKey(Point2f);
string getKpKey3(Point3f);
PointMatcher<float>::DataPoints create_datapoints(Mat);
void ShowBlackCircle( const cv::Mat&, cv::Point, int, Scalar);
vector<vector<float>> get_points(int);
vector<vector<float>> get_3d_bounds(int, Mat);
bool is_in_ellipse(float, float, float , float, float, float, float);
int NumDigits(int);
bool check_bound(float, float, vector<vector<float>> );
bool check_bound2(float, float, vector<vector<float>>);


// TEMPLATED FUNCTIONS
template <typename T>
Mat matrix_transform(Mat M, Mat R, Mat Tr){
    Mat ret(M.size(), M.type());
    Scalar translation( Tr.at<T>(0), Tr.at<T>(1), Tr.at<T>(2));
    for(int i=0; i<M.rows; i++){
        Mat tmp = (((M.row(i)).reshape(1,3)).t())*R;
        ret.at<T>(i, 0) = tmp.at<T>(0) + Tr.at<T>(0);
        ret.at<T>(i, 1) = tmp.at<T>(1) + Tr.at<T>(1); 
        ret.at<T>(i, 2) = tmp.at<T>(2) + Tr.at<T>(2);
    }
    return ret;
}

template <typename T>
void save_vtk(Mat pointcloud, string path) {
    ostringstream os; 
    os << "x,y,z\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<T>(i, 0) << "," << pointcloud.at<T>(i, 1) << "," << pointcloud.at<T>(i, 2) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<T>::loadCSV(in_stream);
    const typename PointMatcher<T>::DataPoints PC(points);
    PC.save(path);
}

#endif 
