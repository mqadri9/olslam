#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "utils.h"    

struct retPointcloud {
    vector<Mat> ret;
    vector<Point2f> src;
    vector<Point2f> dst;
};
    
struct retFiltering {
    vector<int> considered_poses;
    vector<retPointcloud> filteredOutput;
};
// Parameters 
static int MAX_PIXEL_ERR = 30;
static int  MAX_WINDOW_ERR = 200;
static bool sift_use_madnet = false;

retPointcloud createPointCloudsFromStereoPairs(std::map<string, std::tuple<Point2f, Point2f, Point3f>>, 
                                               std::map<string, std::tuple<Point2f, Point2f, Point3f>>,
                                               std::map<string, std::tuple<Point2f, Point2f, Point3f>>,
                                               string,
                                               string,
                                               int);
std::map<string, std::tuple<Point2f, Point2f, Point3f>> stereoKptsTo3D(std::vector< std::vector<DMatch> >, std::vector<KeyPoint>, std::vector<KeyPoint>, bool);
retFiltering filterImagesByMatching(vector<string>);
retPointcloud createPointClouds(Mat, Mat, std::vector<KeyPoint>,  std::vector<KeyPoint>, std::vector< std::vector<DMatch> >);
vector<Mat> createPointClouds(Mat, Mat);
retFiltering filterImages(vector<string>);
float avgDepth(Mat);

#endif