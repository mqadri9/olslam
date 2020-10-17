#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "utils.h"

struct ret_optimize{
    gtsam::Values result;
    NonlinearFactorGraph graph;
    vector<Point3> landmarks3d;
};

struct camPoint {
    Point3 x3D;
    Point3 xyd;
    int frame_id;
};

struct semantic_objects_3d{
    Point3 x3Dw;
    vector<camPoint> cam_frame_3d_points;
    int index;
};

struct worldpoint {
    Point3 x3Dw;
    int index;
    bool matched=false;
};


ret_optimize Optimize(map<int, map<int, Point2f>>, Cal3_S2::shared_ptr, vector<string>, vector<int>, vector<Pose3>);
ret_optimize Optimize_from_stereo(map<int, map<int, Point3f>>, Cal3_S2::shared_ptr, vector<string>, vector<int>, vector<Pose3>);
vector<vector<float>> get_points(int, Mat);
gtsam::Values Optimize_object_loc(ret_optimize, vector<int>, vector<string> frames, Cal3_S2::shared_ptr);
void plot_projected_matches(vector<int>, vector<string>, std::map<int, semantic_objects_3d>, gtsam::Values, Cal3_S2::shared_ptr, int frame_id);
gtsam::Values Optimize_object_loc2(ret_optimize ret_optimizer, 
                                   vector<int> considered_poses, 
                                   vector<string> frames, 
                                   Cal3_S2::shared_ptr Kgt,
                                   vector<vector<Point3f>> each_frames_centers);

#endif