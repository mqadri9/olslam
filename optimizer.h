#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "utils.h"

struct ret_optimize{
    gtsam::Values result;
    NonlinearFactorGraph graph;
};

ret_optimize Optimize(map<int, map<int, Point2f>>, Cal3_S2::shared_ptr, vector<string>, vector<int>, vector<Pose3>);



#endif