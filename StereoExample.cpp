/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    VisualISAM2Example.cpp
 * @brief   A visualSLAM example for the structure-from-motion problem on a
 * simulated dataset This version uses iSAM2 to solve the problem incrementally
 * @author  Duy-Nguyen Ta
 */

/**
 * A structure-from-motion example with landmarks
 *  - The landmarks form a 10 meter cube
 *  - The robot rotates around the landmarks, always facing towards the cube
 */

// For loading the data
#include "utils.h"
#include "pose.h"
#include "optimizer.h"
typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;    
PM::ICP icp;

using namespace std;
using namespace gtsam;

/* ************************************************************************* */
int main(int argc, char* argv[]) {
  // Define the camera calibration parameters
  Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
  Values initialEstimate;
  auto noise = noiseModel::Isotropic::Sigma(2, 2.0);  // 5 pixel in u and v
  Rot3 R1(1, 0, 0, 0, 1, 0, 0, 0, 1);
  Point3 t1;
  t1(0) = 0;
  t1(1) = 0;
  t1(2) = 0;
  Pose3 pose1(R1, t1);

  Rot3 R2(0.993329346, -0.0510207377,  -0.103411965, 0.0284845456,   0.977565944,  -0.208695874 ,0.11173977,   0.204357997,   0.972497761);
  Point3 t2;
  t1(0) = 0.00351440907;
  t1(1) = 0.082824938;
  t1(2) = -0.0234023333;
  Pose3 pose2(R2, t2); 

  initialEstimate.insert(
              Symbol('x', 0), pose1);
  initialEstimate.insert(
              Symbol('x', 1), pose2);

  Point3 landmark3d;
  landmark3d(0) = 0.250332803;
  landmark3d(1) = -0.0757690147;
  landmark3d(2) = 0.426930547;

  const auto model = noiseModel::Isotropic::Sigma(3, 1);    
    // Create a Factor Graph and Values to hold the new data
  NonlinearFactorGraph graph;
  auto poseNoise = noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
          .finished());  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
  graph.addPrior(Symbol('x', 0), pose1, poseNoise);  // add directly to graph

  Point2 measurement;
  measurement(0) = 3814.79395;
  measurement(1) = 1047.32324;
  GenericProjectionFactor<Pose3, Point3, Cal3_S2> factor1(measurement, noise, Symbol('x', 0), Symbol('l', 0), Kgt);

  Point2 measurement2;
  measurement(0) = 3527.53467;
  measurement(1) = 1067.24304;
  GenericProjectionFactor<Pose3, Point3, Cal3_S2> factor2(measurement2, noise, Symbol('x', 1), Symbol('l', 0), Kgt);

  graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(factor1);    
  graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(factor2);    

  initialEstimate.insert<Point3>(Symbol('l', 0), landmark3d);
  auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
  graph.addPrior(Symbol('l', 0), landmark3d,
                        pointNoise);

  graph.print("GRAPH");
  initialEstimate.print("ESTIMATE");

  gtsam::Values result;
  DoglegParams params;
  cout << "Initial error = " << graph.error(initialEstimate) << endl;
  result = DoglegOptimizer(graph, initialEstimate, params).optimize();
  cout << "final error = " << graph.error(result) << endl;
  return 0;
}
/* ************************************************************************* */