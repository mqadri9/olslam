#include "optimizer.h"

#include "SFMdata.h"

ret_optimize Optimize(map<int, map<int, Point2f>> KeypointMapper, 
                       Cal3_S2::shared_ptr Kgt, 
                       vector<string> frames, 
                       vector<int> considered_poses, 
                       vector<Pose3> poses) {
    // Define the camera observation noise model
    auto noise = noiseModel::Isotropic::Sigma(2, 2.0);  // 2 pixel in u and v
    const Cal3_S2Stereo::shared_ptr Kstereo(
      new Cal3_S2Stereo(focal_length, fy, 0, cx, cy, baseline));
    const auto model = noiseModel::Isotropic::Sigma(3, 1);    
    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    NonlinearFactorGraph graph_cp;
    auto poseNoise = noiseModel::Diagonal::Sigmas(
        (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
            .finished());  // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    graph.addPrior(Symbol('x', 0), poses[0], poseNoise);  // add directly to graph
    int N = poses.size();
    map<int, map<int, Point2f> > FilteredKeypointMapper;
    map<int, map<int, Point3f> > landmarks;
    map<int, map<int, Point2f> >::iterator it;

    std::cout << "Filtering KeypointMapper to keep points that appear in N images" << std::endl;
    //printKeypointMapper(KeypointMapper);
    //cout << "Created FilteredKeypointMapper" << endl;
    
    // Create the data structure to hold the initial estimate to the solution
    Values initialEstimate;
    for (size_t i = 0; i < poses.size(); ++i) {
        initialEstimate.insert(
            Symbol('x', i), poses[i]);
    }
    vector<Point3> landmarks3d;
    int landmark_id_in_graph = 0;
    for ( it=KeypointMapper.begin() ; it != KeypointMapper.end(); it++ ) {
        //int frame_id = considered_poses[i];
        map<int, Point2f>::iterator itr;
        int landmark_id = it->first;
        map<int, Point2f> landmark_to_poses = it->second;
        //cout << "landmark_to_poses.size() " << landmark_to_poses.size() << endl;
        //cout << "N " << N << endl;
        if(landmark_to_poses.size() < N) {
            continue;
        }
        Point3 landmark3d;
        double x0, y0, d0;
        int processed_first_pose = false;
        bool skipped_landmark_at_pose = false;
        vector<GenericProjectionFactor<Pose3, Point3, Cal3_S2>> projectionFactors;
        
        for (itr=landmark_to_poses.begin(); itr != landmark_to_poses.end(); itr++ ) {
            int pose_id = itr->first;
            Point2f measurement_cv2 = itr->second;
            Point2 measurement;
            measurement(0) = measurement_cv2.x;
            measurement(1) = measurement_cv2.y;
            string disparity_path = data_folder + "/frame" +  to_string(pose_id) + ".jpg";;
            cout << disparity_path << endl;
            Mat disparity = imread( disparity_path , 0);
            double x = measurement(0);
            double y = measurement(1);
            double d = disparity.at<uchar>((int)y, (int)x);
            double uR = x - d;
            if (d < 20 || uR < 0) {
                cout << "Skipping this iteration d=" << d << "uR= " << uR <<  " x=" << x << " frame " << to_string(pose_id) + ".jpg" << std::endl;
                skipped_landmark_at_pose = true;
                break;
            }
            if(!processed_first_pose) {
                x0 = x;
                y0 = y;
                d0 = d;
                processed_first_pose = true;
            }
            //cout << uL << " | " << uR << " | " << d << endl;
            //graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(
            //   StereoPoint2(uL, uR, v), model, Symbol('x', i), Symbol('l', j), Kstereo);
            //std::cout << "Adding factor " << std::endl;
            GenericProjectionFactor<Pose3, Point3, Cal3_S2> factor(measurement, noise, Symbol('x', pose_id), Symbol('l', landmark_id_in_graph), Kgt);
            projectionFactors.push_back(factor);
            //graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
            //    measurement, noise, Symbol('x', pose_id), Symbol('l', landmark_id), Kgt);  
        }
        if(skipped_landmark_at_pose) {
            continue;
        }
        for(int pr=0; pr < projectionFactors.size(); pr++) {
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(projectionFactors[pr]);            
        }
        float Z = baseline*focal_length/d0;
        float X = (x0-cx)*Z/focal_length;
        float Y = (y0-cy)*Z/focal_length;
        landmark3d(0) = X;
        landmark3d(1) = Y;
        landmark3d(2) = Z;
        landmarks3d.push_back(landmark3d);
        initialEstimate.insert<Point3>(Symbol('l', landmark_id_in_graph), landmark3d);
        // Because the structure-from-motion problem has a scale ambiguity, the
        // problem is still under-constrained Here we add a prior on the position of
        // the first landmark. This fixes the scale by indicating the distance between
        // the first camera and the first landmark. All other landmark positions are
        // interpreted using this scale.
        auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
        if(landmark_id_in_graph == 0) {
            graph.addPrior(Symbol('l', 0), landmark3d,
                        pointNoise);  // add directly to graph
        }

        // TESTING CONDITION NEEDS TO BE REMOVED
        if(landmark_id_in_graph > 2 ) {
           break;
       }
        landmark_id_in_graph++;

    }
    
    cout << "populated poses" << endl;
    cout << poses.size() << endl;
    for(int i=0; i< poses.size(); ++i)
    {
      std::cout << poses[i] << ' ' << endl;
    }
    
    graph.print("Factor Graph:\n");
    initialEstimate.print("initial");

    gtsam::Values result;
    DoglegParams params;
    graph_cp = graph.clone();
    cout << "initial error = " << graph.error(initialEstimate) << endl;
    result = DoglegOptimizer(graph, initialEstimate, params).optimize();
    //result.print("Final results:\n");
    
    cout << "final error = " << graph.error(result) << endl;
    ret_optimize ret_optimizer;
    ret_optimizer.graph = graph_cp;
    ret_optimizer.result = result;
    ret_optimizer.landmarks3d = landmarks3d;
    return ret_optimizer;

}   

vector<worldpoint> extract_world_points(std::map<int, semantic_objects_3d> world_3d_objects){
    vector<worldpoint> w_points_3d;
    map<int, semantic_objects_3d>::iterator it;
    for ( it = world_3d_objects.begin(); it != world_3d_objects.end(); it++ )
    {
        worldpoint tmp;
        tmp.x3Dw = (it->second).x3Dw;
        tmp.index = it->first;
        w_points_3d.push_back(tmp);
    }
    return w_points_3d;
}

double xydist(Point3 p1, Point3 p2) {
    float x1 = p1(0);
    float y1 = p1(1);
    float x2 = p2(0);
    float y2 = p2(1);
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}

gtsam::Values Optimize_object_loc(ret_optimize ret_optimizer, vector<int> considered_poses, Cal3_S2::shared_ptr Kgt) {
    NonlinearFactorGraph graph = ret_optimizer.graph;
    gtsam::Values result = ret_optimizer.result;
    vector<Point3> landmarks3d = ret_optimizer.landmarks3d;   
    cv::Mat Q;
    float data[16] = {1.0, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, focal_length, 0, 0, 1.0/baseline, (cx-cxprime)/baseline};
    Q = Mat(4, 4, CV_32FC1, &data);
    // Each semantic_objects_3d element contains the 3D coordinates of one of the 3D projected pixels 
    // of the semantically segmented fruitlets. It also contains a list mapping each 3D world point to its 
    // corresponding point in each camera frame
    std::map<int, semantic_objects_3d> world_3d_objects;
    for(int i=0; i<considered_poses.size(); i++) {
        //if(i>5) {
        //    break;
        //}
        vector<worldpoint> w_points_3d;
        string disparity_path = data_folder + "/frame"+  to_string(considered_poses[i]) + ".jpg";
        string img_path_target = image_folder + "/frame" + to_string(considered_poses[i]) + ".jpg";
        //cout << "IMAGES FILE " << img_path_target << endl;
        //cout << "disparity FILE " << img_path << endl;
        Mat disparity = imread( disparity_path , 0);
        Mat img = imread( img_path_target);
        cv::Mat reprojection(disparity.size(),CV_32FC3);
        cv::reprojectImageTo3D(disparity, reprojection, Q);
        Pose3 P = result.at(Symbol('x', i).key()).cast<Pose3>();
        //cout << P << endl;
        vector<vector<float>> points = get_points(considered_poses[i], disparity);
        w_points_3d = extract_world_points(world_3d_objects);
        cout << w_points_3d.size() << endl;
        int not_matched = 0;
        for (int ii=0; ii < points.size(); ii++) {
            int x = int(points[ii][0]);
            int y = int(points[ii][1]);
            Vec3f p = reprojection.at<Vec3f>(y,x);
            Point3 x3D; 
            x3D(0) = p[0];
            x3D(1) = p[1];
            x3D(2) = p[2];     
            if(p[2] < baseline*15) {
                Point3 x3Dw = P.transform_from(x3D);
                semantic_objects_3d tmp;
                if(i == 0) {
                    camPoint c;
                    tmp.x3Dw = x3Dw;
                    c.x3D = x3D;
                    c.frame_id = i;
                    tmp.cam_frame_3d_points.push_back(c);
                    tmp.index = ii;
                    world_3d_objects.insert(std::make_pair(ii, tmp));
                    continue;
                }
                double d = 999;
                int index_min = -1;
                for(int j = 0; j < w_points_3d.size(); j++) {
                    double dt = xydist(x3Dw, w_points_3d[j].x3Dw);
                    // Prevent 2 points to be associated with the same 3D world point                    
                    if(dt < d  && !w_points_3d[j].matched && dt <0.005) {
                        d = dt;
                        index_min = j;
                    }
                }
                if(index_min != -1) {
                    w_points_3d[index_min].matched = true;
                    camPoint c;
                    c.frame_id = i;
                    c.x3D = x3D;
                    world_3d_objects.at(index_min).cam_frame_3d_points.push_back(c);                    
                }
                else {
                    not_matched++;
                }  
            }
        }
        //plot_projected_matches(considered_poses, world_3d_objects, result, Kgt, i);
    }

    //update initial pose guess with the output of sift slam

    Values initialEstimate;
    for (size_t i = 0; i < considered_poses.size(); ++i) {
        initialEstimate.insert(
            Symbol('x', i), result.at(Symbol('x', i).key()).cast<Pose3>());
    }
    int l_id = 0;
    for (l_id; l_id < landmarks3d.size(); ++l_id) {
        initialEstimate.insert<Point3>(Symbol('l', l_id), landmarks3d[l_id]);
    }
    cout << "Without fruitlet landmarks initial error = " << graph.error(initialEstimate) << endl;
    auto noise = noiseModel::Isotropic::Sigma(2, 2.0);  // 5 pixel in u and v
    Pose3 Pn = Pose3();

    const Cal3_S2Stereo::shared_ptr Kstereo(
      new Cal3_S2Stereo(focal_length, fy, 0, cx, cy, baseline));

    // world_3d_objects is a map where the key is the index 
    // of a landmark (projected fruitlet pixel to 3D) and the value 
    // is semantic_objects_3d struct
    const auto model = noiseModel::Isotropic::Sigma(3, 5);
    map<int, semantic_objects_3d>::iterator it;
    for ( it = world_3d_objects.begin(); it != world_3d_objects.end(); it++ )
    {
        int id = it->first;
        Point3 x3Dw = (it->second).x3Dw;
        vector<camPoint> cam_frame_3d_points = (it->second).cam_frame_3d_points;
        if (cam_frame_3d_points.size() < 3) {
            continue;
        }
        initialEstimate.insert<Point3>(Symbol('l', l_id + id), x3Dw);
        for (int ep= 0; ep < cam_frame_3d_points.size(); ep++) {
            Point3 x3d_t = cam_frame_3d_points[ep].x3D;
            int frame_id = cam_frame_3d_points[ep].frame_id;
            string disparity_path = data_folder + "/frame"+  to_string(considered_poses[frame_id]) + ".jpg";
            Mat disparity = imread( disparity_path , 0);
            PinholeCamera<Cal3_S2> camera_n(Pn, *Kgt);
            Point2 target = camera_n.project(x3d_t);
            int uL = (int)target(0);
            int v = (int)target(1);
            int d = disparity.at<uchar>(v, uL);
            int uR = uL - d;
            
            graph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(
               StereoPoint2(uL, uR, v), model, Symbol('x', frame_id), Symbol('l', l_id + id), Kstereo);

            //graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
            //    target, noise, Symbol('x', frame_id), Symbol('l', l_id + id), Kgt);
        }
    }
    gtsam::Values result_reoptimize;
    DoglegParams params;
    //initialEstimate.print();
    //graph.print();
    result_reoptimize = DoglegOptimizer(graph, initialEstimate, params).optimize();
    //result.print("Final results:\n");
    cout << "initial error = " << graph.error(initialEstimate) << endl;
    cout << "final error = " << graph.error(result_reoptimize) << endl;
    return result_reoptimize;
}

void plot_projected_matches(vector<int> considered_poses, std::map<int, semantic_objects_3d> world_3d_objects,
                            gtsam::Values result, Cal3_S2::shared_ptr Kgt, int frame_id)
{
    cv::Mat HM; 
    Mat imLeft =  imread(image_folder + "/frame" + to_string(considered_poses[0]) + ".jpg");
    Mat imRight =  imread(image_folder + "/frame" + to_string(considered_poses[frame_id]) + ".jpg");
    cout << image_folder + "/frame" + to_string(considered_poses[0]) + ".jpg" << endl;
    cout << image_folder + "/frame" + to_string(considered_poses[frame_id]) + ".jpg" << endl;
    hconcat(imLeft,imRight,HM); 
    map<int, semantic_objects_3d>::iterator it;
    for ( it = world_3d_objects.begin(); it != world_3d_objects.end(); it++ )
    {
        Pose3 P0 = result.at(Symbol('x', 0).key()).cast<Pose3>();
        //Pose3 Pn = result.at(Symbol('x', frame_id).key()).cast<Pose3>();
        Pose3 Pn = Pose3();   
        Point3 x3Dw = (it->second).x3Dw;
        PinholeCamera<Cal3_S2> camera0(P0, *Kgt);
        Point2 gt = camera0.project(x3Dw);
        vector<camPoint> cam_frame_3d_points = (it->second).cam_frame_3d_points;
        for (int ep= 0; ep < cam_frame_3d_points.size(); ep++) {
            if(cam_frame_3d_points[ep].frame_id == frame_id) {
                Point3 x3d_t = cam_frame_3d_points[ep].x3D;
                PinholeCamera<Cal3_S2> camera_n(Pn, *Kgt);
                Point2 target = camera_n.project(x3d_t);
                int thickness = 1;
                int lineType = cv::LINE_8;
                cv::Point2f start;
                cv::Point2f end;
                start.x = gt(0);
                start.y = gt(1);
                end.x = target(0) + imLeft.size().width;
                end.y = target(1);
                line( HM,
                    start,
                    end,
                    0xffff,
                    thickness,
                    lineType );
            }
        }
    }
        string p = "/home/remote_user2/olslam/pixel_matches/test_" + to_string(frame_id) +".jpg";
        cout << p << endl;
        cv::imwrite( p, HM);
}