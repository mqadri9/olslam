#include "utils.h"
#include "pose.h"
#include "optimizer.h"
typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;    
PM::ICP icp;
#include "Hungarian.h"
#include "test_sfm.h"
#include "hungarian.h"
#include "detector.h"
struct retPointcloud {
    Mat pc;
    vector<Point2f> src;
    vector<Point2f> dst;
};

struct corr {
    vector<Point2f> src2D;
    vector<Point2f> dst2D;
};

struct Frame {
    // This is the id of the frame
    int id;
    
    // these are the stereo correspondences from the stereo image pair 
    vector<vector<float>> stereo_correspondences;

    // These are the temporal_correspondences between this frame and the previous frame
    vector<vector<float>> temporal_correspondences;

    // this contains the pointcloud generated from the stereo pair images
    Mat pc;
    
    // src contains the 2D seeds location in the previous frame (3rd dimension is disparity)
    vector<Point3f> dst3Dstereo;

    vector<Point3f> dst3Dtemporal;
    
    vector<Point3f> src3Dtemporal;

    vector<Point2f> src2D;

    vector<Point2f> dst2D;
    // Left stereo image
    Mat img_l;

    // right stereo image
    Mat img_r;
};

r get_points_2D(string filename) {
    vector<vector<float>>csv = read_csv(filename);
    vector<Point2f> centers;  
    vector<vector<float>> bboxes;  
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][8]/RESIZE_FACTOR;
        float y1 = csv[j][9]/RESIZE_FACTOR;
        float x2 = csv[j][10]/RESIZE_FACTOR;
        float y2 = csv[j][11]/RESIZE_FACTOR;
        float x = csv[j][3] + x1;
        float y = csv[j][4] + y1;
        if (abs(x1 - x2) * abs(y1 - y2) < 15*15) continue;
        bboxes.push_back({x1, y1, x2, y2});
        Point2f center(x, y);
        centers.push_back(center);
    }
    r ret;
    ret.centers = centers;
    ret.bboxes = bboxes;
    return ret;
}


Mat create_pc(vector<vector<float>>* correspondences) {
    vector<float> rawcloud;
    for(int i=0; i<(*correspondences).size(); i++) {
        float x1 = (*correspondences)[i][0];
        float y1 = (*correspondences)[i][1];
        float x2 = (*correspondences)[i][2];
        float y2 = (*correspondences)[i][3];
        float d = x1 - x2;
        if (d < 0) continue;
        float Z = baseline*focal_length/d; 
        float X = (x1-cx)*Z/focal_length;
        float Y = (y1-cy)*Z/focal_length;
        rawcloud.push_back(X);
        rawcloud.push_back(Y);
        rawcloud.push_back(Z);
    }
    Mat m = Mat(rawcloud.size()/3, 1, CV_32FC3);
    memcpy(m.data, rawcloud.data(), rawcloud.size()*sizeof(float)); 
    return m;
}

corr get_2D_temporal_corr(vector<vector<float>>* correspondences) {
    vector<Point2f> src2D;
    vector<Point2f> dst2D;
    for(int i=0; i<(*correspondences).size(); i++) {
        float x1 = (*correspondences)[i][0];
        float y1 = (*correspondences)[i][1];
        float x2 = (*correspondences)[i][2];
        float y2 = (*correspondences)[i][3];
        if(abs(y2-y1) > 60) continue;
        Point2f src_tmp(x1, y1);
        Point2f dst_tmp(x2, y2);
        src2D.push_back(src_tmp);
        dst2D.push_back(dst_tmp);
    }
    corr ret;
    ret.src2D = src2D;
    ret.dst2D = dst2D;
    return ret;
}




void populate_final_matches(Frame prev_frame, Frame* curr_frame) {
    /// This function takes the 2D correspondences between the two consecutive temporal frames
    // and augment each 2D match with its disparity value
    vector<Point2f> src2D = (*curr_frame).src2D; 
    vector<Point2f> dst2D = (*curr_frame).dst2D;
    
    vector<Point3f> dst3Dtemporal;
    vector<Point3f> src3Dtemporal;

    vector<Point3f> dst3Dstereo_curr = (*curr_frame).dst3Dstereo;
    vector<Point3f> dst3Dstereo_prev = prev_frame.dst3Dstereo;
    for(int i=0; i< dst2D.size(); i++) {
        bool found1 = false;
        bool found2 = false;
        Point2f match1 = src2D[i];
        Point2f match2 = dst2D[i];
        Point3f tmp_match1;
        Point3f tmp_match2;
        for(int j=0; j< dst3Dstereo_curr.size(); j++) {
            float xt = match2.x;
            float yt = match2.y;
            float xtmp = dst3Dstereo_curr[j].x;
            float ytmp = dst3Dstereo_curr[j].y;
            if(xt == xtmp && yt == ytmp) {
                float d = dst3Dstereo_curr[j].z;
                tmp_match2.x = xt;
                tmp_match2.y = yt;
                tmp_match2.z = d;
                found1 = true; 
            }
        }
        if(found1 == false) continue;
        for(int j=0; j< dst3Dstereo_prev.size(); j++) {
            float xt = match1.x;
            float yt = match1.y;
            float xtmp = dst3Dstereo_prev[j].x;
            float ytmp = dst3Dstereo_prev[j].y;
            if(xt == xtmp && yt == ytmp) {
                float d = dst3Dstereo_prev[j].z;
                tmp_match1.x = xt;
                tmp_match1.y = yt;
                tmp_match1.z = d;
                found2 = true;
            }
        }
        if(found2) {
            dst3Dtemporal.push_back(tmp_match2);
            src3Dtemporal.push_back(tmp_match1);
        }
    }
    (*curr_frame).dst3Dtemporal = dst3Dtemporal;
    (*curr_frame).src3Dtemporal = src3Dtemporal;
}


RNG rng(12345);


int main(int argc, char* argv[]) {
    r points_prev;
    r points_curr;
    vector<retPointcloud> filteredOutput;
    vector<int> considered_poses;
    vector<Frame> frames; 
    vector<string> framesImages;
    int vertical_threshold_temporal = 250;
    int vertical_threshold_stereo = 20;
    int horizontal_threshold_stereo = 200;
    int horizontal_threshold_temporal = 300;
    int cost_threshold_stereo = 200;
    int cost_threshold_temporal = 200;
    vector<vector<Point3f>> each_frames_centers;
    int pose_id = 1;
    vector<Pose3> poses;
    vector<Pose3> poses_between;
    vector<Pose3> poses_tmp;
    map<int, map<int, Point3f>> KeypointMapper;
    map<string,int> prevKeypointIndexer;
    map<string,int> currKeypointIndexer;
    std::map<int,vector<Point3f>>::iterator it;
    std::map<string,int>::iterator itr;
    Rot3 R(1, 0, 0, 0, 1, 0, 0, 0, 1);
    Point3 t;
    t(0) = 0;
    t(1) = 0;
    t(2) = 0;
    Pose3 pose(R, t);
    Pose3 I(R,t);
    poses.push_back(pose);
    poses_tmp.push_back(pose);
    NonlinearFactorGraph graph;
    map<int,int> landmark_id_to_graph_id;
    Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    ret_optimize ret_optimizer;
    string detector = "combined";
    for(int i=0; i<100 ; i++) {
        framesImages.push_back(to_string(i));
        cout << "Processing a new image index:" << i << endl;
        Frame frame;
        frame.id = i;
        string filename_l =  csv_folder + "/res_" + to_string(i) + ".csv";
        string filename_r =  csv_folder_right + "/res_" + to_string(i) + ".csv";
        
        string img_path_l =  image_folder + "/frame" + to_string(i) + ".jpg";
        string img_path_r =  image_folder_right + "/frame" + to_string(i) + ".jpg"; 
        
        Mat img_l = imread( img_path_l);
        Mat img_r = imread( img_path_r);
        
        frame.img_l = img_l;
        frame.img_r = img_r;

        r left = get_points_2D(filename_l);
        r right = get_points_2D(filename_r);

        int img_width = img_l.size().width;
        vector<vector<float>> stereo_correspondences;
        if(detector == "surf") {
            stereo_correspondences = find_correspondences_surf(left, 
                                                                right, 
                                                                img_width, 
                                                                vertical_threshold_stereo, 
                                                                horizontal_threshold_stereo, 
                                                                cost_threshold_stereo,
                                                                img_l,
                                                                img_r,
                                                                i,
                                                                "stereo");


        }
        else if (detector == "sift") {
            stereo_correspondences = find_correspondences_sift(left, 
                                                                right, 
                                                                img_width, 
                                                                vertical_threshold_stereo, 
                                                                horizontal_threshold_stereo, 
                                                                cost_threshold_stereo,
                                                                img_l,
                                                                img_r,
                                                                i,
                                                                "stereo");
        }
        else if(detector=="akaze") {
            stereo_correspondences = find_correspondences_akaze(left, 
                                                                right, 
                                                                img_width, 
                                                                vertical_threshold_stereo, 
                                                                horizontal_threshold_stereo, 
                                                                cost_threshold_stereo,
                                                                img_l,
                                                                img_r,
                                                                i,
                                                                "stereo");
        }
        else if(detector=="orb") {
            stereo_correspondences = find_correspondences_orb(left, 
                                                                right, 
                                                                img_width, 
                                                                vertical_threshold_stereo, 
                                                                horizontal_threshold_stereo, 
                                                                cost_threshold_stereo,
                                                                img_l,
                                                                img_r,
                                                                i,
                                                                "stereo");
        }
        else if(detector == "hungarian") {
            stereo_correspondences = run_hungarian(left, 
                                                        right, 
                                                        img_width, 
                                                        vertical_threshold_stereo, 
                                                        horizontal_threshold_stereo, 
                                                        cost_threshold_stereo,
                                                        img_l,
                                                        img_r,
                                                        i,
                                                        "stereo");
        }
        else if(detector == "combined") {
            stereo_correspondences = run_hungarian(left, 
                                                        right, 
                                                        img_width, 
                                                        vertical_threshold_stereo, 
                                                        horizontal_threshold_stereo, 
                                                        cost_threshold_stereo,
                                                        img_l,
                                                        img_r,
                                                        i,
                                                        "stereo");

            vector<vector<float>> sift_stereo_correspondences = find_correspondences_sift(left, 
                                                                right, 
                                                                img_width, 
                                                                vertical_threshold_stereo, 
                                                                horizontal_threshold_stereo, 
                                                                cost_threshold_stereo,
                                                                img_l,
                                                                img_r,
                                                                i,
                                                                "stereo");

            for (int jk =0; jk<sift_stereo_correspondences.size(); jk++) {
                stereo_correspondences.push_back(sift_stereo_correspondences[jk]);
            }

        }


        Mat pc = create_pc(&stereo_correspondences);
        frame.pc = pc;

        for(int l=0; l< stereo_correspondences.size(); l++ ) {
            float x = stereo_correspondences[l][0];
            float y = stereo_correspondences[l][1];
            float d = x - stereo_correspondences[l][2];
            frame.dst3Dstereo.push_back({x, y, d});
        }        

        write_correspondences(img_l, img_r, i, stereo_correspondences);

        if (i==0) {
            frames.push_back(frame);
            continue;
        }

        string filename_prev =  csv_folder + "/res_" + to_string(i-1) + ".csv";
        r prev = get_points_2D(filename_prev);

        Mat img_prev = frames[i-1].img_l;
        vector<vector<float>> correspondences_temporal;
        if (detector == "surf") {
            correspondences_temporal = find_correspondences_surf(prev, 
                                                                left, 
                                                                img_width, 
                                                                vertical_threshold_temporal, 
                                                                horizontal_threshold_temporal, 
                                                                cost_threshold_temporal,
                                                                img_prev,
                                                                frame.img_l,
                                                                i,
                                                                "temporal"
                                                                );
        }
        else if (detector == "sift") {
            correspondences_temporal = find_correspondences_sift(prev, 
                                                                left, 
                                                                img_width, 
                                                                vertical_threshold_temporal, 
                                                                horizontal_threshold_temporal, 
                                                                cost_threshold_temporal,
                                                                img_prev,
                                                                frame.img_l,
                                                                i,
                                                                "temporal"
                                                                );
        }
        else if(detector=="akaze") {
            correspondences_temporal = find_correspondences_akaze(prev, 
                                                                left, 
                                                                img_width, 
                                                                vertical_threshold_temporal, 
                                                                horizontal_threshold_temporal, 
                                                                cost_threshold_temporal,
                                                                img_prev,
                                                                frame.img_l,
                                                                i,
                                                                "temporal"
                                                                );
        }
        else if(detector=="orb") {
            correspondences_temporal = find_correspondences_orb(prev, 
                                                                left, 
                                                                img_width, 
                                                                vertical_threshold_temporal, 
                                                                horizontal_threshold_temporal, 
                                                                cost_threshold_temporal,
                                                                img_prev,
                                                                frame.img_l,
                                                                i,
                                                                "temporal"
                                                                );
        }
        else if(detector == "hungarian") {
            correspondences_temporal = run_hungarian(prev, 
                                                    left, 
                                                    img_width, 
                                                    vertical_threshold_temporal, 
                                                    horizontal_threshold_temporal, 
                                                    cost_threshold_temporal,
                                                    img_prev,
                                                    frame.img_l,
                                                    i,
                                                    "temporal"
                                                    );
        }
        else if(detector == "combined") {
            correspondences_temporal = run_hungarian(prev, 
                                                    left, 
                                                    img_width, 
                                                    vertical_threshold_temporal, 
                                                    horizontal_threshold_temporal, 
                                                    cost_threshold_temporal,
                                                    img_prev,
                                                    frame.img_l,
                                                    i,
                                                    "temporal"
                                                    );
            vector<vector<float>> sift_correspondences_temporal = find_correspondences_sift(prev, 
                                                            left, 
                                                            img_width, 
                                                            vertical_threshold_temporal, 
                                                            horizontal_threshold_temporal, 
                                                            cost_threshold_temporal,
                                                            img_prev,
                                                            frame.img_l,
                                                            i,
                                                            "temporal"
                                                            );
            for (int jk =0; jk<sift_correspondences_temporal.size(); jk++) {
                correspondences_temporal.push_back(sift_correspondences_temporal[jk]);
            }
        } 

        write_correspondences_temporal(img_prev, img_l, i, correspondences_temporal);
        frame.temporal_correspondences = correspondences_temporal; 

        corr ret = get_2D_temporal_corr(&correspondences_temporal);
        frame.src2D = ret.src2D;
        frame.dst2D = ret.dst2D;
        populate_final_matches(frames[i-1], &frame);

        /*
        cout << frame.pc.size() << endl;
        cout << frame.src2D.size() << endl;
        cout << frame.dst2D.size() << endl;
        cout << frame.src3Dtemporal.size() << endl;
        cout << frame.dst3Dtemporal.size() << endl;
        */

        // If it is the second image then add the matches from image 0 as well
        if(i == 1) {
            each_frames_centers.push_back(frame.src3Dtemporal);
        }
        each_frames_centers.push_back(frame.dst3Dtemporal);
        frames.push_back(frame);
        if(i>=1) {
            considered_poses.push_back(i);
            Mat m1 = frames[i-1].pc;
            Mat m2 = frames[i].pc;
            vector<Point3f> src = frames[i].src3Dtemporal;
            vector<Point3f> dst = frames[i].dst3Dtemporal;
            cout << m1.size() << endl;
            cout << m2.size() << endl;
            retPose p = getPose(m1, m2, "icp"); 
            //Rot3 R(1, 0, 0, 0, 1, 0, 0, 0, 1);
            //Point3 t;
            //t(0) = p.t(0);
            //t(1) = 0;
            //t(2) = 0;
            //Pose3 pose(R, t);
            Pose3 pose(p.R, p.t);
            cout << "POSE OUTPUT " << pose << endl;
            poses_tmp.push_back(poses.back()*pose);
            poses_between.push_back(pose);
            // Need to construct the landmarks array
            // Initialize and create gtsam graph here
            // i is the image index
            cout << "SIZE OF POSES " << poses_tmp.size() << endl;
            if(poses_tmp.size() == 2) {
                //cout << "Initializing prevKeypointIndexer" << endl;
                for(int l =0; l < dst.size(); l++) {
                    // assign incremental landmark IDs for the first two images
                    //cout << l << " | " << src[l].x << " | " << dst[l].x << " | " << src[l].y << " | " << dst[l].y << endl;
                    KeypointMapper[l].insert(make_pair(pose_id-1, src[l]));
                    KeypointMapper[l].insert(make_pair(pose_id, dst[l]));
                    prevKeypointIndexer[getKpKey3(dst[l])] = l;
                }
                
                ret_optimizer = Optimize_from_stereo(KeypointMapper, Kgt, framesImages, considered_poses, poses_tmp, poses_between, &landmark_id_to_graph_id);         
                graph = ret_optimizer.graph;
                gtsam::Values result = ret_optimizer.result;
                int init_pose = 0; 
                poses.clear();
                while(result.exists(Symbol('x', init_pose))) {
                    Pose3 P = result.at(Symbol('x', init_pose).key()).cast<Pose3>();
                    poses.push_back(P);
                    init_pose++;
                }
                poses_tmp = poses;
                pose_id++;
                continue;
            }
            /* For each keypoint in the new image
            Check if the match at image i-1 already exists (data associated)
            If it does, get the of the corresponding landmark_id and populate KeypointMapper[landmark_id]
            If it does not, assign a new landmark_id and populate KeypointMapper[landmark_id]
            */
        
            if(poses_tmp.size() > 2) {
                cout << "=========================Handling pose======================= " << pose_id << endl;
                for(int l =0; l < dst.size(); l++) {
                    itr = prevKeypointIndexer.find(getKpKey3(src[l]));
                    int landmark_id;
                    if ( itr != prevKeypointIndexer.end() ) {
                        landmark_id = itr->second;
                    }
                    else{
                        int largest_landmark_id = KeypointMapper.rbegin()->first;
                        landmark_id = largest_landmark_id + 1;
                    }
                    KeypointMapper[landmark_id].insert(make_pair(pose_id, dst[l]));
                    currKeypointIndexer[getKpKey3(dst[l])] = landmark_id;
                }
                        
                prevKeypointIndexer.clear();
                prevKeypointIndexer = currKeypointIndexer;
                ret_optimizer = Reoptimize_from_stereo(KeypointMapper, 
                                                        Kgt, 
                                                        framesImages, 
                                                        considered_poses,
                                                        poses_tmp,
                                                        poses_between, 
                                                        &landmark_id_to_graph_id,
                                                        pose_id,
                                                        ret_optimizer);
                gtsam::Values result = ret_optimizer.result;
                int init_pose = 0; 
                poses.clear();
                while(result.exists(Symbol('x', init_pose))) {
                    Pose3 P = result.at(Symbol('x', init_pose).key()).cast<Pose3>();
                    poses.push_back(P);
                    init_pose++;
                }
                poses_tmp = poses;
            }
            pose_id++;
        }
    }

    //cout << "===================KEYPOINTMAPPER====================" << endl;
    //cout << KeypointMapper.size() << endl;
    //gtsam::Values result;
    
    //ret_optimize ret_optimizer = Optimize_from_stereo(KeypointMapper, Kgt, framesImages, considered_poses, poses);
    //result = ret_optimizer.result;

    // Test GTSAM output  
    //test_sfm(result, Kgt, considered_poses);
    //reconstruct_pointcloud(result, Kgt, considered_poses, framesImages);
    //Cal3_S2::shared_ptr Kgt_res(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    //cout << "SIZE OF EACH_FRAMES_CENTERS " << each_frames_centers.size() << " | " <<  each_frames_centers[0].size() << " | " <<  each_frames_centers[1].size() << endl;
    //gtsam::Values result_reoptimize = Optimize_object_loc2(ret_optimizer, considered_poses, framesImages, Kgt_res, each_frames_centers);
    return 0;


}
