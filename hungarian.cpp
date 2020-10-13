#include "utils.h"
#include "pose.h"
#include "optimizer.h"
typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;    
PM::ICP icp;
#include "Hungarian.h"
#include "test_sfm.h"

struct r {
    Mat m;
    vector<Point2f> centers;
    Mat pc;
    vector<vector<float>> stereo_correspondences;
    vector<vector<float>> bboxes;
};

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

void write_correspondences(Mat img_l, Mat img_r, int frame_id, vector<vector<float>> stereo_correspondences)
{
    cv::Mat HM; 
    hconcat(img_l,img_r,HM); 
    for(int j=0; j < stereo_correspondences.size(); j++) {
        vector<float> entry = stereo_correspondences[j];
        int thickness = 1;
        int lineType = cv::LINE_8;
        cv::Point2f start;
        cv::Point2f end;
        start.x = entry[0];
        start.y = entry[1];
        end.x = entry[2] + img_l.size().width;
        end.y = entry[3];
        if(abs(end.y - start.y) > 50) continue;
        line( HM,
            start,
            end,
            0xffff,
            thickness,
            lineType );
        //if (j > 50) {
        //    break;
        //}
    }
    imwrite("/home/remote_user2/olslam/hungarian_matches/frame" + to_string(frame_id) + ".jpg", HM); 
}

void write_correspondences_temporal(Mat img_l, Mat img_r, int frame_id, vector<vector<float>> stereo_correspondences)
{
    cv::Mat HM; 
    hconcat(img_l,img_r,HM); 
    for(int j=0; j < stereo_correspondences.size(); j++) {
        vector<float> entry = stereo_correspondences[j];
        int thickness = 1;
        int lineType = cv::LINE_8;
        cv::Point2f start;
        cv::Point2f end;
        start.x = entry[0];
        start.y = entry[1];
        end.x = entry[2] + img_l.size().width;
        end.y = entry[3];
        //cout << " | xs " << start.x << " | ys " << start.y << " | xe " << end.x << " | ye " << end.y << endl; 
        if(abs(end.y - start.y) > 50) continue;
        line( HM,
            start,
            end,
            0xffff,
            thickness,
            lineType );
        //if (j > 50) {
        //    break;
        //}
    }
    imwrite("/home/remote_user2/olslam/hungarian_temporal_matches/frame" + to_string(frame_id) + ".jpg", HM); 
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


void find_left_score(unordered_map<int, float>*left_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(x - x_tmp < 30 && x - x_tmp > 0 && abs(y_tmp - y) < 30 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy;            
            score += area;
        }
    }
    (*left_map)[k] = score;
}

void find_right_score(unordered_map<int, float>*right_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(x_tmp - x < 30 && x_tmp - x > 0 && abs(y_tmp - y) < 30 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy;            
            score += area;
        }
    }
    (*right_map)[k] = score;
}

vector<vector<float>> run_hungarian(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold) 
{
    vector<Point2f> centers_l = left.centers;
    vector<Point2f> centers_r = right.centers;
    vector<vector<float>> bboxes_l = left.bboxes;
    vector<vector<float>> bboxes_r = right.bboxes;

    unordered_map<int, Point2f> index_centers_l;
    unordered_map<int, Point2f> index_centers_r;

    unordered_map<int, float> left_to_left;
    unordered_map<int, float> left_to_right;
    unordered_map<int, float> right_to_left;
    unordered_map<int, float> right_to_right;

    vector<Point2f> centers_l_filtered;
    vector<Point2f> centers_r_filtered;
    vector<vector<float>> bboxes_l_filtered;
    vector<vector<float>> bboxes_r_filtered;

    // Filter out centers and bboxes;
    for(int ii=0; ii<centers_l.size(); ii++) {
        if(centers_l[ii].x > horizontal_threshold) {
            centers_l_filtered.push_back(centers_l[ii]);
            bboxes_l_filtered.push_back(bboxes_l[ii]);
        }
    }

    for(int ii=0; ii<centers_r.size(); ii++) {
        if(centers_r[ii].x < img_width - horizontal_threshold) {
            centers_r_filtered.push_back(centers_r[ii]);
            bboxes_r_filtered.push_back(bboxes_r[ii]);
        }
    }
    cout << "=============================" << endl;
    cout << centers_l_filtered.size() << endl;
    cout << bboxes_l_filtered.size() << endl;
    cout << centers_r_filtered.size() << endl;
    cout << bboxes_r_filtered.size() << endl;

    for(int k=0; k<centers_l_filtered.size(); k++) {
        index_centers_l[k] = centers_l_filtered[k];
        find_left_score(&left_to_left, &centers_l_filtered, &bboxes_l_filtered, centers_l_filtered[k], k);
        find_right_score(&left_to_right, &centers_l_filtered, &bboxes_l_filtered, centers_l_filtered[k], k);
        //cout << centers_l_filtered[k] << " | " << left_to_right[k] << endl;
    }

    for(int k=0; k<centers_r_filtered.size(); k++) {
        index_centers_r[k] = centers_r_filtered[k];
        find_left_score(&right_to_left, &centers_r_filtered, &bboxes_r_filtered, centers_r_filtered[k], k);
        find_right_score(&right_to_right, &centers_r_filtered, &bboxes_r_filtered, centers_r_filtered[k], k);
    }

    int ss = max(centers_l_filtered.size(), centers_r_filtered.size());
    std::vector<std::vector<double>> HungarianCost(ss, std::vector<double>(ss, 10e8));
    for(int k=0; k < centers_l_filtered.size(); k++)
    {
        vector<float> bb_left = left.bboxes[k];
        float dx_left = abs(bb_left[0] - bb_left[2]);
        float dy_left = abs(bb_left[1] - bb_left[3]);
        float area_left = dx_left*dy_left;
        for(int j =0 ; j < centers_r_filtered.size(); j++) {
            vector<float> bb_right = right.bboxes[j];
            float dx_right = abs(bb_right[0] - bb_right[2]);
            float dy_right = abs(bb_right[1] - bb_right[3]);
            float area_right = dx_right*dy_right;
            float vdist = abs(centers_l_filtered[k].y - centers_r_filtered[j].y);
            float xdist = centers_l_filtered[k].x - centers_r_filtered[j].x;
            //if(bb_left[0] < 500 || bb_right[0] > img_width - 500) {
            //    HungarianCost[k][j] = 10e8;
            //}
            if(xdist < 200) {
                HungarianCost[k][j] = 10e8;
            }
            else if(vdist > vertical_threshold) {
                HungarianCost[k][j] = 10e8;
            }
            else{
                HungarianCost[k][j] = vdist;//*abs(area_left - area_right)*100;
                HungarianCost[k][j] += max(left_to_left.at(k), right_to_left.at(j))/min(left_to_left.at(k), right_to_left.at(j))*10 ;
                HungarianCost[k][j] += max(left_to_right.at(k), right_to_right.at(j))/min(left_to_right.at(k), right_to_right.at(j))*10;
                //cout << max(left_to_left.at(k), right_to_left.at(j))/min(left_to_left.at(k), right_to_left.at(j)) << endl;
                //cout << max(left_to_right.at(k), right_to_right.at(j))/min(left_to_right.at(k), right_to_right.at(j)) << endl;
                //cout << "====================================" << endl;
            }
        }
        
    }
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;

    double cost = HungAlgo.Solve(HungarianCost, assignment);
    double costfiltered = 0;
    vector<vector<float>> stereo_correspondences;
    for (unsigned int x = 0; x < HungarianCost.size(); x++) {
        //std::cout << x << "," << assignment[x] << "\t";
        vector<float> assign;
        if(x != -1 && assignment[x] != -1) {
            if(HungarianCost[x][assignment[x]] > cost_threshold) continue;
            costfiltered+=HungarianCost[x][assignment[x]];
            stereo_correspondences.push_back({index_centers_l[x].x, index_centers_l[x].y, index_centers_r[assignment[x]].x, index_centers_r[assignment[x]].y});
        }
    }

    std::cout << "\ncost: " << cost << " | cost after filtering: " << costfiltered << std::endl;
    return stereo_correspondences;
}

void populate_final_matches(Frame prev_frame, Frame* curr_frame) {
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
    int cost_threshold_stereo = 100;
    int cost_threshold_temporal = 100;
    for(int i=0; i<30 ; i++) {
        framesImages.push_back(to_string(i));
        cout << "Processing a new image index:" << i << endl;
        Frame frame;
        frame.id = i;
        string filename_l =  "/home/remote_user2/olslam/sorghum_dataset/row4/final_op_row4_left/res_" + to_string(i) + ".csv";
        string filename_r =  "/home/remote_user2/olslam/sorghum_dataset/row4/final_op_row4_right/res_" + to_string(i) + ".csv";
        
        string img_path_l =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/rect1_fullres/frame" + to_string(i) + ".jpg";
        string img_path_r =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/rect0_fullres/frame" + to_string(i) + ".jpg"; 
        
        Mat img_l = imread( img_path_l);
        Mat img_r = imread( img_path_r);
        
        frame.img_l = img_l;
        frame.img_r = img_r;

        r left = get_points_2D(filename_l);
        r right = get_points_2D(filename_r);
        
        int img_width = img_l.size().width;

        vector<vector<float>> stereo_correspondences = run_hungarian(left, right, img_width, vertical_threshold_stereo, horizontal_threshold_stereo, cost_threshold_stereo);

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

        string filename_prev =  "/home/remote_user2/olslam/sorghum_dataset/row4/final_op_row4_left/res_" + to_string(i-1) + ".csv";
        r prev = get_points_2D(filename_prev);

        vector<vector<float>> correspondences_temporal = run_hungarian(prev, left, img_width, vertical_threshold_temporal, horizontal_threshold_temporal, cost_threshold_temporal);

        Mat img_prev = frames[i-1].img_l;
        write_correspondences_temporal(img_prev, img_l, i, correspondences_temporal);
        frame.temporal_correspondences = correspondences_temporal; 

        corr ret = get_2D_temporal_corr(&correspondences_temporal);
        frame.src2D = ret.src2D;
        frame.dst2D = ret.dst2D;
        populate_final_matches(frames[i-1], &frame);
    
        cout << frame.pc.size() << endl;
        cout << frame.src2D.size() << endl;
        cout << frame.dst2D.size() << endl;
        cout << frame.src3Dtemporal.size() << endl;
        cout << frame.dst3Dtemporal.size() << endl;
        frames.push_back(frame);
    }
    int pose_id = 1;
    vector<Pose3> poses;
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
    for(int i=1; i<frames.size(); ++i) {
        considered_poses.push_back(i);
        Mat m1 = frames[i-1].pc;
        Mat m2 = frames[i].pc;
        vector<Point3f> src = frames[i].src3Dtemporal;
        vector<Point3f> dst = frames[i].dst3Dtemporal;
        cout << m1.size() << endl;
        cout << m2.size() << endl;
        retPose p = getPose(m1, m2, "icp"); 
        Pose3 pose(p.R, p.t);
        cout << pose << endl;
        poses.push_back(poses.back()*pose);
        // Need to construct the landmarks array
        // Initialize and create gtsam graph here
        // i is the image index
        if(poses.size() == 2) {
            //cout << "Initializing prevKeypointIndexer" << endl;
            for(int l =0; l < dst.size(); l++) {
                // assign incremental landmark IDs for the first two images
                //cout << l << " | " << src[l].x << " | " << dst[l].x << " | " << src[l].y << " | " << dst[l].y << endl;
                KeypointMapper[l].insert(make_pair(pose_id-1, src[l]));
                KeypointMapper[l].insert(make_pair(pose_id, dst[l]));
                prevKeypointIndexer[getKpKey3(dst[l])] = l;               
            }
            pose_id++;
            continue;
        }
        /* For each keypoint in the new image
           Check if the match at image i-1 already exists (data associated)
           If it does, get the of the corresponding landmark_id and populate KeypointMapper[landmark_id]
           If it does not, assign a new landmark_id and populate KeypointMapper[landmark_id]
        */
        if(poses.size() > 2) {
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
        }
        pose_id++;
    }
    cout << "===================KEYPOINTMAPPER====================" << endl;
    cout << KeypointMapper.size() << endl;
    gtsam::Values result;
    Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    ret_optimize ret_optimizer = Optimize_from_stereo(KeypointMapper, Kgt, framesImages, considered_poses, poses);
    result = ret_optimizer.result;

    // Test GTSAM output  
    //test_sfm(result, Kgt, considered_poses);
    reconstruct_pointcloud(result, Kgt, considered_poses, framesImages);

    return 0;


}