#include "hungarian.h"

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
        //if(abs(end.y - start.y) > 50) continue;
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
        //if(abs(end.y - start.y) > 50) continue;
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

void print_centers(vector<Point2f> centers, Mat img, string save_path)
{
    RNG rng(12345);
    Mat img_copy = img.clone();
    Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    for (int ii=0; ii< centers.size(); ii++) {
        ShowBlackCircle(img_copy, centers[ii], 10, color);
    }
    imwrite(save_path, img_copy); 
}

void find_top_score(unordered_map<int, float>*top_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    float total_dist = 0;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(y - y_tmp < 1000 && y - y_tmp > 0 && abs(x - x_tmp) < 15 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy;
            float dist = sqrt(pow(x_tmp - x, 2) + pow(y_tmp - y, 2)); 
            total_dist += dist;
            score += 1;
        }
    }
    (*top_map)[k] = total_dist;
}

void find_bottom_score(unordered_map<int, float>*bottom_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    float total_dist = 0;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(y_tmp - y < 1000 && y_tmp - y > 0 && abs(x - x_tmp) < 15 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy;
            float dist = sqrt(pow(x_tmp - x, 2) + pow(y_tmp - y, 2)); 
            total_dist += dist;
            score += 1;
        }
    }
    (*bottom_map)[k] = total_dist;
}


void find_left_score(unordered_map<int, float>*left_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    float total_dist = 0;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(x - x_tmp < 1000 && x - x_tmp > 0 && abs(y_tmp - y) < 15 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy;
            float dist = sqrt(pow(x_tmp - x, 2) + pow(y_tmp - y, 2)); 
            total_dist += dist;
            score += 1;
        }
    }
    (*left_map)[k] = total_dist;
}

void find_right_score(unordered_map<int, float>*right_map, vector<Point2f>*centers, vector<vector<float>>* bboxes, Point2f center, int k)
{
    float x = center.x;
    float y = center.y;
    float score = 1;
    float total_dist = 0;
    for(int i=0; i < (*centers).size(); i++) {
        float x_tmp = (*centers)[i].x;
        float y_tmp = (*centers)[i].y;
        if(x_tmp - x < 1000 && x_tmp - x > 0 && abs(y_tmp - y) < 15 ) {
            //cout << "Taken " << center << " | " << (*centers)[i] << endl;
            vector<float> bb = (*bboxes)[i];
            float dx = abs(bb[0] - bb[2]);
            float dy = abs(bb[1] - bb[3]);
            float area = dx*dy; 
            float dist = sqrt(pow(x_tmp - x, 2) + pow(y_tmp - y, 2)); 
            total_dist += dist;
            score += 1;
        }
    }
    //cout << total_dist/score << endl;
    (*right_map)[k] = total_dist;
}

vector<vector<float>> run_(vector<Point2f> centers_l_filtered, 
                           vector<Point2f> centers_r_filtered,
                           vector<vector<float>> bboxes_l_filtered,
                           vector<vector<float>> bboxes_r_filtered,
                           unordered_map<int, float> left_to_left,
                           unordered_map<int, float> left_to_right,
                           unordered_map<int, float> right_to_left,
                           unordered_map<int, float> right_to_right,
                           unordered_map<int, float> left_to_top,
                           unordered_map<int, float> left_to_bottom,
                           unordered_map<int, float> right_to_top,
                           unordered_map<int, float> right_to_bottom,
                           int vertical_threshold, 
                           int horizontal_threshold, 
                           int cost_threshold,
                           unordered_map<int, Point2f> index_centers_l,
                           unordered_map<int, Point2f> index_centers_r) 
{
    vector<vector<float>> stereo_correspondences;
    if(centers_l_filtered.size() == 0 || centers_r_filtered.size() == 0) {
        return stereo_correspondences;
    }
    int ss = max(centers_l_filtered.size(), centers_r_filtered.size());
    std::vector<std::vector<double>> HungarianCost(ss, std::vector<double>(ss, 10e8));
    for(int k=0; k < centers_l_filtered.size(); k++)
    {
        vector<float> bb_left = bboxes_l_filtered[k];
        float dx_left = abs(bb_left[0] - bb_left[2]);
        float dy_left = abs(bb_left[1] - bb_left[3]);
        float area_left = dx_left*dy_left;
        for(int j =0 ; j < centers_r_filtered.size(); j++) {
            vector<float> bb_right = bboxes_r_filtered[j];
            float dx_right = abs(bb_right[0] - bb_right[2]);
            float dy_right = abs(bb_right[1] - bb_right[3]);
            float area_right = dx_right*dy_right;
            float vdist = abs(centers_l_filtered[k].y - centers_r_filtered[j].y);
            float xdist = centers_l_filtered[k].x - centers_r_filtered[j].x;
            //if(bb_left[0] < 500 || bb_right[0] > img_width - 500) {
            //    HungarianCost[k][j] = 10e8;
            //}
            if(xdist < 50) {
                HungarianCost[k][j] = 10e8;
            }
            else if(vdist > vertical_threshold) {
                HungarianCost[k][j] = 10e8;
            }
            else{
                HungarianCost[k][j] = vdist;//*abs(area_left - area_right)*100;
                //HungarianCost[k][j] += max(left_to_left.at(k), right_to_left.at(j))/min(left_to_left.at(k), right_to_left.at(j))*10 ;
                //HungarianCost[k][j] += max(left_to_right.at(k), right_to_right.at(j))/min(left_to_right.at(k), right_to_right.at(j))*10;
                HungarianCost[k][j] += 10*max(left_to_left.at(k), right_to_left.at(j))/(min(left_to_left.at(k), right_to_left.at(j)) + 0.01) ;
                HungarianCost[k][j] += 10*max(left_to_right.at(k), right_to_right.at(j))/(min(left_to_right.at(k), right_to_right.at(j)) + 0.01); 
                HungarianCost[k][j] += 10*max(left_to_top.at(k), right_to_top.at(j))/(min(left_to_top.at(k), right_to_top.at(j)) +0.01) ;
                HungarianCost[k][j] += 10*max(left_to_bottom.at(k), right_to_bottom.at(j))/(min(left_to_bottom.at(k), right_to_bottom.at(j)) +0.01); 
                //cout << max(left_to_top.at(k), right_to_top.at(j)) - min(left_to_top.at(k), right_to_top.at(j)) << endl;
                //cout << max(left_to_right.at(k), right_to_right.at(j)) - min(left_to_right.at(k), right_to_right.at(j)) << endl;          
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
    
    for (unsigned int x = 0; x < HungarianCost.size(); x++) {
        //std::cout << x << "," << assignment[x] << "\t";
        vector<float> assign;
        if(x != -1 && assignment[x] != -1) {
            //cout << HungarianCost[x][assignment[x]] << endl;
            if(HungarianCost[x][assignment[x]] > cost_threshold) continue;
            costfiltered+=HungarianCost[x][assignment[x]];
            stereo_correspondences.push_back({index_centers_l[x].x, index_centers_l[x].y, index_centers_r[assignment[x]].x, index_centers_r[assignment[x]].y});
        }
    }

    std::cout << "\ncost: " << cost << " | cost after filtering: " << costfiltered << std::endl;
    return stereo_correspondences;
}


vector<vector<float>> run_hungarian(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) 
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
    vector<float> bboxes_l_areas;
    vector<float> bboxes_r_areas;

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
    /*
    cout << "=============================" << endl;
    cout << centers_l_filtered.size() << endl;
    cout << bboxes_l_filtered.size() << endl;
    cout << centers_r_filtered.size() << endl;
    cout << bboxes_r_filtered.size() << endl;
    */

    for(int k=0; k<centers_l_filtered.size(); k++) {
        index_centers_l[k] = centers_l_filtered[k];
        float x1 = bboxes_l_filtered[k][0];
        float y1 = bboxes_l_filtered[k][1];
        float x2 = bboxes_l_filtered[k][2];
        float y2 = bboxes_l_filtered[k][3];
        float area = abs(x2-x1)*abs(y2-y1);
        bboxes_l_areas.push_back(area);
        find_left_score(&left_to_left, &centers_l_filtered, &bboxes_l_filtered, centers_l_filtered[k], k);
        find_right_score(&left_to_right, &centers_l_filtered, &bboxes_l_filtered, centers_l_filtered[k], k);
        //cout << centers_l_filtered[k] << " | " << left_to_right[k] << endl;
    }

    for(int k=0; k<centers_r_filtered.size(); k++) {
        index_centers_r[k] = centers_r_filtered[k];
        float x1 = bboxes_r_filtered[k][0];
        float y1 = bboxes_r_filtered[k][1];
        float x2 = bboxes_r_filtered[k][2];
        float y2 = bboxes_r_filtered[k][3];
        float area = abs(x2-x1)*abs(y2-y1);
        bboxes_r_areas.push_back(area);
        find_left_score(&right_to_left, &centers_r_filtered, &bboxes_r_filtered, centers_r_filtered[k], k);
        find_right_score(&right_to_right, &centers_r_filtered, &bboxes_r_filtered, centers_r_filtered[k], k);
    }

    accumulator_set<double, stats<tag::mean, tag::variance> > acc_l;
    for_each(bboxes_l_areas.begin(), bboxes_l_areas.end(), bind<void>(ref(acc_l), _1));       
    float mean_l = boost::accumulators::mean(acc_l);
    float std_l = sqrt(variance(acc_l));
    
    accumulator_set<double, stats<tag::mean, tag::variance> > acc_r;
    for_each(bboxes_r_areas.begin(), bboxes_r_areas.end(), bind<void>(ref(acc_r), _1));       
    float mean_r = boost::accumulators::mean(acc_r);
    float std_r = sqrt(variance(acc_r));
    cout << "MEAN RIGHT " << mean_r << " | " << std_r << endl; 
    cout << "MEAN LEFT " << mean_l << " | " << std_l << endl; 
    
    float th_low_r;
    float th_high_r;
    float th_low_l;
    float th_high_l;
    if (std_r < mean_r/2) {
        th_low_r = 3*mean_r; //- std_r/2;
        th_high_r = 3*mean_r; //+ std_r/2;
    }
    else {
        th_low_r = 100*mean_r; //- std_r/4 ;
        th_high_r = 100*mean_r; //+ std_r/4;
    }

    if (std_l < mean_l/2) {
        th_low_l = 3*mean_l ;//- std_l/2;
        th_high_l = 3*mean_l ;//+ std_l/2;
    }
    else {
        th_low_l = 100*mean_l; //- std_l/4;
        th_high_l = 100*mean_l; //+ std_l/4;
    }

    unordered_map<int, int> left_lower_indexer;
    unordered_map<int, int> left_higher_indexer;
    unordered_map<int, int> right_lower_indexer;
    unordered_map<int, int> right_higher_indexer;
    vector<Point2f> centers_l_filtered_lower;
    vector<Point2f> centers_r_filtered_lower;
    vector<Point2f> centers_l_filtered_upper;
    vector<Point2f> centers_r_filtered_upper;
    vector<vector<float>> bboxes_l_filtered_lower;
    vector<vector<float>> bboxes_r_filtered_lower;
    vector<vector<float>> bboxes_l_filtered_upper;
    vector<vector<float>> bboxes_r_filtered_upper;

    int left_lower_index = 0;
    int left_higher_index = 0;    
    int right_lower_index = 0;
    int right_higher_index = 0;

    unordered_map<int, float> left_to_left_lower;
    unordered_map<int, float> left_to_right_lower;
    unordered_map<int, float> right_to_left_lower;
    unordered_map<int, float> right_to_right_lower;
    unordered_map<int, float> left_to_left_upper;
    unordered_map<int, float> left_to_right_upper;
    unordered_map<int, float> right_to_left_upper;
    unordered_map<int, float> right_to_right_upper;

    unordered_map<int, float> left_to_bottom_lower;
    unordered_map<int, float> left_to_top_lower;
    unordered_map<int, float> right_to_bottom_lower;
    unordered_map<int, float> right_to_top_lower;
    unordered_map<int, float> left_to_bottom_upper;
    unordered_map<int, float> left_to_top_upper;
    unordered_map<int, float> right_to_bottom_upper;
    unordered_map<int, float> right_to_top_upper;

    unordered_map<int, Point2f> index_centers_l_lower;
    unordered_map<int, Point2f> index_centers_l_upper;

    unordered_map<int, Point2f> index_centers_r_lower;
    unordered_map<int, Point2f> index_centers_r_upper;

    for(int k=0; k<centers_l_filtered.size(); k++) {
        if (bboxes_l_areas[k] < th_low_l) {
            left_lower_indexer[left_lower_index] = k;
            index_centers_l_lower[left_lower_index] = centers_l_filtered[k];
            centers_l_filtered_lower.push_back(centers_l_filtered[k]);
            bboxes_l_filtered_lower.push_back(bboxes_l_filtered[k]);
            left_lower_index++;
        }
        else if (bboxes_l_areas[k] > th_high_l) {
            left_higher_indexer[left_higher_index] = k;
            index_centers_l_upper[left_higher_index] = centers_l_filtered[k];
            centers_l_filtered_upper.push_back(centers_l_filtered[k]);
            bboxes_l_filtered_upper.push_back(bboxes_l_filtered[k]);
            left_higher_index++;
        }
    }

    for(int k=0; k<centers_r_filtered.size(); k++) {
        if (bboxes_r_areas[k] < th_low_r) {
            right_lower_indexer[right_lower_index] = k;
            index_centers_r_lower[right_lower_index] = centers_r_filtered[k];
            centers_r_filtered_lower.push_back(centers_r_filtered[k]);
            bboxes_r_filtered_lower.push_back(bboxes_r_filtered[k]);
            right_lower_index++;
        }
        else if (bboxes_r_areas[k] > th_high_r) {
            right_higher_indexer[right_higher_index] = k;
            index_centers_r_upper[right_higher_index] = centers_r_filtered[k];
            centers_r_filtered_upper.push_back(centers_r_filtered[k]);
            bboxes_r_filtered_upper.push_back(bboxes_r_filtered[k]);
            right_higher_index++;
        }
    }

    string bbox_lower_left;
    string bbox_lower_right;
    string bbox_upper_right;
    string bbox_upper_left;

    cout << "Processing image " << to_string(frame_id) << " | " << letype << endl;
    if (letype=="stereo") {
        bbox_lower_left = "/home/remote_user2/olslam/stereo_small/left/frame" + to_string(frame_id) + ".jpg";
        bbox_lower_right = "/home/remote_user2/olslam/stereo_small/right/frame" + to_string(frame_id) + ".jpg";
        bbox_upper_right = "/home/remote_user2/olslam/stereo_large/right/frame" + to_string(frame_id) + ".jpg";
        bbox_upper_left =  "/home/remote_user2/olslam/stereo_large/left/frame" + to_string(frame_id) + ".jpg";
    }
    else {
        bbox_lower_left = "/home/remote_user2/olslam/temporal_small/left/frame" + to_string(frame_id) + ".jpg";
        bbox_lower_right = "/home/remote_user2/olslam/temporal_small/right/frame" + to_string(frame_id) + ".jpg";
        bbox_upper_right = "/home/remote_user2/olslam/temporal_large/right/frame" + to_string(frame_id) + ".jpg";
        bbox_upper_left =  "/home/remote_user2/olslam/temporal_large/left/frame" + to_string(frame_id) + ".jpg"   ;     
    }

    cout << centers_l_filtered_lower.size() << endl;
    cout << centers_l_filtered_upper.size() << endl;
    cout << centers_l_filtered.size() << endl;
    
    cout << centers_r_filtered_lower.size() << endl;
    cout << centers_r_filtered_upper.size() << endl;
    cout << centers_r_filtered.size() << endl;

    print_centers(centers_l_filtered_lower, img_l, bbox_lower_left);
    print_centers(centers_l_filtered_upper, img_l, bbox_upper_left);
    print_centers(centers_r_filtered_lower, img_r, bbox_lower_right);
    print_centers(centers_r_filtered_upper, img_r, bbox_upper_right);

    for(int k=0; k<centers_l_filtered_lower.size(); k++) {
        find_left_score(&left_to_left_lower, &centers_l_filtered_lower, &bboxes_l_filtered_lower, centers_l_filtered_lower[k], k);
        find_right_score(&left_to_right_lower, &centers_l_filtered_lower, &bboxes_l_filtered_lower, centers_l_filtered_lower[k], k);
        find_top_score(&left_to_top_lower, &centers_l_filtered_lower, &bboxes_l_filtered_lower, centers_l_filtered_lower[k], k);
        find_bottom_score(&left_to_bottom_lower, &centers_l_filtered_lower, &bboxes_l_filtered_lower, centers_l_filtered_lower[k], k);
    }

    for(int k=0; k<centers_l_filtered_upper.size(); k++) {
        find_left_score(&left_to_left_upper, &centers_l_filtered_upper, &bboxes_l_filtered_upper, centers_l_filtered_upper[k], k);
        find_right_score(&left_to_right_upper, &centers_l_filtered_upper, &bboxes_l_filtered_upper, centers_l_filtered_upper[k], k);
        find_top_score(&left_to_top_upper, &centers_l_filtered_upper, &bboxes_l_filtered_upper, centers_l_filtered_upper[k], k);
        find_bottom_score(&left_to_bottom_upper, &centers_l_filtered_upper, &bboxes_l_filtered_upper, centers_l_filtered_upper[k], k);
    }

    for(int k=0; k<centers_r_filtered_lower.size(); k++) {
        find_left_score(&right_to_left_lower, &centers_r_filtered_lower, &bboxes_r_filtered_lower, centers_r_filtered_lower[k], k);
        find_right_score(&right_to_right_lower, &centers_r_filtered_lower, &bboxes_r_filtered_lower, centers_r_filtered_lower[k], k);
        find_top_score(&right_to_top_lower, &centers_r_filtered_lower, &bboxes_r_filtered_lower, centers_r_filtered_lower[k], k);
        find_bottom_score(&right_to_bottom_lower, &centers_r_filtered_lower, &bboxes_r_filtered_lower, centers_r_filtered_lower[k], k);
    }

    for(int k=0; k<centers_r_filtered_upper.size(); k++) {
        find_left_score(&right_to_left_upper, &centers_r_filtered_upper, &bboxes_r_filtered_upper, centers_r_filtered_upper[k], k);
        find_right_score(&right_to_right_upper, &centers_r_filtered_upper, &bboxes_r_filtered_upper, centers_r_filtered_upper[k], k);
        find_top_score(&right_to_top_upper, &centers_r_filtered_upper, &bboxes_r_filtered_upper, centers_r_filtered_upper[k], k);
        find_bottom_score(&right_to_bottom_upper, &centers_r_filtered_upper, &bboxes_r_filtered_upper, centers_r_filtered_upper[k], k);
    }



    vector<vector<float>> stereo_correspondences_lower = run_(centers_l_filtered_lower, 
                                                              centers_r_filtered_lower, 
                                                              bboxes_l_filtered_lower, 
                                                              bboxes_r_filtered_lower,
                                                              left_to_left_lower,
                                                              left_to_right_lower,
                                                              right_to_left_lower,
                                                              right_to_right_lower,
                                                              left_to_top_lower,
                                                              left_to_bottom_lower,
                                                              right_to_top_lower,
                                                              right_to_bottom_lower,
                                                              vertical_threshold,
                                                              horizontal_threshold,
                                                              cost_threshold,
                                                              index_centers_l_lower,
                                                              index_centers_r_lower);     

   vector<vector<float>> stereo_correspondences_higher = run_(centers_l_filtered_upper, 
                                                              centers_r_filtered_upper, 
                                                              bboxes_l_filtered_upper, 
                                                              bboxes_r_filtered_upper,
                                                              left_to_left_upper,
                                                              left_to_right_upper,
                                                              right_to_left_upper,
                                                              right_to_right_upper,
                                                              left_to_top_upper,
                                                              left_to_bottom_upper,
                                                              right_to_top_upper,
                                                              right_to_bottom_upper,
                                                              vertical_threshold,
                                                              horizontal_threshold,
                                                              cost_threshold,
                                                              index_centers_l_upper,
                                                              index_centers_r_upper);     

    /*vector<vector<float>> stereo_correspondences = run_(centers_l_filtered, 
                                                        centers_r_filtered, 
                                                        bboxes_l_filtered, 
                                                        bboxes_r_filtered,
                                                        left_to_left,
                                                        left_to_right,
                                                        right_to_left,
                                                        right_to_right,
                                                        vertical_threshold,
                                                        horizontal_threshold,
                                                        cost_threshold,
                                                        index_centers_l,
                                                        index_centers_r);
    */
    vector<vector<float>> stereo_correspondences;
    for(int k=0; k<stereo_correspondences_lower.size(); k++) {
        stereo_correspondences.push_back(stereo_correspondences_lower[k]);
    }
    for(int k=0; k<stereo_correspondences_higher.size(); k++) {
        stereo_correspondences.push_back(stereo_correspondences_higher[k]);
    }    
                               

    return stereo_correspondences;
}