#include "test_sfm.h"
#include <math.h>
#include<cmath>

int NumDigits(int x)  
{  
    x = abs(x);  
    return (x < 10 ? 1 :   
        (x < 100 ? 2 :   
        (x < 1000 ? 3 :   
        (x < 10000 ? 4 :   
        (x < 100000 ? 5 :   
        (x < 1000000 ? 6 :   
        (x < 10000000 ? 7 :  
        (x < 100000000 ? 8 :  
        (x < 1000000000 ? 9 :  
        10)))))))));  
}  

#define PI   3.1415926535897932384626433832795

bool check_bound(float x, float y, vector<vector<float>>  bounds3d) {
    for (int i=0; i < bounds3d.size(); i++) {
        float x1 = bounds3d[i][0];
        float y1 = bounds3d[i][1];
        float x2 = bounds3d[i][2];
        float y2 = bounds3d[i][3];
        if(x > x1 && y > y1 && x < x2 && y < y2){
            return true;
        }
    }
    return false;
}

bool check_bound2(float x, float y, vector<vector<float>>  points) {
    for (int i=0; i < points.size(); i++) {
        float x1 = points[i][0];
        float y1 = points[i][1];
        if(abs(x - x1) < 10e-2 && abs(y - y1) < 10e-2) {
            return true;
        }
    }
    return false;
}


bool is_in_ellipse(float xp, float yp,  float x, float y, float a, float b, float alpha)
{
    float v = pow(cos(alpha)*(xp - x) + sin(alpha)*(yp - y), 2) / pow(a, 2) + pow(sin(alpha)*(xp - x) - cos(alpha)*(yp - y), 2) / pow(b, 2);
    if(v <= 1) {
        return true;
    }
    else {
        return false;
    }
}

vector<vector<float>> get_points(int pose_id, Mat disparity) {
    string idx; 
    idx = to_string(pose_id);
    string filename =  "/home/remote_user2/test_icp_simple/clusters/clusters2/151_5_30/final_op_151_5.30.19_umass/res_" + idx + ".csv";
    cout << "CSV FILE " << filename << endl;
    vector<vector<float>> csv = read_csv(filename);
    vector<vector<float>> points;
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][8];
        float y1 = csv[j][9];
        float x2 = csv[j][10];
        float y2 = csv[j][11];
       
        float d1 = disparity.at<uchar>((int)y1,(int)x1);
        if(d1 == 0) {
            continue;
        }
        int w = int(x2) - int(x1);
        int h = int(y2) - int(y1);
        float x = csv[j][3];
        float y = csv[j][4];
        float a = csv[j][5]/2;
        float b = csv[j][6]/2;
        float alpha = csv[j][7]*PI/180;
        for(int ww=0; ww < w; ww++) {
            for (int hh=0; hh < h; hh++) {
                if(is_in_ellipse(ww, hh, x, y, a, b, alpha)) {
                    float xf = ww + x1;
                    float yf = hh + y1;
                    vector<float> point;
                    point.push_back(xf);
                    point.push_back(yf);
                    points.push_back(point);
                }
            }
        }
    }
    return points;
}

vector<vector<float>> get_3d_bounds(int pose_id, Mat disparity){
    string idx; 
    if (NumDigits(pose_id) == 1) idx = "00" + to_string(pose_id);
    if (NumDigits(pose_id) == 2) idx = "0" + to_string(pose_id);
    if (NumDigits(pose_id) == 3) idx = to_string(pose_id);
    idx = to_string(pose_id);
    string filename =  "/home/remote_user2/test_icp_simple/clusters/clusters2/151_5_30/final_op_151_5.30_umass/res_" + idx + ".csv";
    cout << "CSV FILE " << filename << endl;
    vector<vector<float>> csv = read_csv(filename);
    vector<vector<float>> bounds3d;
    for(int j = 0; j < csv.size(); j++) {
        
        float x1 = csv[j][8];
        float y1 = csv[j][9];
        float x2 = csv[j][10];
        float y2 = csv[j][11];
        float d1 = disparity.at<uchar>((int)y1,(int)x1);
        vector<float> bound;
        if(d1 == 0) {
            continue;
        }
        float z1 = baseline*focal_length/d1; 
        x1 = (x1-cx)*z1/focal_length;
        y1 = (y1-cy)*z1/focal_length;
        float d2 = disparity.at<uchar>((int)y2,(int)x2);
        if(d2 == 0) {
            continue;
        }
        float z2 = baseline*focal_length/d2; 
        x2 = (x2-cx)*z2/focal_length;
        y2 = (y2-cy)*z2/focal_length;
        bound.push_back(x1);
        bound.push_back(y1);
        bound.push_back(x2);
        bound.push_back(y2);
        bounds3d.push_back(bound);
    } 
    return bounds3d;
}

void reconstruct_pointcloud(gtsam::Values result, Cal3_S2::shared_ptr Kgt, vector<int> considered_poses) 
{
    cv::Mat Q;
    float data[16] = {1.0, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, focal_length, 0, 0, 1.0/baseline, (cx-cxprime)/baseline};
    Q = Mat(4, 4, CV_32FC1, &data);
    for(int i=0; i<considered_poses.size(); i++) {
        vector<float> pointcloud;
        string img_path = data_folder + "/frame"+  to_string(considered_poses[i]) + ".jpg";
        string img_path_target = image_folder + "/frame" + to_string(considered_poses[i]) + ".jpg";
        cout << "IMAGES FILE " << img_path_target << endl;
        cout << "disparity FILE " << img_path << endl;
        Mat disparity = imread( img_path , 0);
        Mat img = imread( img_path_target);


        cv::Mat reprojection(disparity.size(),CV_32FC3);
        cv::reprojectImageTo3D(disparity, reprojection, Q);
        Pose3 P = result.at(Symbol('x', i).key()).cast<Pose3>();
        cout << P << endl;
        //vector<vector<float>> bounds3d = get_3d_bounds(considered_poses[i], disparity);
        vector<vector<float>> points = get_points(considered_poses[i], disparity);
        Vec3b color;
        color[0] = 255;
        color[1] = 255;
        color[2] = 255;
        for (int i=0; i < points.size(); i++) {
            int x = int(points[i][0]);
            int y = int(points[i][1]);
            //cout << x << "|" << y << endl;
            img.at<Vec3b>(y, x) = color;
            Vec3f p = reprojection.at<Vec3f>(y,x);
            Point3 x3D; 
            x3D(0) = p[0];
            x3D(1) = p[1];
            x3D(2) = p[2];
            if(p[2] < baseline*35) {
                Point3 x3Dw = P.transform_from(x3D);
                pointcloud.push_back(x3Dw(0));
                pointcloud.push_back(x3Dw(1));
                pointcloud.push_back(x3Dw(2));      
            }        
            
        }
        imwrite("/home/remote_user2/test_icp_simple/results/frame"+to_string(considered_poses[i])+".jpg",img); 
        
        /*
        for(int r = 0; r < reprojection.rows; r++) {
            // We obtain a pointer to the beginning of row r
            cv::Point3f* ptr = reprojection.ptr<Point3f>(r);          
            for(int c = 0; c < reprojection.cols; c++) {
                if(ptr[c].z < baseline*35) {
                    Point3 x3D; 
                    x3D(0) = ptr[c].x;
                    x3D(1) = ptr[c].y;
                    x3D(2) = ptr[c].z;
                    if(check_bound2(ptr[c].x, ptr[c].y, points)) {
                        Point3 x3Dw = P.transform_from(x3D);
                        pointcloud.push_back(x3Dw(0));
                        pointcloud.push_back(x3Dw(1));
                        pointcloud.push_back(x3Dw(2));
                    }
                }
            }
        }
        */
        Mat m = Mat(pointcloud.size()/3, 1, CV_32FC3);
        cout << m.size() << endl;
        memcpy(m.data, pointcloud.data(), pointcloud.size()*sizeof(float)); 
        save_vtk<float>(m, "/home/remote_user2/test_icp_simple/vtk/frame" + to_string(considered_poses[i]) + ".vtk");
    }



}

void test_sfm(gtsam::Values result, Cal3_S2::shared_ptr Kgt, vector<int> considered_poses)
{
    cout << "==================== TESTING =================" << endl;
    //cout << "image 0 :" << img_path << endl;
    int pose_id = considered_poses.size()-1;
    
    int target_frame_id = considered_poses[pose_id];
    int initial_frame_id = considered_poses[0];
    cout << "initial_frame_id " << initial_frame_id << endl;
    cout << "target_frame_id " << target_frame_id << endl;

    string img_path_target = image_folder + "/frame" + to_string(target_frame_id) + ".jpg";
    Mat img_target = imread(  img_path_target );
    string img_path = image_folder + "/frame" + to_string(initial_frame_id) + ".jpg";
    Mat img_orig = imread( img_path );
    
    string idx; 
    if (NumDigits(initial_frame_id) == 1) idx = "00" + to_string(initial_frame_id);
    if (NumDigits(initial_frame_id) == 2) idx = "0" + to_string(initial_frame_id);
    if (NumDigits(initial_frame_id) == 3) idx = to_string(initial_frame_id);
    
    string filename =  "/home/remote_user2/test_icp_simple/clusters/clusters2/151_5_30/bbox/frame000" + idx + "_FRUITLET.csv";
    cout << "CSV FILE " << filename << endl;
    vector<vector<float>> csv = read_csv(filename);
    img_path = data_folder + "/frame"+  to_string(initial_frame_id) + ".jpg";
    cout << "disparity FILE " << img_path << endl;
    Mat disparity = imread( img_path , 0);
    
    RNG rng(12345);
    
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][0];
        float y1 = csv[j][1];
        float x2 = csv[j][2];
        float y2 = csv[j][3];
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::Point2f pt1(x1, y1);
        // and its bottom right corner.
        cv::Point2f  pt2(x2, y2);
        // These two calls...
        //cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
        //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow( "Display window", img );                   // Show our image inside it.
        //waitKey(0);         
        
        float x = (x1+x2)/2;
        float y = (y1+y2)/2;
        
        cv::Point2f pt3(x, y);
        
        cout << "x= " << x << " | y= " << y << endl;
        float d = disparity.at<uchar>((int)y,(int)x);
        float z = baseline*focal_length/d;
        x = (x-cx)*z/focal_length;
        y = (y-cy)*z/focal_length;
        Point3 landmark3d;
        landmark3d(0) = x;
        landmark3d(1) = y;
        landmark3d(2) = z;
        Pose3 P0 = result.at(Symbol('x', 0).key()).cast<Pose3>();
        Pose3 P = result.at(Symbol('x', pose_id).key()).cast<Pose3>();
        //cout << P0 << endl;
        //landmark3d = P0.rotation()*(landmark3d - P0.translation());
        cout << landmark3d << endl;
        PinholeCamera<Cal3_S2> camera(P, *Kgt);
        Point2 measurement = camera.project(landmark3d);
        Point2f mmm;
        mmm.x = measurement(0);
        mmm.y = measurement(1);
        cout << measurement << endl;
        ShowBlackCircle(img_orig, pt3, 5, color);
        ShowBlackCircle(img_target, mmm, 5, color);
    }
    int t_out = 0;
    std::string win_name = "circle";
    cv::imshow( win_name, img_orig ); cv::waitKey( t_out );
    cv::imshow( win_name, img_target ); cv::waitKey( t_out );
}
