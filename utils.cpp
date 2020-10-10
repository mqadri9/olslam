#include "utils.h"    
    

vector<vector<float>> read_csv(std::string filename){

    // Create an input filestream
    std::ifstream myFile(filename);
    vector<vector<float>> result;
    std::string line;
    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    while(std::getline(myFile, line))
    {
        vector<float> tmp;
        std::stringstream ss(line);
        while( ss.good() )
        {
            string substr;
            getline( ss, substr, ',' );
            float d = boost::lexical_cast<float>(substr);
            tmp.push_back( d );
        }
        result.push_back(tmp);
    }
    return result;
}

void printKeypointMapper(map<int, map<int, Point2f>> mainMap) {
  map<int, map<int, Point2f> >::iterator it;
  map<int, Point2f>::iterator inner_it;
  for ( it=mainMap.begin() ; it != mainMap.end(); it++ ) {
    cout << "\n\nNew element\n" << (*it).first << endl;

    for( inner_it=(*it).second.begin(); inner_it != (*it).second.end(); inner_it++)
      cout << (*inner_it).first << " => " << (*inner_it).second << endl;
  }
}

void printKeypointIndexer(map<string, int> mainMap)
{
  map<string, int>::iterator it;

  for ( it=mainMap.begin() ; it != mainMap.end(); it++ ) {
    cout << it->first << " => " << it->second << endl;
  }
}

string getKpKey(Point2f m){
    ostringstream key ;
    key << m.x << "," << m.y;
    return key.str();
}

string getKpKey3(Point3f m){
    ostringstream key ;
    key << m.x << "," << m.y;
    return key.str();
}

PointMatcher<float>::DataPoints create_datapoints(Mat pointcloud) {
    ostringstream os; 
    os << "x,y,z\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<float>(i, 0) << "," << pointcloud.at<float>(i, 1) << "," << pointcloud.at<float>(i, 2) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<float>::loadCSV(in_stream);
    const PointMatcher<float>::DataPoints PC(points);
    return PC;
}

void ShowBlackCircle( const cv::Mat & img, cv::Point cp, int radius, Scalar color)
{
    cv::circle( img, cp, radius, color ,CV_FILLED, 8,0);
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
    string filename =  csv_folder + "/res_" + idx + ".csv";
    cout << "CSV FILE " << filename << endl;
    vector<vector<float>> csv = read_csv(filename);
    vector<vector<float>> points;
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][8];
        float y1 = csv[j][9];
        float x2 = csv[j][10];
        float y2 = csv[j][11];
        float x_center = (x1 + x2)/2;
        float y_center = (y1+y2) /2;
        float d1 = disparity.at<uchar>((int)y_center,(int)x_center);
        if(d1 == 0) {
            continue;
        }
        int w = int(x2) - int(x1);
        int h = int(y2) - int(y1);
        float x = csv[j][3]*ellipse_resize_factor;
        float y = csv[j][4]*ellipse_resize_factor;
        float a = ellipse_resize_factor*csv[j][5]/2;
        float b = ellipse_resize_factor*csv[j][6]/2;
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
    string filename =  "/home/remote_user2/olslam/sorghum_dataset/final_op_row2/res_" + idx + ".csv";
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
