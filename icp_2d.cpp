#include "utils.h"
#include "pose.h"
typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;    
PM::ICP icp;
#include "Eigen/StdVector"
#include "Eigen/Core"
#include "Eigen/Geometry"
struct r {
    Mat m;
    vector<Point2f> centers;
    Mat pc;
    vector<vector<float>> stereo_correspondences;
};

r get_points_2D(string filename) {
    vector<vector<float>>csv = read_csv(filename);
    vector<Point2f> centers;    
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][8]/RESIZE_FACTOR;
        float y1 = csv[j][9]/RESIZE_FACTOR;
        float x = csv[j][3] + x1;
        float y = csv[j][4] + y1;
        Point2f center(x, y);
        centers.push_back(center);
    }
    r ret;
    ret.centers = centers;
    return ret;
}

double xydist(Point3 p1, Point3 p2) {
    float x1 = p1(0);
    float y1 = p1(1);
    float x2 = p2(0);
    float y2 = p2(1);
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}


float closest_match(vector<Point2f>* centers_right, float x, float y) {
    float dist = 9999;
    for(int i =0; i<(*centers_right).size();i++) {
        Point2f point = (*centers_right)[i];
        if(abs(point.y - y) > 20 ) continue;
            if(sqrt(pow(x-point.x, 2) + pow(y-point.y, 2)) < dist) {
                dist = sqrt(pow(x-point.x, 2) + pow(y-point.y, 2));
            }

    }
    return dist;
} 

r get_points_2D_3D(string filename_l, string filename_r, Mat disparity) {
    vector<vector<float>>csv = read_csv(filename_l);
    r right_points = get_points_2D(filename_r);
    vector<Point2f> centers_right = right_points.centers;
    vector<float> points;
    vector<float> pointcloud;
    vector<Point2f> centers;
    vector<vector<float>> stereo_correspondences;
    for(int j = 0; j < csv.size(); j++) {
        float x1 = csv[j][8]/RESIZE_FACTOR;
        float y1 = csv[j][9]/RESIZE_FACTOR;
        float x = csv[j][3] + x1;
        float y = csv[j][4] + y1;
        float d = disparity.at<uchar>((int)(y*RESIZE_FACTOR),(int)(x*RESIZE_FACTOR));
        d = d/RESIZE_FACTOR;
        double uR = x - d;
        float dist = closest_match(&centers_right, uR, y);
        if(dist>40) continue;
        vector<float> point;
        Point2f center(x, y);
        points.push_back(x);    
        points.push_back(y);
        centers.push_back(center);
        if(d < MIN_DISPARITY || uR < 0) {
            continue;
        }        
        stereo_correspondences.push_back({x, y, uR, d});
        float Z = baseline*focal_length/d; 
        float X = (x-cx)*Z/focal_length;
        float Y = (y-cy)*Z/focal_length;
        pointcloud.push_back(X);
        pointcloud.push_back(Y);
        pointcloud.push_back(Z);
    }
    Mat m1 = Mat(points.size()/2, 1, CV_32FC2);
    memcpy(m1.data, points.data(), points.size()*sizeof(float)); 
    
    Mat m2 = Mat(pointcloud.size()/3, 1, CV_32FC3);
    memcpy(m2.data, pointcloud.data(), pointcloud.size()*sizeof(float));
    r ret;
    ret.m = m1;
    ret.centers = centers;
    ret.pc = m2;
    ret.stereo_correspondences = stereo_correspondences;
    return ret;
}

PointMatcher<float>::DataPoints create_datapoints_2D(Mat pointcloud) {
    ostringstream os; 
    os << "x,y\n";
    for(int i = 0; i < pointcloud.rows; i++)
    {
       os << pointcloud.at<float>(i, 0) << "," << pointcloud.at<float>(i, 1) << "\n";
    } 
    string s = os.str();
    std::istringstream in_stream(s);
    auto points = PointMatcherIO<float>::loadCSV(in_stream);
    const PointMatcher<float>::DataPoints PC(points);
    return PC;
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
        end.y = entry[1];
        line( HM,
            start,
            end,
            0xffff,
            thickness,
            lineType );
    }
    imwrite("/home/remote_user2/olslam/stereo_correspondences/frame" + to_string(frame_id) + ".jpg", HM); 
}

RNG rng(12345);

int main(int argc, char* argv[]) {
    r points_prev;
    r points_curr;

	PM::ICP icp;
	PointMatcherSupport::Parametrizable::Parameters params;
	std::string name;
	
	// Uncomment for console outputs
	setLogger(PM::get().LoggerRegistrar.create("FileLogger"));

	// Prepare reading filters
	name = "MinDistDataPointsFilter";
	params["minDist"] = "0";
	std::shared_ptr<PM::DataPointsFilter> minDist_read =
		PM::get().DataPointsFilterRegistrar.create(name, params);
	params.clear();

	name = "RandomSamplingDataPointsFilter";
	params["prob"] = "1";
	std::shared_ptr<PM::DataPointsFilter> rand_read =
		PM::get().DataPointsFilterRegistrar.create(name, params);
	params.clear();

	// Prepare reference filters
	name = "MinDistDataPointsFilter";
	params["minDist"] = "0";
	std::shared_ptr<PM::DataPointsFilter> minDist_ref =
		PM::get().DataPointsFilterRegistrar.create(name, params);
	params.clear();

	name = "RandomSamplingDataPointsFilter";
	params["prob"] = "1";
	std::shared_ptr<PM::DataPointsFilter> rand_ref =
		PM::get().DataPointsFilterRegistrar.create(name, params);
	params.clear();

	// Prepare matching function
	name = "KDTreeMatcher";
	params["knn"] = "1";
	params["epsilon"] = "3.16";
	std::shared_ptr<PM::Matcher> kdtree =
		PM::get().MatcherRegistrar.create(name, params);
	params.clear();

	// Prepare outlier filters
	name = "TrimmedDistOutlierFilter";
	params["ratio"] = "0.85";
	std::shared_ptr<PM::OutlierFilter> trim =
		PM::get().OutlierFilterRegistrar.create(name, params);
	params.clear();

	// Prepare error minimization
	name = "PointToPointTranslationErrorMinimizer";
	std::shared_ptr<PM::ErrorMinimizer> pointToPoint =
		PM::get().ErrorMinimizerRegistrar.create(name);

	// Prepare transformation checker filters
	name = "CounterTransformationChecker";
	params["maxIterationCount"] = "30";
	std::shared_ptr<PM::TransformationChecker> maxIter =
		PM::get().TransformationCheckerRegistrar.create(name, params);
	params.clear();

	name = "DifferentialTransformationChecker";
	params["minDiffRotErr"] = "0.001";
	params["minDiffTransErr"] = "0";
	params["smoothLength"] = "4";
	std::shared_ptr<PM::TransformationChecker> diff =
		PM::get().TransformationCheckerRegistrar.create(name, params);
	params.clear();

	// Prepare inspector
	std::shared_ptr<PM::Inspector> nullInspect =
		PM::get().InspectorRegistrar.create("NullInspector");

	//name = "VTKFileInspector";
    //	params["dumpDataLinks"] = "1"; 
    //	params["dumpReading"] = "1"; 
    //	params["dumpReference"] = "1"; 

	//PM::Inspector* vtkInspect =
	//	PM::get().InspectorRegistrar.create(name, params);
	params.clear();
	
	// Prepare transformation
	std::shared_ptr<PM::Transformation> rigidTrans =
		PM::get().TransformationRegistrar.create("RigidTransformation");
	
	// Build ICP solution
	icp.readingDataPointsFilters.push_back(minDist_read);
	icp.readingDataPointsFilters.push_back(rand_read);

	icp.referenceDataPointsFilters.push_back(minDist_ref);
	icp.referenceDataPointsFilters.push_back(rand_ref);

	icp.matcher = kdtree;
	
	icp.outlierFilters.push_back(trim);
	
	icp.errorMinimizer = pointToPoint;

	icp.transformationCheckers.push_back(maxIter);
	icp.transformationCheckers.push_back(diff);
	
	// toggle to write vtk files per iteration
	icp.inspector = nullInspect;
	//icp.inspector = vtkInspect;

	icp.transformations.push_back(rigidTrans);

	// Prepare error minimization
	/*name = "PointToPointTranslationErrorMinimizer";
	std::shared_ptr<PM::ErrorMinimizer> pointToPoint =
		PM::get().ErrorMinimizerRegistrar.create(name);

	// Prepare transformation checker filters
	name = "CounterTransformationChecker";
	params["maxIterationCount"] = "20";
	std::shared_ptr<PM::TransformationChecker> maxIter =
		PM::get().TransformationCheckerRegistrar.create(name, params);
	params.clear();
		
	icp.errorMinimizer = pointToPoint;

	icp.transformationCheckers.push_back(maxIter);

	// toggle to write vtk files per iteration
	//icp.inspector = vtkInspect;
    */
    for(int i=0; i < 44; i++) {
        cout << "Processing a new image" << endl;
        string filename_l =  "/home/remote_user2/olslam/sorghum_dataset/row4/final_op_row4_left/res_" + to_string(i) + ".csv";
        string filename_r =  "/home/remote_user2/olslam/sorghum_dataset/row4/final_op_row4_right/res_" + to_string(i) + ".csv";

        string img_path_l =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/rect1_fullres/frame" + to_string(i) + ".jpg";
        string img_path_r =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/rect0_fullres/frame" + to_string(i) + ".jpg";        

        string disp_path_l =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/disparities/frame" + to_string(i) + ".jpg";

        Mat disparity = imread( disp_path_l , 0);

        Mat img_l = imread( img_path_l);
        Mat img_r = imread( img_path_r);
        
        
        if (i==0) {
            points_curr = get_points_2D_3D(filename_l, filename_r, disparity);
            vector<vector<float>> stereo_correspondences = points_curr.stereo_correspondences;
            write_correspondences(img_l, img_r, i, stereo_correspondences);
            continue;
        }
        points_prev = points_curr;
        points_curr = get_points_2D_3D(filename_l, filename_r, disparity);
        vector<vector<float>> stereo_correspondences = points_curr.stereo_correspondences;
        write_correspondences(img_l, img_r, i, stereo_correspondences);        
        
        const DP ref = create_datapoints(points_prev.pc);
        const DP data = create_datapoints(points_curr.pc);        
        

        PM::TransformationParameters T = icp(data, ref);
        DP data_out(data);
        icp.transformations.apply(data_out, T);
        ref.save("/home/remote_user2/olslam/saved_vtks/res_ref.vtk");
        data.save("/home/remote_user2/olslam/saved_vtks/res_init.vtk");
        data_out.save("/home/remote_user2/olslam/saved_vtks/res_transformed.vtk");
        cout << "i " << i << T << endl;
        string img_path_prev =  "/home/remote_user2/olslam/sorghum_dataset/row4/stereo_tmp_seed/rect1_fullres/frame" + to_string(i-1) + ".jpg";
        Mat img_prev = imread( img_path_prev);
        cout << img_prev.size() << endl;
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& pts(data_out.features);
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        for(int c = 0; c < pts.cols(); c++)
        {
            float x = cx + pts(0, c)*focal_length/pts(2,c);
            float y = cy + pts(1, c)*focal_length/pts(2,c);
            Point2f t(x,y);
            ShowBlackCircle(img_prev, t, 10, Scalar(0,0,225));
        }
        cout << "/home/remote_user2/olslam/centers_left/frame" + to_string(i-1) + ".jpg" << endl;
        imwrite("/home/remote_user2/olslam/centers_left/frame" + to_string(i-1) + ".jpg", img_prev);

        //Mat img_prev_2 = imread( img_path_prev);
        
        //ShowBlackCircle(img_l, points_left.centers[j], 2, color);

        /*
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        const DP ref = create_datapoints_2D(points_left.m);
        const DP data = create_datapoints_2D(points_right.m);
        for(int j=0; j < points_left.centers.size(); j++) {
            ShowBlackCircle(img_l, points_left.centers[j], 1, color);
        }
        //for(int j=0; j < points_right.centers.size(); j++) {
        //    ShowBlackCircle(img_r, points_right.centers[j], 1, color);
        //}
        cout << "/home/remote_user2/olslam/centers_left/frame000" + idx + ".jpg" << endl;
        cout << img_r.size() << endl;
        imwrite("/home/remote_user2/olslam/centers_left/frame000" + idx + ".jpg", img_l); 
        imwrite("/home/remote_user2/olslam/centers_right/frame000" + idx + ".jpg", img_r); 
        icp.setDefault();
        PM::TransformationParameters T = icp(ref, data);
        cout << T << endl;*/
    }


    return 0;


}