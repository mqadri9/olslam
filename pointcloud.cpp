#include "pointcloud.h"

Point3f pointMatched(Point2f x2Dtarget, std::map<string, std::tuple<Point2f, Point2f, Point3f>> targetMap) {
    std::map<string, std::tuple<Point2f, Point2f, Point3f>>::iterator iter ; 
    float dist = 99999;
    Point2f match;
    Point3f matched3D;
    for(iter = targetMap.begin(); iter != targetMap.end(); ++iter)
    {  
        tuple<Point2f, Point2f, Point3f> p = iter->second;
        Point2f pt = get<1>(p);
        //std::cout << pt.x << " | " << pt.y << " | " << x2Dtarget.x << " | " << x2Dtarget.y << std::endl;
        float currdist = sqrt(pow(pt.x - x2Dtarget.x, 2) + pow(pt.y - x2Dtarget.y, 2));
        if(currdist < dist) {
            match.x = pt.x;
            match.y = pt.y;
            matched3D = get<2>(p);
            dist = currdist;
        }
    }
    float matchdist = sqrt(pow(x2Dtarget.x - match.x, 2) + pow(x2Dtarget.y - match.y, 2));
    if(matchdist < MAX_WINDOW_ERR) {
        return  matched3D;
    }
    return Point3f(-1, -1, -1);
}


retPointcloud createPointCloudsFromStereoPairs(std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DPrevCurr, 
                                               std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DCurr, 
                                               std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DPrev,
                                               string img1_path,
                                               string img2_path,
                                               int id)
{
    cout << "map2D3DPrevCurr " << map2D3DPrevCurr.size() << endl;
    cout << "map2D3DPrev " << map2D3DPrev.size() << endl;
    cout << "map2D3DCurr " << map2D3DCurr.size() << endl;
    Mat imgleft = imread( img1_path  );
    Mat imgright = imread( img2_path );
    vector<float> rawcloud1;
    vector<float> rawcloud2;
    vector<Point2f> src;
    vector<Point2f> dst;
    vector<float> errors;
    std::map<string, std::tuple<Point2f, Point2f, Point3f>>::iterator iter ;
    Mat HM;
    hconcat(imgleft,imgright,HM);   
    for(iter = map2D3DPrevCurr.begin(); iter != map2D3DPrevCurr.end(); ++iter)
    {
        std::tuple<Point2f, Point2f, Point3f> pt =  iter->second;
        Point2f x2Dprev = get<0>(pt);
        Point2f x2Dcurr = get<1>(pt);
        Point3f x3Dprev = pointMatched(x2Dprev, map2D3DPrev);
        Point3f x3Dcurr = pointMatched(x2Dcurr, map2D3DCurr);
        //cout << pt << endl;
        // if pt found in both curr and prev map
        //std::cout << pt << " | " << !(map2D3DCurr.find(pt) == map2D3DCurr.end()) << " | " << !(map2D3DPrev.find(pt) == map2D3DPrev.end()) << std::endl;
        if ( x3Dprev.x != -1 && x3Dcurr.x != -1) {
            //std::cout << x3Dprev.x << " | " << x3Dprev.y << " | " << x3Dprev.z << std::endl;
            //std::cout << x3Dcurr.x << " | " << x3Dcurr.y << " | " << x3Dcurr.z << std::endl;
            rawcloud1.push_back(x3Dprev.x);
            rawcloud1.push_back(x3Dprev.y);
            rawcloud1.push_back(x3Dprev.z);
            rawcloud2.push_back(x3Dcurr.x);
            rawcloud2.push_back(x3Dcurr.y);
            rawcloud2.push_back(x3Dcurr.z);
            errors.push_back(abs(x3Dcurr.z - x3Dprev.z));
            Point2f start, end;
            start.x = x2Dprev.x;
            start.y = x2Dprev.y;
            end.x = x2Dcurr.x + imgleft.size().width;
            end.y = x2Dcurr.y;
            src.push_back(x2Dprev);
            dst.push_back(x2Dcurr);
            int thickness = 1;
            int lineType = cv::LINE_8;
            line( HM,
                start,
                end,
                0xffff,
                thickness,
                lineType );
        }
    }
    imwrite("/home/remote_user2/olslam/finalmatches/frame"+ to_string(id) + ".jpg",HM);  
    accumulator_set<double, stats<tag::mean, tag::variance> > acc;
    for_each(errors.begin(), errors.end(), bind<void>(ref(acc), _1));       
    float mean = boost::accumulators::mean(acc);
    float std = sqrt(variance(acc));
    vector<float> pointcloud1;
    vector<float> pointcloud2;
    int j = 0;      
    for(size_t l=0; l < errors.size(); l++) {
        // Only select points that have depth error less than 1cm
        pointcloud1.push_back(rawcloud1[j]);
        pointcloud1.push_back(rawcloud1[j+1]);
        pointcloud1.push_back(rawcloud1[j+2]);
        pointcloud2.push_back(rawcloud2[j]);
        pointcloud2.push_back(rawcloud2[j+1]);
        pointcloud2.push_back(rawcloud2[j+2]);                
        j = j + 3;
    }
    cout << "pointcloud size " << pointcloud1.size()/3 << endl;
    Mat m1 = Mat(pointcloud1.size()/3, 1, CV_32FC3);
    cout << m1.size() << endl;
    memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
    
    //save_vtk<float>(m1, "/home/remote_user2/olslam/sift_matches/1.vtk");

    Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
    memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float));

    //save_vtk<float>(m2, "/home/remote_user2/olslam/sift_matches/2.vtk");
    vector<Mat> ret;
    ret.push_back(m1);
    ret.push_back(m2);
    
    retPointcloud s;
    s.ret = ret;
    s.src = src;
    s.dst = dst;
        
    return s;

}


std::map<string, std::tuple<Point2f, Point2f, Point3f>> stereoKptsTo3D(std::vector< std::vector<DMatch> > knn_matches, 
                                                                        std::vector<KeyPoint> keypointsCurrLeft,
                                                                        std::vector<KeyPoint> keypointsCurrRight,
                                                                        bool check_error) {

    cout << "knn_matches " << knn_matches.size() << std::endl; 
    cout << "keypointsCurrLeft " << keypointsCurrLeft.size() << std::endl; 
    cout << "keypointsCurr " << keypointsCurrRight.size() << std::endl; 
    std::map<string, std::tuple<Point2f, Point2f, Point3f>> d2d3;
    for (auto &m : knn_matches) {
        if(m[0].distance < 0.7*m[1].distance) {
            float x1 = keypointsCurrLeft[m[0].queryIdx].pt.x;
            float y1 = keypointsCurrLeft[m[0].queryIdx].pt.y;
            float x2 = keypointsCurrRight[m[0].trainIdx].pt.x;
            float y2 = keypointsCurrRight[m[0].trainIdx].pt.y;
            if(!check_error || abs(y1-y2) < MAX_PIXEL_ERR) {
                float d = abs(x1 - x2);
                float Z = baseline*focal_length/d; 
                float X = (x1-cx)*Z/focal_length;
                float Y = (y1-cy)*Z/focal_length;
                string key = to_string((int)x1) + "_" +to_string((int)y1) + "_" +  to_string((int)x2) + "_" +to_string((int)y2);
                Point3f loc3D(X, Y, Z);
                std::tuple<Point2f, Point2f, Point3f> value(keypointsCurrLeft[m[0].queryIdx].pt, keypointsCurrRight[m[0].trainIdx].pt, loc3D);
                d2d3.insert( pair<string, std::tuple<Point2f, Point2f, Point3f>>(key, value)); 
            }        
        }
    }
    return d2d3;
}


retFiltering filterImagesByMatching(vector<string> frames)
{
    vector<int> tmp_considered;
    vector<int> considered_poses;
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypointsCurrLeft, keypointsCurrRight, keypointsPrevLeft, keypointsPrevRight;
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    //cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    Mat descriptorsCurrLeft, descriptorsCurrRight, descriptorsPrevLeft, descriptorsPrevRight; 
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector< std::vector<DMatch> > knn_matches_stereo_prev;
    std::vector< std::vector<DMatch> > knn_matches_stereo_curr;
    std::vector< std::vector<DMatch> > knn_matches_in_time;
    std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DPrev;
    std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DCurr;
    std::map<string, std::tuple<Point2f, Point2f, Point3f>> map2D3DPrevCurr;
    //cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    const float ratio_thresh = 0.7f;
    vector<retPointcloud> filteredOutput;
    retFiltering rf;   
    for(size_t i=0; i<frames.size(); ++i) {
        string img_left_path = image_folder + "/frame" + frames[i] + ".jpg";
        string img_right_path = image_folder_right + "/frame" + frames[i] + ".jpg";
        Mat imgLeft = imread( img_left_path  );
        if (imgLeft.empty()) {
            throw std::invalid_argument( "Unable to open image");
        }       
        Mat imgRight = imread( img_right_path  );
        if (imgRight.empty()) {
            throw std::invalid_argument( "Unable to open image");
        }
        if (i==0) {
            // At time 0, find matches from the left and right stereo images
            detector->detectAndCompute( imgLeft, noArray(), keypointsCurrLeft, descriptorsCurrLeft );
            detector->detectAndCompute( imgRight, noArray(), keypointsCurrRight, descriptorsCurrRight );
            matcher->knnMatch( descriptorsCurrLeft, descriptorsCurrRight, knn_matches_stereo_curr, 2 );  
            std::cout << "======================================CURRRIGHTPREVLEFTRIGHT================================" << std::endl;
            map2D3DCurr = stereoKptsTo3D(knn_matches_stereo_curr, keypointsCurrLeft, keypointsCurrRight, true);
            continue;
        }
        // At time t, find matches from the left and right stereo images
        // Save descriptors and keypoints from prev time step

        descriptorsPrevLeft = descriptorsCurrLeft;
        descriptorsPrevRight = descriptorsCurrRight;
        keypointsPrevLeft = keypointsCurrLeft;
        keypointsPrevRight = keypointsCurrRight;
        knn_matches_stereo_prev = knn_matches_stereo_curr;
        knn_matches_stereo_curr.clear();
        map2D3DPrev = map2D3DCurr;

        drawMatchesSift(knn_matches_stereo_prev, 
                        keypointsPrevLeft, 
                        keypointsPrevRight,
                        image_folder + "/frame" + frames[i-1] + ".jpg",
                        image_folder_right + "/frame" + frames[i-1] + ".jpg",
                        "/home/remote_user2/olslam/matches_stereo_prev/frame"+to_string(i-1)+".jpg");

        // Find matches from the left and right stereo images at time t
        detector->detectAndCompute( imgLeft, noArray(), keypointsCurrLeft, descriptorsCurrLeft );
        detector->detectAndCompute( imgRight, noArray(), keypointsCurrRight, descriptorsCurrRight );        
        matcher->knnMatch( descriptorsCurrLeft, descriptorsCurrRight, knn_matches_stereo_curr, 2 );
        std::cout << "======================================CURRRIGHTCURRLEFT================================" << std::endl;
        map2D3DCurr = stereoKptsTo3D(knn_matches_stereo_curr, keypointsCurrLeft, keypointsCurrRight, true); 

        drawMatchesSift(knn_matches_stereo_curr, 
                        keypointsCurrLeft, 
                        keypointsCurrRight,
                        image_folder + "/frame" + frames[i] + ".jpg",
                        image_folder_right + "/frame" + frames[i] + ".jpg",
                        "/home/remote_user2/olslam/matches_stereo_curr/frame"+to_string(i)+".jpg");

        // Find matches between the left image at time t-1 and left image at time t
        knn_matches_in_time.clear();
        matcher->knnMatch( descriptorsPrevLeft, descriptorsCurrLeft, knn_matches_in_time, 2 );
        drawMatchesSift(knn_matches_in_time, 
                        keypointsPrevLeft, 
                        keypointsCurrLeft,
                        image_folder + "/frame" + frames[i-1] + ".jpg",
                        image_folder + "/frame" + frames[i] + ".jpg",
                        "/home/remote_user2/olslam/matches_stereo_time/frame"+to_string(i)+".jpg");
        std::cout << "======================================CURRRIGHTPREVCURRENT================================" << std::endl;
        map2D3DPrevCurr = stereoKptsTo3D(knn_matches_in_time, keypointsPrevLeft, keypointsCurrLeft, false);
        retPointcloud s =  createPointCloudsFromStereoPairs(map2D3DPrevCurr, 
                                                            map2D3DCurr, 
                                                            map2D3DPrev,
                                                            image_folder + "/frame" + frames[i-1] + ".jpg",
                                                            image_folder + "/frame" + frames[i] + ".jpg",
                                                            i);
     
        Mat m1 = s.ret[0];
        Mat m2 = s.ret[1];
        vector<Point2f> src = s.src;
        vector<Point2f> dst = s.dst;
        //cout << src.size() << endl;
        //cout << dst.size() << endl;
        //cout << "==========================================" << endl;
        filteredOutput.push_back(s);
        if(m1.size().height < 100) {
            cout << "Number of matches " << m1.size().height<< " found between frames " << i-1 << " and  "<< i << " is too low. Skipping frame." << i << endl;
            if(tmp_considered.size() > considered_poses.size()){
                considered_poses = tmp_considered;
            }
            tmp_considered.clear();
            continue;
        }
        tmp_considered.push_back(stoi(frames[i]));
    }
    if(tmp_considered.size() > considered_poses.size()){
        considered_poses = tmp_considered;
    }
    vector<retPointcloud> finalFilteredOutput;
    for(size_t i=0; i<frames.size(); ++i) {
        if (std::count(considered_poses.begin(), considered_poses.end(), stoi(frames[i]))) { 
            finalFilteredOutput.push_back(filteredOutput[i-1]);
        }
    }
    rf.considered_poses = considered_poses;
    rf.filteredOutput = finalFilteredOutput;
    std::cout << "filteeredOutput size " << finalFilteredOutput.size() << std::endl;
    //rf.descriptors1 = d1;
    //rf.descriptors2 = d2;
    //cout << rf.filteredOutput[0].ret[0] << endl;
    //cout << "==================================================================" << endl;
    //cout << rf.filteredOutput[27].ret[0] << endl;
    return rf;
}


retFiltering filterImages(vector<string> frames)
{
    vector<int> tmp_considered;
    vector<int> considered_poses;
    // Create SIFT detector and define parameters
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    //Ptr<SURF> detector = SURF::create( minHessian );
    cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    //cout <<  << endl;
    const float ratio_thresh = 0.7f;
    vector<retPointcloud> filteredOutput;
    vector<retPointcloud> tmpFilteredOutput;
    retFiltering rf;
    for(size_t i=0; i<frames.size(); ++i) {
        std::vector< std::vector<DMatch> > knn_matches;

        // Read image at index i 
        string img_path = image_folder + "/" + frames[i];
        cout << img_path <<"\n";
        Mat img = imread( img_path , IMREAD_GRAYSCALE );
        if (img.empty()) {
            throw std::invalid_argument( "Unable to open image");
        }        
        
        // If it is the first image in the sequence, detect the keypoints and continue 
        // to next image
        if (i==0) {
            detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
            tmp_considered.push_back(i);
            continue;
        }
        keypoints1 = keypoints2; 
        descriptors1 = descriptors2;         
        detector->detectAndCompute( img, noArray(), keypoints2, descriptors2 );
        
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );    
                       
        img_path = data_folder + "/" + frames[i-1];
        Mat disparity1 = imread( img_path , 0);
        img_path = data_folder + "/" + frames[i];
         
        Mat disparity2 = imread( img_path , 0);
        retPointcloud s = createPointClouds(disparity1, disparity2, keypoints1, keypoints2, knn_matches);
        //vector<Mat> rr = createPointClouds(disparity1, disparity2);
        Mat m11 = s.ret[0];
        //s.ret[0] = rr[0];
        //s.ret[1] = rr[1];
        
        Mat m1 = s.ret[0];
        Mat m2 = s.ret[1];
        //Mat m1 = rr[0];
        //Mat m2 = rr[1];
        
        vector<Point2f> src = s.src;
        vector<Point2f> dst = s.dst;
        
        if(m11.size().height < 100) {
            cout << "Number of matches " << m1.size().height<< " found between frames " << i-1 << " and  "<< i << " is too low. Skipping frame." << i << endl;
            if(tmp_considered.size() > considered_poses.size()){
                considered_poses = tmp_considered;
                filteredOutput = tmpFilteredOutput;
            }
            tmp_considered.clear();
            tmpFilteredOutput.clear();    
            tmp_considered.push_back(i);
            continue;
        }
        tmp_considered.push_back(i);
        tmpFilteredOutput.push_back(s);
    }
    
    
    if(tmp_considered.size() > considered_poses.size()){
        considered_poses = tmp_considered;
        filteredOutput = tmpFilteredOutput;
    }
    //vector<retPointcloud> finalFilteredOutput;
    //for(size_t i=0; i<frames.size(); ++i) {
    //    if (std::count(considered_poses.begin(), considered_poses.end(), stoi(frames[i]))) { 
    //        finalFilteredOutput.push_back(filteredOutput[i-1]);
    //    }
    // }
    rf.considered_poses = considered_poses;
    rf.filteredOutput = filteredOutput;
    //rf.descriptors1 = d1;
    //rf.descriptors2 = d2;
    //cout << rf.filteredOutput[0].ret[0] << endl;
    //cout << "==================================================================" << endl;
    //cout << rf.filteredOutput[27].ret[0] << endl;
    return rf;
}

retPointcloud createPointClouds(Mat disparity1, Mat disparity2, std::vector<KeyPoint> keypoints1, std::vector<KeyPoint> keypoints2, std::vector< std::vector<DMatch> > knn_matches)
{
    vector<float> errors;
    vector<float> rawcloud1;
    vector<float> rawcloud2;
    vector<Point2f> src;
    vector<Point2f> dst; 


    for (auto &m : knn_matches) {
        if(m[0].distance < 0.7*m[1].distance) {
            float x1 = keypoints1[m[0].queryIdx].pt.x;
            float y1 = keypoints1[m[0].queryIdx].pt.y;
            float d1 = disparity1.at<uchar>((int)y1,(int)x1);
            if(d1 == 0) {
                continue;
            }
            float z1 = baseline*focal_length/d1; 
            x1 = (x1-cx)*z1/focal_length;
            y1 = (y1-cy)*z1/focal_length;
            
            
            float x2 = keypoints2[m[0].trainIdx].pt.x;
            float y2 = keypoints2[m[0].trainIdx].pt.y;
            float d2 = disparity2.at<uchar>((int)y2,(int)x2);
            if(d2 == 0) {
                continue;
            }
            float z2 = baseline*focal_length/d2; 
            x2 = (x2-cx)*z2/focal_length;
            y2 = (y2-cy)*z2/focal_length;
            rawcloud1.push_back(x1);
            rawcloud1.push_back(y1);
            rawcloud1.push_back(z1);    
            rawcloud2.push_back(x2);
            rawcloud2.push_back(y2);
            rawcloud2.push_back(z2);
            errors.push_back(abs(z1 - z2));
            // Save the keypoint indices where the threshold condition 
            // is met for image i-1 and i
            src.push_back(keypoints1[m[0].queryIdx].pt);
            dst.push_back(keypoints2[m[0].trainIdx].pt);
        }
    }
    accumulator_set<double, stats<tag::mean, tag::variance> > acc;
    for_each(errors.begin(), errors.end(), bind<void>(ref(acc), _1));       
    float mean = boost::accumulators::mean(acc);
    float std = sqrt(variance(acc));
    vector<float> pointcloud1;
    vector<float> pointcloud2;
    int j = 0;      
    for(size_t l=0; l < errors.size(); l++) {
        // Only select points that have depth error less than 1cm
        if (errors[l] < mean/10 ) {
            pointcloud1.push_back(rawcloud1[j]);
            pointcloud1.push_back(rawcloud1[j+1]);
            pointcloud1.push_back(rawcloud1[j+2]);
            pointcloud2.push_back(rawcloud2[j]);
            pointcloud2.push_back(rawcloud2[j+1]);
            pointcloud2.push_back(rawcloud2[j+2]);                
        }
        j = j + 3;
    }
    //cout << pointcloud1.size()/3 << endl;
    Mat m1 = Mat(pointcloud1.size()/3, 1, CV_32FC3);
    cout << m1.size() << endl;
    memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
    
    //save_vtk<float>(m1, "/home/remote_user2/olslam/sift_matches/1.vtk");

    Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
    memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float));

    //save_vtk<float>(m2, "/home/remote_user2/olslam/sift_matches/2.vtk");
    vector<Mat> ret;
    ret.push_back(m1);
    ret.push_back(m2);
    
    retPointcloud s;
    s.ret = ret;
    s.src = src;
    s.dst = dst;
        
    return s;
}

vector<Mat> createPointClouds(Mat disparity1, Mat disparity2) 
{
    cv::Mat Q;
    float data[16] = {1.0, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, focal_length, 0, 0, 1.0/baseline, (cx-cxprime)/baseline};
    Q = Mat(4, 4, CV_32FC1, &data);
    //cout << Q << endl;
    cv::Mat reprojection1(disparity1.size(),CV_32FC3);
    cv::reprojectImageTo3D(disparity1, reprojection1, Q);
    //cout << pointcloud1.size() << endl;
    cv::Mat reprojection2(disparity2.size(),CV_32FC3);
    cv::reprojectImageTo3D(disparity2, reprojection2, Q);
    
    /*
    int yy = 94;
    int xx = 0;
    Point3f p = reprojection1.at<Point3f>(yy,xx);
    std::cout << p <<std::endl;
    float d1 = disparity1.at<uchar>(yy,xx);
    float z1 = baseline*focal_length/d1; 
    float x1 = (xx-cx)*z1/focal_length;
    float y1 = (yy-cy)*z1/focal_length;    
    cout << x1 << " | " << y1 << " | " << z1 << endl;
    
    
    std::cout << reprojection1.isContinuous() << std::endl;
    */
    
    vector<float> pointcloud1;
    vector<float> pointcloud2;
    
    for(int r = 0; r < reprojection1.rows; r++) {
        // We obtain a pointer to the beginning of row r
        cv::Point3f* ptr = reprojection1.ptr<Point3f>(r);
        for(int c = 0; c < reprojection1.cols; c++) {
            float x = ptr[c].x;
            float y = ptr[c].y;
            float z = ptr[c].z;
            pointcloud1.push_back(x);
            pointcloud1.push_back(y);
            pointcloud1.push_back(z);
        }
    }

    for(int r = 0; r < reprojection2.rows; r++) {
        // We obtain a pointer to the beginning of row r
        cv::Point3f* ptr = reprojection2.ptr<Point3f>(r);
        for(int c = 0; c < reprojection2.cols; c++) {
            float x = ptr[c].x;
            float y = ptr[c].y;
            float z = ptr[c].z;
            pointcloud2.push_back(x);
            pointcloud2.push_back(y);
            pointcloud2.push_back(z);
        }
    }
    Mat m1 = Mat(pointcloud1.size()/3, 1, CV_32FC3);
    //cout << m1.size() << endl;
    memcpy(m1.data, pointcloud1.data(), pointcloud1.size()*sizeof(float)); 
    
    Mat m2 = Mat(pointcloud2.size()/3, 1, CV_32FC3);
    memcpy(m2.data, pointcloud2.data(), pointcloud2.size()*sizeof(float));
    
    vector<Mat> ret;
    ret.push_back(m1);
    ret.push_back(m2);
    
    return ret;
}





