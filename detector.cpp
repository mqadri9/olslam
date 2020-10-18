
#include "detector.h"


vector<vector<float>> find_correspondences_surf(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    const float ratio = 0.8f;    
    Ptr<SURF> detector = SURF::create( minHessian );
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    std::vector< std::vector<DMatch> > knn_matches;
    detector->detectAndCompute( img_l, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img_r, noArray(), keypoints2, descriptors2 );
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); ++i)
    {
        //const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
        {
            float x1 = keypoints1[knn_matches[i][0].queryIdx].pt.x;
            float y1 = keypoints1[knn_matches[i][0].queryIdx].pt.y;
            float x2 = keypoints2[knn_matches[i][0].trainIdx].pt.x;
            float y2 = keypoints2[knn_matches[i][0].trainIdx].pt.y;  
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {          
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    Mat img_matches;
    drawMatches( img_l, keypoints1, img_r, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cout <<"Writing image " <<letype << endl;
    if(letype=="stereo") {
        imwrite(root_directory + "/surf_stereo/frame" + to_string(frame_id) + ".jpg", img_matches); 
    } 
    else{
        imwrite(root_directory + "/surf_temporal/frame" + to_string(frame_id) + ".jpg", img_matches); 
    }


    vector<vector<float>> correspondences;
    for (auto &m : knn_matches) {
        if (m[0].distance < ratio * m[1].distance) {
            float x1 = keypoints1[m[0].queryIdx].pt.x;
            float y1 = keypoints1[m[0].queryIdx].pt.y;
            float x2 = keypoints2[m[0].trainIdx].pt.x;
            float y2 = keypoints2[m[0].trainIdx].pt.y;
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {
                correspondences.push_back({x1, y1, x2, y2});
            }
        }
    }
    return correspondences;
}


vector<vector<float>> find_correspondences_sift(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    const float ratio = 0.8f;    
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    std::vector< std::vector<DMatch> > knn_matches;
    detector->detectAndCompute( img_l, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img_r, noArray(), keypoints2, descriptors2 );
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); ++i)
    {
        //const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
        {
            float x1 = keypoints1[knn_matches[i][0].queryIdx].pt.x;
            float y1 = keypoints1[knn_matches[i][0].queryIdx].pt.y;
            float x2 = keypoints2[knn_matches[i][0].trainIdx].pt.x;
            float y2 = keypoints2[knn_matches[i][0].trainIdx].pt.y;  
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {          
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    Mat img_matches;
    drawMatches( img_l, keypoints1, img_r, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cout <<"Writing image " <<letype << endl;
    if(letype=="stereo") {
        imwrite(root_directory + "/sift_stereo/frame" + to_string(frame_id) + ".jpg", img_matches); 
    } 
    else{
        imwrite(root_directory + "/sift_temporal/frame" + to_string(frame_id) + ".jpg", img_matches); 
    }


    vector<vector<float>> correspondences;
    for (auto &m : knn_matches) {
        if (m[0].distance < ratio * m[1].distance) {
            float x1 = keypoints1[m[0].queryIdx].pt.x;
            float y1 = keypoints1[m[0].queryIdx].pt.y;
            float x2 = keypoints2[m[0].trainIdx].pt.x;
            float y2 = keypoints2[m[0].trainIdx].pt.y;
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {
                correspondences.push_back({x1, y1, x2, y2});
            }
        }
    }
    return correspondences;
}

vector<vector<float>> find_correspondences_orb(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    const float ratio = 0.8f;    
    cv::Ptr<cv::xfeatures2d::ORB> detector = cv::xfeatures2d::ORB::create();
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    std::vector< std::vector<DMatch> > knn_matches;
    detector->detectAndCompute( img_l, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img_r, noArray(), keypoints2, descriptors2 );
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); ++i)
    {
        //const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
        {
            float x1 = keypoints1[knn_matches[i][0].queryIdx].pt.x;
            float y1 = keypoints1[knn_matches[i][0].queryIdx].pt.y;
            float x2 = keypoints2[knn_matches[i][0].trainIdx].pt.x;
            float y2 = keypoints2[knn_matches[i][0].trainIdx].pt.y;  
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {          
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    Mat img_matches;
    drawMatches( img_l, keypoints1, img_r, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cout <<"Writing image " <<letype << endl;
    if(letype=="stereo") {
        imwrite(root_directory + "/orb_stereo/frame" + to_string(frame_id) + ".jpg", img_matches); 
    } 
    else{
        imwrite(root_directory + "/orb_temporal/frame" + to_string(frame_id) + ".jpg", img_matches); 
    }


    vector<vector<float>> correspondences;
    for (auto &m : knn_matches) {
        if (m[0].distance < ratio * m[1].distance) {
            float x1 = keypoints1[m[0].queryIdx].pt.x;
            float y1 = keypoints1[m[0].queryIdx].pt.y;
            float x2 = keypoints2[m[0].trainIdx].pt.x;
            float y2 = keypoints2[m[0].trainIdx].pt.y;
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {
                correspondences.push_back({x1, y1, x2, y2});
            }
        }
    }
    return correspondences;
}

vector<vector<float>> find_correspondences_akaze(r left, r right, int img_width, int vertical_threshold, int horizontal_threshold, int cost_threshold, Mat img_l, Mat img_r, int frame_id, string letype) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    int minHessian = 400;
    const float ratio = 0.8f;    
    cv::Ptr<cv::xfeatures2d::AKAZE> detector = cv::xfeatures2d::AKAZE::create();
    Mat descriptors1, descriptors2; 
    Mat d1, d2;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    std::vector< std::vector<DMatch> > knn_matches;
    detector->detectAndCompute( img_l, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img_r, noArray(), keypoints2, descriptors2 );
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    vector<cv::DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); ++i)
    {
        //const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance)
        {
            float x1 = keypoints1[knn_matches[i][0].queryIdx].pt.x;
            float y1 = keypoints1[knn_matches[i][0].queryIdx].pt.y;
            float x2 = keypoints2[knn_matches[i][0].trainIdx].pt.x;
            float y2 = keypoints2[knn_matches[i][0].trainIdx].pt.y;  
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {          
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    Mat img_matches;
    drawMatches( img_l, keypoints1, img_r, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cout <<"Writing image " <<letype << endl;
    if(letype=="stereo") {
        imwrite(root_directory + "/akaze_stereo/frame" + to_string(frame_id) + ".jpg", img_matches); 
    } 
    else{
        imwrite(root_directory + "/akaze_temporal/frame" + to_string(frame_id) + ".jpg", img_matches); 
    }


    vector<vector<float>> correspondences;
    for (auto &m : knn_matches) {
        if (m[0].distance < ratio * m[1].distance) {
            float x1 = keypoints1[m[0].queryIdx].pt.x;
            float y1 = keypoints1[m[0].queryIdx].pt.y;
            float x2 = keypoints2[m[0].trainIdx].pt.x;
            float y2 = keypoints2[m[0].trainIdx].pt.y;
            if(x1 > horizontal_threshold && x2 < img_width - horizontal_threshold &&
               x1 - x2 > 50 && abs(y2 - y1) < vertical_threshold ) {
                correspondences.push_back({x1, y1, x2, y2});
            }
        }
    }
    return correspondences;
}
