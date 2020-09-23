#include "utils.h" 
#include "pointcloud.h" 
#include "optimizer.h"
#include "test_sfm.h"
#include "pose.h"
#include <boost/filesystem.hpp>
#include <boost/iterator/filter_iterator.hpp>
namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
    
    // Keypoint Mapper is a map that takes the landmark i 
    // and returns a map which maps each image j to the coordinate
    // where landmark i appeared in image j
    map<int, map<int, Point2f>> KeypointMapper;
    map<string,int> prevKeypointIndexer;
    map<string,int> currKeypointIndexer;
    std::map<int,vector<Point2f>>::iterator it;
    std::map<string,int>::iterator itr;
               
    vector<string> frames;
    vector<string> frames1;
    vector<Pose3> poses;
    vector<Mat> disparities;
    retPointcloud s;

    fs::path p(data_folder);
    fs::directory_iterator dir_first(p), dir_last;

    auto pred = [](const fs::directory_entry& p)
    {
        return fs::is_regular_file(p);
    };

    
    int count = std::distance(boost::make_filter_iterator(pred, dir_first, dir_last),
                      boost::make_filter_iterator(pred, dir_last, dir_last));

    for(int k=0; k <6; k++) {
        frames.push_back(to_string(k));
    }
    cout << frames.size() << endl;
    
    retFiltering filtered;
    filtered = filterImages(frames);
    
    vector<int> considered_poses = filtered.considered_poses;
    vector<retPointcloud> filteredOutput = filtered.filteredOutput;
    
    //cout << "considered_poses " << considered_poses.size() << endl;
    for(int i=0; i< considered_poses.size(); ++i)
        cout << considered_poses[i] << endl;;
    int pose_id = 1;

    std::cout << "length of considered poses " << considered_poses.size() << std::endl;
    std::cout << "length of filetereoutout " << filteredOutput.size() << std::endl;
    Rot3 R(1, 0, 0, 0, 1, 0, 0, 0, 1);
    Point3 t;
    t(0) = 0;
    t(1) = 0;
    t(2) = 0;
    Pose3 pose(R, t);
    poses.push_back(pose);
    for(size_t i=0; i<considered_poses.size(); ++i) {
        Mat m1 = filteredOutput[i].ret[0];
        Mat m2 = filteredOutput[i].ret[1];
        vector<Point2f> src = filteredOutput[i].src;
        vector<Point2f> dst = filteredOutput[i].dst;
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
                KeypointMapper[l].insert(make_pair(pose_id-1, src[l]));
                KeypointMapper[l].insert(make_pair(pose_id, dst[l]));
                prevKeypointIndexer[getKpKey(dst[l])] = l;
               
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
                itr = prevKeypointIndexer.find(getKpKey(src[l]));
                int landmark_id;
                if ( itr != prevKeypointIndexer.end() ) {
                    landmark_id = itr->second;
                }
                else{
                    int largest_landmark_id = KeypointMapper.rbegin()->first;
                    landmark_id = largest_landmark_id + 1;
                }
                KeypointMapper[landmark_id].insert(make_pair(pose_id, dst[l]));
                currKeypointIndexer[getKpKey(dst[l])] = landmark_id;
            }
                    
            prevKeypointIndexer.clear();
            prevKeypointIndexer = currKeypointIndexer;
        }
        pose_id++;
        cout << pose_id << endl;
    }

    cout << "Populated maps" << endl;
    
    // Run GTSAM bundle adjustment
    gtsam::Values result;
    Cal3_S2::shared_ptr Kgt(new Cal3_S2(focal_length, focal_length, 0 /* skew */, cx, cy));
    ret_optimize ret_optimizer = Optimize(KeypointMapper, Kgt, frames, considered_poses, poses);
    result = ret_optimizer.result;

    // Test GTSAM output  
    //test_sfm(result, Kgt, considered_poses);
    //reconstruct_pointcloud(result, Kgt, considered_poses);
    //gtsam::Values result_reoptimize = Optimize_object_loc(ret_optimizer, considered_poses, Kgt);
    //int num_sift_landmarks = ret_optimizer.landmarks3d.size();
    //result_to_vtk(result_reoptimize, num_sift_landmarks);

    return 0;
}

