 #include "pose.h"
 
 
 retPose getPose(Mat m1, Mat m2, const std::string& method) {
    Rot3 R;
    Point3 t;
    if(method == "procrustes") {
        Procrustes proc(false, false);
        proc.procrustes(m1, m2);
        //cout << proc.error << endl;
        Mat Rr = proc.rotation ;
        Mat Tr = proc.translation;
        Rot3 R1(
            Rr.at<float>(0,0),
            Rr.at<float>(0,1),
            Rr.at<float>(0,2),
    
            Rr.at<float>(1,0),
            Rr.at<float>(1,1),
            Rr.at<float>(1,2),
    
            Rr.at<float>(2,0),
            Rr.at<float>(2,1),
            Rr.at<float>(2,2)
        );
        
        t(0) = Tr.at<float>(0);
        t(1) = Tr.at<float>(1);
        t(2) = Tr.at<float>(2);
        R = R1;
    }
    
    else if(method == "icp") {   
        typedef PointMatcher<float> PM;
        typedef PM::DataPoints DP;    
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
        const DP ref = create_datapoints(m1);
        const DP data = create_datapoints(m2);
        //icp.setDefault();
        PM::TransformationParameters T = icp(data, ref);
        Rot3 R1(
            T(0,0),
            T(0,1),
            T(0,2),
    
            T(1,0),
            T(1,1),
            T(1,2),
    
            T(2,0),
            T(2,1),
            T(2,2) 
        );
        
        t(0) = T(0,3);
        t(1) = T(1,3);
        t(2) = T(2,3);
        R = R1;     
    }
    retPose s;
    s.R = R;
    s.t = t;
    //Mat m2_transformed = matrix_transform<float>(m2, Rr, Tr);

    //save_vtk<float>(m2_transformed, "m2_transformed.vtk");
    //save_vtk<float>(m2, "m2.vtk"); 
    //save_vtk<float>(m1, "m1.vtk"); 
        
    return s;
}