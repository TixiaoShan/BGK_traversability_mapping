#include "utility.h"

class TraversabilityMapping{

private:

    ros::NodeHandle nh;

    std::mutex mtx;

    tf::TransformListener listener;
    tf::StampedTransform transform;

    ros::Subscriber subFilteredGroundCloud;
    ros::Publisher pubElevationCloud;

    pcl::PointCloud<PointType>::Ptr laserCloud;
    pcl::PointCloud<PointType>::Ptr laserCloudElevation;

    int mapArrayCount;
    int **mapArrayInd;
    int **elevationPredictArrayFlag;
    vector<childMap_t*> mapArray;

    PointType robotPoint;

    cv::Mat matCov, matEig, matVec;

    vector<mapCell_t*> observingList;
    vector<mapCell_t*> traversabilityCalculatingList; 


public:
    TraversabilityMapping():
        nh("~"),
        mapArrayCount(0){

        subFilteredGroundCloud = nh.subscribe<sensor_msgs::PointCloud2>("/filtered_pointcloud", 1, &TraversabilityMapping::cloudHandler, this);
        pubElevationCloud = nh.advertise<sensor_msgs::PointCloud2> ("/elevation_pointcloud", 1);

        allocateMemory(); 
    }

    ~TraversabilityMapping(){}

    

    void allocateMemory(){
        // allocate memory for point cloud
        laserCloud.reset(new pcl::PointCloud<PointType>());
        laserCloudElevation.reset(new pcl::PointCloud<PointType>());
        
        // initialize array for cmap
        mapArrayInd = new int*[mapArrayLength];
        for (int i = 0; i < mapArrayLength; ++i)
            mapArrayInd[i] = new int[mapArrayLength];

        for (int i = 0; i < mapArrayLength; ++i)
            for (int j = 0; j < mapArrayLength; ++j)
                mapArrayInd[i][j] = -1;

        // initialize array for predicting elevation sub-maps
        elevationPredictArrayFlag = new int*[mapArrayLength];
        for (int i = 0; i < mapArrayLength; ++i)
            elevationPredictArrayFlag[i] = new int[mapArrayLength];

        for (int i = 0; i < mapArrayLength; ++i)
            for (int j = 0; j < mapArrayLength; ++j)
                elevationPredictArrayFlag[i][j] = false;


        // Matrix Initialization
        matCov = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matEig = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matVec = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        resetParams();
    }

    void resetParams(){

        int listSize = observingList.size();
        for (int i = 0; i < listSize; ++i)
            observingList[i]->observingFlag = false;
        observingList.clear();

        for (int i = 0; i < mapArrayLength; ++i){
            for (int j = 0; j < mapArrayLength; ++j){
                elevationPredictArrayFlag[i][j] = false;
                if (mapArrayInd[i][j] != -1)
                    mapArray[mapArrayInd[i][j]]->clearInferenceData();
            }
        }
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        std::lock_guard<std::mutex> lock(mtx);

        pcl::fromROSMsg(*laserCloudMsg, *laserCloud);

        if (getRobotPosition() == false) return;
        
        elevationMap();

        // for real-time application, the traversability calculation and prediction section (function: traversabilityMap)
        // should be implemented in a separate process using std::thread (use mutex to lock memory for safe operation)
        // for the clarity of the code, that part is omitted

        traversabilityMap();

        publishTraversabilityMap();

        resetParams();
    }

    void elevationMap(){
        int cloudSize = laserCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            elevationMap(&laserCloud->points[i]);
        }

        int listSize = observingList.size();
        for (int i = 0; i < listSize; ++i){
            observingList[i]->observingFlag = true;

            grid_t thisGrid = observingList[i]->grid;
            elevationPredictArrayFlag[thisGrid.cubeX][thisGrid.cubeY] = true;
            elevationPredictArrayFlag[thisGrid.cubeX-1][thisGrid.cubeY] = true;
            elevationPredictArrayFlag[thisGrid.cubeX+1][thisGrid.cubeY] = true;
            elevationPredictArrayFlag[thisGrid.cubeX][thisGrid.cubeY-1] = true;
            elevationPredictArrayFlag[thisGrid.cubeX][thisGrid.cubeY+1] = true;
        }

        for (int i = 0; i < mapArrayLength; ++i){
            for (int j = 0; j < mapArrayLength; ++j){

                if (elevationPredictArrayFlag[i][j] == false)
                    continue;

                if (mapArrayInd[i][j] == -1){
                    childMap_t *thisChildMap = new childMap_t(mapArrayCount, i, j);
                    mapArray.push_back(thisChildMap);
                    mapArrayInd[i][j] = mapArrayCount;
                    ++mapArrayCount;
                }

                vector<float> xTrainVec;
                vector<float> yTrainVec;

                for (int k = -1; k <= 1; ++k){
                    for (int l = -1; l <= 1; ++l){
                        if (std::abs(k) + std::abs(l) == 2)
                            continue;
                        int mapInd = mapArrayInd[i+k][j+l];
                        if (mapInd == -1) continue;
                        mapArray[mapInd]->findTrainingData("elevation");
                        xTrainVec.insert(std::end(xTrainVec), std::begin(mapArray[mapInd]->xTrain), std::end(mapArray[mapInd]->xTrain));
                        yTrainVec.insert(std::end(yTrainVec), std::begin(mapArray[mapInd]->yTrain), std::end(mapArray[mapInd]->yTrain));
                    }
                }

                if (xTrainVec.size() == 0 || yTrainVec.size() == 0)
                    continue;

                Eigen::MatrixXf xTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVec.data(), xTrainVec.size() / 2, 2);
                Eigen::MatrixXf yTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVec.data(), yTrainVec.size(), 1);

                int mapInd = mapArrayInd[i][j];
                mapArray[mapInd]->findTestingData();

                vector<float> xTestVec = mapArray[mapInd]->xTest;
                vector<mapCell_t*> testList = mapArray[mapInd]->testList;

                if (xTestVec.size() == 0)
                    continue;

                Eigen::MatrixXf xTest = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVec.data(), xTestVec.size() / 2, 2);

                Eigen::MatrixXf Ks;
                covSparse(xTest, xTrain, Ks);

                Eigen::MatrixXf ybar = (Ks * yTrain).array();
                Eigen::MatrixXf kbar = Ks.rowwise().sum().array();

                mapArray[mapInd]->Ks = Ks;
                mapArray[mapInd]->kbar = kbar;

                for (int k = 0; k < testList.size(); ++k){

                    if (std::isnan(ybar(k,0)) || std::isnan(kbar(k,0)))
                        continue;

                    if (ybar(k,0) == 0 && kbar(k,0) == 0)
                        continue;

                    mapCell_t* thisCell = testList[k];

                    float alphaElev = thisCell->alphaElev;
                    float betaElev = thisCell->betaElev;
                    float elevationVar = thisCell->elevationVar;
                    alphaElev += ybar(k,0);
                    betaElev  += kbar(k,0) - ybar(k,0);
                    float elevation = alphaElev / (alphaElev + betaElev);
                    elevationVar = 1.0f / (1.0f / elevationVar + kbar(k,0));

                    if (!std::isnan(elevation) && !std::isnan(elevationVar)){

                        thisCell->alphaElev = alphaElev;
                        thisCell->betaElev = betaElev;

                        thisCell->updateElevation(elevation);
                        thisCell->updateElevationVar(elevationVar);
                    }
                }
            }
        }
    }

    bool elevationMap(PointType *point){

        grid_t thisGrid;
        if (findPointGridInMap(&thisGrid, point) == false)
            return false;

        mapCell_t *thisCell = grid2Cell(&thisGrid);
        observingList.push_back(thisCell);

        if (thisCell->observedFlag == false){
        	thisCell->observedFlag = true; 
            thisCell->updateElevation(point->z);
        }else{
            thisCell->updateElevation(std::max(point->z, thisCell->elevation));
        }

        if (point->intensity == 1)
            thisCell->updateOccupancy(1);
        thisCell->observedNonTraversableFlag = (point->intensity == 1) ? true : false;

        return true;
    }

    mapCell_t* grid2Cell(grid_t *thisGrid){
        return mapArray[mapArrayInd[thisGrid->cubeX][thisGrid->cubeY]]->cellArray[thisGrid->gridX][thisGrid->gridY];
    }

    bool findPointGridInMap(grid_t *gridOut, PointType *point){

        grid_t thisGrid;
        getPointCubeIndex(&thisGrid.cubeX, &thisGrid.cubeY, point);

        if (thisGrid.cubeX >= 0 && thisGrid.cubeX < mapArrayLength && 
            thisGrid.cubeY >= 0 && thisGrid.cubeY < mapArrayLength){

            if (mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] == -1){
                childMap_t *thisChildMap = new childMap_t(mapArrayCount, thisGrid.cubeX, thisGrid.cubeY);
                mapArray.push_back(thisChildMap);
                mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] = mapArrayCount;
                ++mapArrayCount;
            }
        }else{
            ROS_ERROR("Point cloud is out of elevation map boundary. Increase params ->mapArrayLength<-");
            return false;
        }

        thisGrid.mapID = mapArrayInd[thisGrid.cubeX][thisGrid.cubeY];

        thisGrid.gridX = (int)((point->x - mapArray[thisGrid.mapID]->originX) / mapResolution);
        thisGrid.gridY = (int)((point->y - mapArray[thisGrid.mapID]->originY) / mapResolution);
        if (thisGrid.gridX < 0 || thisGrid.gridY < 0 || thisGrid.gridX >= mapCubeArrayLength || thisGrid.gridY >= mapCubeArrayLength)
            return false;

        *gridOut = thisGrid;
        return true;
    }

    void getPointCubeIndex(int *cubeX, int *cubeY, PointType *point){
        *cubeX = int((point->x + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        *cubeY = int((point->y + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;

        if (point->x + mapCubeLength/2.0 < 0)  --*cubeX;
        if (point->y + mapCubeLength/2.0 < 0)  --*cubeY;
    }




    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const {
        d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
        for (int i = 0; i < xStar.rows(); ++i) {
            d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
        }
    }

    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const {
        dist(xStar/(1.0+0.1), xTrain/(1.0+0.1), Kxz);
        Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
              (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * 1.0f;
        for (int i = 0; i < Kxz.rows(); ++i)
            for (int j = 0; j < Kxz.cols(); ++j)
                if (Kxz(i,j) < 0) Kxz(i,j) = 0;
    }

    void publishTraversabilityMap(){

        if (pubElevationCloud.getNumSubscribers() == 0)
            return;

        int currentCubeX = int((robotPoint.x + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        int currentCubeY = int((robotPoint.y + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;

        if (robotPoint.x + mapCubeLength/2.0 < 0)  --currentCubeX;
        if (robotPoint.y + mapCubeLength/2.0 < 0)  --currentCubeY;

        for (int i = -200 + currentCubeX; i <= 200 + currentCubeX; ++i){
            for (int j = -200 + currentCubeY; j <= 200 + currentCubeY; ++j){

                if (mapArrayInd[i][j] == -1) continue;

                *laserCloudElevation += mapArray[mapArrayInd[i][j]]->cloud;
            }
        }

        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*laserCloudElevation, laserCloudTemp);
        laserCloudTemp.header.frame_id = "/map";
        laserCloudTemp.header.stamp = ros::Time::now();
        pubElevationCloud.publish(laserCloudTemp);

        laserCloudElevation->clear();
    }

    bool getRobotPosition(){
        try{listener.lookupTransform("map","base_link", ros::Time(0), transform); } 
        catch (tf::TransformException ex){ ROS_ERROR("Transfrom Failure."); return false; }

        robotPoint.x = transform.getOrigin().x();
        robotPoint.y = transform.getOrigin().y();
        robotPoint.z = transform.getOrigin().z();

        return true;
    }



    void traversabilityMap(){

        traversabilityCalculatingList = observingList;

        int listSize = traversabilityCalculatingList.size();
        if (listSize == 0)
            return;

        for (int i = 0; i < listSize; ++i){

            mapCell_t *thisCell = traversabilityCalculatingList[i];

            if (thisCell->observedNonTraversableFlag == true){
                thisCell->updateOccupancy(1);
                continue;
            }
            
            vector<float> xyzVector = findNeighborElevations(thisCell);

            if (xyzVector.size() <= 2)
                continue;

            Eigen::MatrixXf matPoints = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xyzVector.data(), xyzVector.size() / 3, 3);

            float minElevation = matPoints.col(2).minCoeff();
            float maxElevation = matPoints.col(2).maxCoeff();
            float maximumStep = maxElevation - minElevation;

            Eigen::MatrixXf centered = matPoints.rowwise() - matPoints.colwise().mean();
            Eigen::MatrixXf cov = (centered.adjoint() * centered);
            cv::eigen2cv(cov, matCov);
            cv::eigen(matCov, matEig, matVec);

            float slopeAngle = std::acos(std::abs(matVec.at<float>(2, 2))) / M_PI * 180;

            if (std::isnan(slopeAngle))
                continue;

            float v = 0;

            float v1 = 1.0f / (1.0f + exp(-(slopeAngle - 30.0)));
            float v2 = maximumStep / 2.0;
            float v3 = std::sqrt(cov(2,2) )/ 3.0;

            if (v1 > 1 || v2 > 1 || v3 > 1)
                v = 1;
            else
                v = std::min(0.9 * v1 + 0.105 * v2 + 0.05 * v3, 1.0);

            thisCell->updateOccupancy(v);
        }

        for (int i = 0; i < mapArrayLength; ++i){
            for (int j = 0; j < mapArrayLength; ++j){

                if (elevationPredictArrayFlag[i][j] == false)
                    continue;

                if (mapArrayInd[i][j] == -1) continue;

                vector<float> yTrainVec;

                for (int k = -1; k <= 1; ++k){
                    for (int l = -1; l <= 1; ++l){
                        if (std::abs(k) + std::abs(l) == 2)
                            continue;
                        int mapInd = mapArrayInd[i+k][j+l];
                        if (mapInd == -1) continue;
                        mapArray[mapInd]->findTrainingData("traversability");
                        yTrainVec.insert(std::end(yTrainVec), std::begin(mapArray[mapInd]->yTrain2), std::end(mapArray[mapInd]->yTrain2));
                    }
                }

                if (yTrainVec.size() == 0)
                    continue;

                Eigen::MatrixXf yTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVec.data(), yTrainVec.size(), 1);

                int mapInd = mapArrayInd[i][j];

                if (mapArray[mapInd]->xTest.size() == 0)
                    continue;

                Eigen::MatrixXf Ks = mapArray[mapInd]->Ks;

                Eigen::MatrixXf ybar = (Ks * yTrain).array();
                Eigen::MatrixXf kbar = mapArray[mapInd]->kbar;

                vector<mapCell_t*> testList = mapArray[mapInd]->testList;

                for (int k = 0; k < testList.size(); ++k){

                    if (std::isnan(ybar(k,0)) || std::isnan(kbar(k,0)))
                        continue;

                    if (ybar(k,0) == 0 && kbar(k,0) == 0)
                        continue;

                    mapCell_t* thisCell = testList[k];

                    float alphaOccu = thisCell->alphaOccu;
                    float betaOccu = thisCell->betaOccu;
                    alphaOccu += ybar(k,0);
                    betaOccu  += kbar(k,0) - ybar(k,0);
                    float occupancy = alphaOccu / (alphaOccu + betaOccu);
                    float occupancyVar = (alphaOccu*betaOccu)/((alphaOccu+betaOccu)*(alphaOccu+betaOccu));


                    if (!std::isnan(occupancy) && !std::isnan(occupancyVar)){
                        thisCell->alphaOccu = alphaOccu;
                        thisCell->betaOccu = betaOccu;

                        thisCell->updateOccupancy(occupancy);
                        thisCell->updateOccupancyVar(occupancyVar);
                    }
                }
            }
        }

        listSize = traversabilityCalculatingList.size();
        for (int i = 0; i < listSize; ++i){
            traversabilityCalculatingList[i]->observedNonTraversableFlag = false;
        }
        traversabilityCalculatingList.clear();
    }

    vector<float> findNeighborElevations(mapCell_t *centerCell){

        vector<float> xyzVector;

        grid_t centerGrid = centerCell->grid;
        grid_t thisGrid;

        for (int k = -3; k <= 3; ++k){
            for (int l = -3; l <= 3; ++l){

                if (std::sqrt(float(k*k + l*l)) * mapResolution > 0.3)
                    continue;

                thisGrid.cubeX = centerGrid.cubeX;
                thisGrid.cubeY = centerGrid.cubeY;
                thisGrid.gridX = centerGrid.gridX + k;
                thisGrid.gridY = centerGrid.gridY + l;

                if(thisGrid.gridX < 0){ --thisGrid.cubeX; thisGrid.gridX = thisGrid.gridX + mapCubeArrayLength;
                }else if(thisGrid.gridX >= mapCubeArrayLength){ ++thisGrid.cubeX; thisGrid.gridX = thisGrid.gridX - mapCubeArrayLength; }
                if(thisGrid.gridY < 0){ --thisGrid.cubeY; thisGrid.gridY = thisGrid.gridY + mapCubeArrayLength;
                }else if(thisGrid.gridY >= mapCubeArrayLength){ ++thisGrid.cubeY; thisGrid.gridY = thisGrid.gridY - mapCubeArrayLength; }

                int mapInd = mapArrayInd[thisGrid.cubeX][thisGrid.cubeY];
                if (mapInd == -1) continue;

                mapCell_t *thisCell = grid2Cell(&thisGrid);

                if (thisCell->elevation != -FLT_MAX){
                    xyzVector.push_back(thisCell->xyz->x);
                    xyzVector.push_back(thisCell->xyz->y);
                    xyzVector.push_back(thisCell->xyz->z);
                }
            }
        }
        return xyzVector;
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "traversability_mapping");
    
    TraversabilityMapping tMapping;

    ROS_INFO("\033[1;32m---->\033[0m Traversability Mapping Started.");

    ros::spin();

    return 0;
}