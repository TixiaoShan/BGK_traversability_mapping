#ifndef _UTILITY_TV_H_
#define _UTILITY_TV_H_


#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <pcl_ros/transforms.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <array> // c++11
#include <thread> // c++11
#include <mutex> // c++11

using namespace std;

typedef pcl::PointXYZI  PointType;

/*
    Mapping Configuration
    */
extern const float mapResolution = 0.1;
extern const float mapCubeLength = 1.0;
extern const int mapCubeArrayLength = 10;
extern const int mapArrayLength = 2000;
extern const int rootCubeIndex = 1000;

struct grid_t;
struct mapCell_t;
struct childMap_t;
struct state_t;
struct neighbor_t;

struct grid_t{
    int mapID;
    int cubeX;
    int cubeY;
    int gridX;
    int gridY;
    int gridIndex;
};

struct mapCell_t{

    PointType *xyz;

    grid_t grid;

    size_t times;
    
    float occupancy;
    float elevation;

    float occupancyVar;
    float elevationVar;

    float alphaOccu, betaOccu;
    float alphaElev, betaElev;

    bool observingFlag;
    bool observedFlag;
    bool observedNonTraversableFlag;


    mapCell_t(){

        times = 0;

        occupancy = 0;
        elevation = -FLT_MAX;

        occupancyVar = 0;
        elevationVar = 1;

        alphaOccu = 1e-3; betaOccu = 1e-3;
        alphaElev = 0; betaElev = 0;
        
        observingFlag = false;
        observedFlag = false;
        observedNonTraversableFlag = false;
    }

    void updatePoint(){
        xyz->z = elevation;
        xyz->intensity = occupancy;
    }

    void updateElevation(float elevIn){
        elevation = elevIn;
        updatePoint();
    }
    void updateOccupancy(float occupIn){
        occupancy = occupIn;
        updatePoint();
    }

    void updateElevationVar(float var){
        elevationVar = var;
    }

    void updateOccupancyVar(float var){
        occupancyVar = var;
    }
};


struct childMap_t{

    vector<vector<mapCell_t*> > cellArray;
    int subInd;
    int indX;
    int indY;
    float originX;
    float originY;
    pcl::PointCloud<PointType> cloud;

    Eigen::MatrixXf Ks;
    Eigen::MatrixXf kbar;

    childMap_t(int id, int indx, int indy){

        subInd = id;
        indX = indx;
        indY = indy;
        originX = (indX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;
        originY = (indY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;

        cellArray.resize(mapCubeArrayLength);
        for (int i = 0; i < mapCubeArrayLength; ++i)
            cellArray[i].resize(mapCubeArrayLength);

        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j)
                cellArray[i][j] = new mapCell_t;

        cloud.points.resize(mapCubeArrayLength*mapCubeArrayLength);

        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j)
                cellArray[i][j]->xyz = &cloud.points[i + j*mapCubeArrayLength];

        for (int i = 0; i < mapCubeArrayLength; ++i){
            for (int j = 0; j < mapCubeArrayLength; ++j){
                
                int index = i + j * mapCubeArrayLength;
                cloud.points[index].x = originX + i * mapResolution;
                cloud.points[index].y = originY + j * mapResolution;
                cloud.points[index].z = std::numeric_limits<float>::quiet_NaN();
                cloud.points[index].intensity = cellArray[i][j]->occupancy;

                cellArray[i][j]->grid.mapID = subInd;
                cellArray[i][j]->grid.cubeX = indX;
                cellArray[i][j]->grid.cubeY = indY;
                cellArray[i][j]->grid.gridX = i;
                cellArray[i][j]->grid.gridY = j;
                cellArray[i][j]->grid.gridIndex = index;
            }
        }
    }

    vector<float> xTrain;
    vector<float> yTrain;
    vector<float> yTrain2;
    vector<float> xTest;
    vector<mapCell_t*> trainList;
    vector<mapCell_t*> testList;

    void findTrainingData(std::string inferType){
        if (inferType == "elevation"){
            if (xTrain.size() != 0 || yTrain.size() != 0)
                return;

            for (int i = 0; i < mapCubeArrayLength; ++i){
                for (int j = 0; j < mapCubeArrayLength; ++j){
                    if (cellArray[i][j]->observingFlag == true){
                        xTrain.push_back(cellArray[i][j]->xyz->x);
                        xTrain.push_back(cellArray[i][j]->xyz->y);
                        yTrain.push_back(cellArray[i][j]->elevation);
                        trainList.push_back(cellArray[i][j]);
                    }
                }
            }
        }
        if (inferType == "traversability"){
            if (yTrain2.size() != 0)
                return;
            int listSize = trainList.size();
            for (int i = 0; i < listSize; ++i){
                yTrain2.push_back(trainList[i]->occupancy);
            }
        }
    }

    void findTestingData(){
        if (xTest.size() != 0 || testList.size() != 0)
            return;
        for (int i = 0; i < mapCubeArrayLength; ++i){
            for (int j = 0; j < mapCubeArrayLength; ++j){
                if (cellArray[i][j]->observedFlag == false){
                    xTest.push_back(cellArray[i][j]->xyz->x);
                    xTest.push_back(cellArray[i][j]->xyz->y);
                    testList.push_back(cellArray[i][j]);
                }
            }
        }
    }

    void clearInferenceData(){
        xTrain.clear();
        yTrain.clear();
        yTrain2.clear();
        xTest.clear();
        trainList.clear();
        testList.clear();
    }
};

#endif
