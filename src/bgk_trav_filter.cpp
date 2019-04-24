#include "utility.h"

class TraversabilityFilter{
private:

    ros::NodeHandle nh;

    ros::Subscriber subCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr laserCloudOut1;
    pcl::PointCloud<PointType>::Ptr laserCloudOut2;

    tf::TransformListener listener;
    tf::StampedTransform transform;

    ros::Publisher pubCloud;
    ros::Publisher pubLaserScan;

    sensor_msgs::LaserScan laserScan;

    cv::Mat rangeMatrix;

    PointType robotPoint;

    float **minHeight;
    float **maxHeight;
    bool **initFlag;

public:
    TraversabilityFilter():
        nh("~"){

        subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points_registered", 1, &TraversabilityFilter::cloudHandler, this);

        pubCloud = nh.advertise<sensor_msgs::PointCloud2> ("/filtered_pointcloud", 1);
        pubLaserScan = nh.advertise<sensor_msgs::LaserScan> ("/pointcloud_2_laserscan", 1);  

        allocateMemory();

        pointcloud2laserscanInitialization();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudOut1.reset(new pcl::PointCloud<PointType>());
        laserCloudOut2.reset(new pcl::PointCloud<PointType>());

        
        initFlag = new bool*[1000];
        for (int i = 0; i < 1000; ++i)
            initFlag[i] = new bool[1000];

        minHeight = new float*[1000];
        for (int i = 0; i < 1000; ++i)
            minHeight[i] = new float[1000];

        maxHeight = new float*[1000];
        for (int i = 0; i < 1000; ++i)
            maxHeight[i] = new float[1000];

        resetParams();
    }

    void resetParams(){

        laserCloudIn->clear();
        laserCloudOut1->clear();
        laserCloudOut2->clear();

        for (int i = 0; i < 1000; ++i)
            for (int j = 0; j < 1000; ++j)
                initFlag[i][j] = false;
    }


    ~TraversabilityFilter(){}

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        if (getTransformation() == false) return;

        extractPointCloud(laserCloudMsg);

        transformCloud();

        buildHeightMap();

        publishCloud();

        publishLaserScan();

        resetParams();
    }

    bool getTransformation(){

        try{listener.lookupTransform("map","base_link", ros::Time(0), transform); }
        catch (tf::TransformException ex){ ROS_ERROR("Transfrom Failure."); return false; }

        robotPoint.x = transform.getOrigin().x();
        robotPoint.y = transform.getOrigin().y();
        robotPoint.z = transform.getOrigin().z();

        return true;
    }

    void extractPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        pcl::PointCloud<PointType> laserCloudTemp;
        pcl::fromROSMsg(*laserCloudMsg, laserCloudTemp);

        // extract point cloud
        int cloudSize = laserCloudTemp.points.size();
        for (int i = 0; i < cloudSize; ++i){            
            laserCloudIn->push_back(laserCloudTemp.points[i]);
        }
    }

    void transformCloud(){

        laserCloudIn->header.frame_id = "base_link";
        laserCloudIn->header.stamp = 0;

        pcl::PointCloud<PointType> laserCloudTemp;
        pcl_ros::transformPointCloud("map", *laserCloudIn, laserCloudTemp, listener);
        *laserCloudIn = laserCloudTemp;
    }

    void buildHeightMap(){

    	float roundedX = float(int(robotPoint.x * 10.0f)) / 10.0f;
    	float roundedY = float(int(robotPoint.y * 10.0f)) / 10.0f;

        PointType filterHeightMapOriginPoint;
        filterHeightMapOriginPoint.x = roundedX - 50.0;
        filterHeightMapOriginPoint.y = roundedY - 50.0;

        int cloudSize = laserCloudIn->points.size();
        for (int i = 0; i < cloudSize; ++i){

            int idx = (laserCloudIn->points[i].x - filterHeightMapOriginPoint.x) / mapResolution;
            int idy = (laserCloudIn->points[i].y - filterHeightMapOriginPoint.y) / mapResolution;
            // points out of boundry
            if (idx < 0 || idy < 0 || idx >= 1000 || idy >= 1000)
                continue;

            if (initFlag[idx][idy] == false){
                minHeight[idx][idy] = laserCloudIn->points[i].z;
                maxHeight[idx][idy] = laserCloudIn->points[i].z;
                initFlag[idx][idy] = true;
            } else {
                minHeight[idx][idy] = std::min(minHeight[idx][idy], laserCloudIn->points[i].z);
                maxHeight[idx][idy] = std::max(maxHeight[idx][idy], laserCloudIn->points[i].z);
            }
        }


        for (int i = 0; i < 1000; ++i){
            for (int j = 0; j < 1000; ++j){
                
                if (initFlag[i][j] == false)
                    continue;

                PointType thisPoint;
                thisPoint.x = filterHeightMapOriginPoint.x + i * mapResolution + mapResolution / 2.0;
                thisPoint.y = filterHeightMapOriginPoint.y + j * mapResolution + mapResolution / 2.0;
                thisPoint.z = maxHeight[i][j];

                if (maxHeight[i][j] - minHeight[i][j] > 0.05){
                    thisPoint.intensity = 1; // obstacle
                    laserCloudOut1->push_back(thisPoint);
                }else{
                    thisPoint.intensity = 0; // free
                    laserCloudOut2->push_back(thisPoint);
                }
            }
        }
    }

    void publishCloud(){
        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*laserCloudOut1 + *laserCloudOut2, laserCloudTemp);
        laserCloudTemp.header.stamp = ros::Time::now();
        laserCloudTemp.header.frame_id = "map";
        pubCloud.publish(laserCloudTemp);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// Point Cloud to Laser Scan  ///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void pointcloud2laserscanInitialization(){

        laserScan.header.frame_id = "base_link"; // assume laser has the same frame as the robot

        laserScan.angle_min = -M_PI;
        laserScan.angle_max =  M_PI;
        laserScan.angle_increment = 1.0f / 180 * M_PI;
        laserScan.time_increment = 0;

        laserScan.scan_time = 0.1;
        laserScan.range_min = 0.3;
        laserScan.range_max = 100;

        int range_size = std::ceil((laserScan.angle_max - laserScan.angle_min) / laserScan.angle_increment);
        laserScan.ranges.assign(range_size, laserScan.range_max + 1.0);
    }

    void updateLaserScan(){

        int cloudSize = laserCloudOut1->points.size();
        for (int i = 0; i < cloudSize; ++i){
            PointType *point = &laserCloudOut1->points[i];
            float x = point->x - robotPoint.x;
            float y = point->y - robotPoint.y;
            float range = std::sqrt(x*x + y*y);
            float angle = std::atan2(y, x);
            int index = (angle - laserScan.angle_min) / laserScan.angle_increment;
            laserScan.ranges[index] = std::min(laserScan.ranges[index], range);
        } 
    }

    void publishLaserScan(){

        updateLaserScan();

        laserScan.header.stamp = ros::Time::now();
        pubLaserScan.publish(laserScan);
        // initialize laser scan for new scan
        std::fill(laserScan.ranges.begin(), laserScan.ranges.end(), laserScan.range_max + 1.0);
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "traversability_mapping");
    
    TraversabilityFilter TFilter;

    ROS_INFO("\033[1;32m---->\033[0m Traversability Filter Started.");

    ros::spin();

    return 0;
}