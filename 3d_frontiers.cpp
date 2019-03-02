// Yash Trikannad
// For Pluto (DARPA SubT Challenge UPenn)
// ROS Node to Find the Frontiers in 3D cube
// Subscribes to Voxel Map
// Input - Voxel Map
// Output - Centers of the Frontiers

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Frontiers_3D.h"
#include <pluto_msgs/VoxelMap.h>
#include <chrono>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <cmath>


class FrontierDetection{

private:
    int dimX;
    int dimY;
    int dimZ;
    Eigen::Tensor<int8_t , 3> map_3d;
    Eigen::Matrix<double, 26, 1> neighbours;
    meanFrontier group_mean;
    double group_count;

public:
    Eigen::Tensor<bool, 3> frontier_map;
    Eigen::Tensor<int, 3> group_map;
    std::vector<meanFrontier> frontier_centroids;
    Eigen::Tensor<bool, 3> visited;

    // Constructor
    explicit FrontierDetection(const Eigen::Tensor<int8_t , 3> &map_In): map_3d(map_In)
    {
        Eigen::Tensor<int8_t , 3>::Dimensions dims = map_3d.dimensions();
        dimX = int(dims[0]);
        dimY = int(dims[1]);
        dimZ = int(dims[2]);
        visited = Eigen::Tensor<bool, 3>(dimX,dimY,dimZ);
        visited.setConstant(false);
        frontier_map = Eigen::Tensor<bool, 3>(dimX,dimY,dimZ);
        frontier_map.setConstant(false);
        group_map = Eigen::Tensor<int, 3>(dimX,dimY,dimZ);
        group_map.setConstant(0);
        neighbours.setZero();
        visited.setConstant(false);
        group_mean = {0, 0, 0};
        group_count = 0;
    };

    // Find Frontier Groups among using the Frontier Map predicted in the previous step
    void FindFrontierGroups(const Eigen::Tensor<bool, 3> &frontier_map)
    {
        
        meanFrontier mean = {0, 0, 0};
        for (int i = 0; i < this->dimX; ++i)
            for (int j = 0; j < this->dimY; ++j)
                for (int k = 0; k < this->dimZ; ++k)
                    if (frontier_map(i, j, k) && !this->visited(i, j, k)) // If a cell with value 1 is not
                    {
                        this->resetPrevious();
                        DFS(frontier_map, i, j, k);// Visit all cells in this frontier
                        mean.x = this->group_mean.x/this->group_count;
                        mean.y = this->group_mean.y/this->group_count;
                        mean.z = this->group_mean.z/this->group_count;
                        this->frontier_centroids.push_back(mean);
                        std::cout << "Row: " <<mean.x << " Column: " << mean.y << " Depth: " << mean.z << '\n';
                    }
    }

        // Recursive Function to Mark all Frontier Points for a single group
    void DFS(const Eigen::Tensor<bool, 3> &frontier_map, const int &row, const int &col, const int &dep)
    {
        // These arrays are used to get row and column numbers of 8 neighbours
        // of a given cell
        static int rowNbr[] = {-1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, 0, -1, -1, -1,  0, 0,  1, 1, 1, 0};
        static int colNbr[] = {-1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0};
        static int depNbr[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        // Mark this cell as visited
        this->visited(row, col, dep) = true;

        this->group_map(row, col, dep) = 1;
        this->group_count += 1;
        this->group_mean.x += row;
        this->group_mean.y += col;
        this->group_mean.z += dep;

        // Recur for all connected neighbours
        for (int k = 0; k < 26; ++k)
            if (isSafe(frontier_map, row + rowNbr[k], col + colNbr[k], dep + depNbr[k]) )
                DFS(frontier_map, row + rowNbr[k], col + colNbr[k],dep + depNbr[k]);
    }

    // Resets Previous Group
    void resetPrevious(){
        this->group_map.setConstant(0);
        this->group_mean = {0, 0, 0};
        this->group_count = 0;
    }

    // Checks if a particular gridcell is a Frontier
    bool is_Frontier(const int &i, const int &j, const int&k){
        if ((i > 0) && (i < this->dimX-1) &&
            (j > 0) && (j < this->dimY-1) &&
            (k > 0) && (k < this->dimZ-1)) {
        this->neighbours << this->map_3d(i-1, j-1, k), this->map_3d(i+1 , j-1, k), this->map_3d(i-1, j , k),
            this->map_3d(i+1, j , k), this->map_3d(i-1, j+1, k ), this->map_3d(i+1, j+1, k ), this->map_3d(i , j-1, k),
            this->map_3d(i , j+1, k), this->map_3d(i-1, j-1, k-1), this->map_3d(i+1 , j-1, k-1), this->map_3d(i-1, j , k-1),
            this->map_3d(i+1, j , k-1), this->map_3d(i-1, j+1, k-1 ), this->map_3d(i+1, j+1, k-1 ), this->map_3d(i , j-1, k-1),
            this->map_3d(i , j+1, k-1), this->map_3d(i-1, j-1, k+1), this->map_3d(i+1 , j-1, k+1), this->map_3d(i-1, j , k+1),
            this->map_3d(i+1, j , k+1), this->map_3d(i-1, j+1, k+1 ), this->map_3d(i+1, j+1, k+1 ), this->map_3d(i , j-1, k),
            this->map_3d(i , j+1, k+1) ,this->map_3d(i , j, k+1), this->map_3d(i, j , k-1);
        return (this->neighbours.array() == 100).any();
        }
        else return false;
    }


    // A function to check if a given cell (row, col, dep) can be included in DFS
    int isSafe(const Eigen::Tensor<bool, 3> &frontier_map,const int &row,const int &col,const int &dep)
    {
        // Check for Boundary Conditions
        return (row >= 0) && (row < this->dimX) &&
               (col >= 0) && (col < this->dimY) &&
               (dep >= 0) && (dep < this->dimZ) &&
               (frontier_map(row, col, dep) && !this->visited(row, col, dep));
    }


};
// End of Class



std::vector<meanFrontier> NaiveActiveAreaFrontierDetection(const Eigen::Tensor<int8_t , 3> &map_3d){

    FrontierDetection fd(map_3d);
    Eigen::Tensor<int8_t , 3>::Dimensions dims = map_3d.dimensions();

    // Prepares the Frontier Map
    for (int i = 0; i < dims[0]; i++){
        for (int j = 0; j < dims[1]; j++){
            for (int k =0; k < dims[2]; k++){

                if(map_3d(i, j, k) == 0){

                    if(fd.is_Frontier(i, j, k)){
                        // Update Frontier Map
                        fd.frontier_map(i, j, k) = true;
                    }

                }

            }
        }
    }

    std::cout << " FRONTIER MEANS are: " << '\n';
    // Does Grouping in the Frontier Map
    fd.FindFrontierGroups(fd.frontier_map);

    // Returns vector of type (MeanFrontiers)
    return fd.frontier_centroids;

}



class FrontierClass {

public:

    std::vector<meanFrontier> frontiers;

    FrontierClass(ros::NodeHandle& nh){
    };

    void frontierCallback(pluto_msgs::VoxelMap _voxelMap) {

        // Convert VoxelMap to TensorMap
        Eigen::TensorMap<Eigen::Tensor<int8_t, 3>> data_tensor(_voxelMap.data.data(), _voxelMap.dim.x, _voxelMap.dim.y, _voxelMap.dim.z);

        // Function to detect Frontier
        frontiers = NaiveActiveAreaFrontierDetection(data_tensor);

    }

};


int main(int argc, char **argv) {

    ros::init(argc, argv, "frontier_detector");
    ros::NodeHandle nh("~");

    FrontierClass frontier_detector(nh);

    ros::Rate r(10);

    ros::Subscriber sub = nh.subscribe("/juliett/voxel_map", 1, &FrontierClass::frontierCallback, &frontier_detector);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);


    while (ros::ok())
  {

    visualization_msgs::Marker points;
    points.header.frame_id = "/my_frame";
    points.header.stamp = ros::Time::now();
    points.ns = "Frontier_Mean";
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;

    points.id = 0;

    points.type = visualization_msgs::Marker::POINTS;

    // POINTS markers use x and y scale for width/height respectively
    points.scale.x = 0.2;
    points.scale.y = 0.2;

    // Points are green
    points.color.g = 1.0f;
    points.color.a = 1.0;

    // Create the vertices for the points and lines

    for (f : frontier_detector.frontiers)
    {
        geometry_msgs::Point p;
        p.x = static_cast<float>(std::round(f.x));
        p.y = static_cast<float>(std::round(f.y));
        p.z = static_cast<float>(std::round(f.z)); 
        points.points.push_back(p);
    }


    marker_pub.publish(points);

    r.sleep();
    ros::spinOnce();
   }

  //

    ros::spin();
}



