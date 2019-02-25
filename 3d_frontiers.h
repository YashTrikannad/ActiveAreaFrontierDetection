#pragma once
#include <queue>

class FrontierDetection{

private:
    struct meanF{
        double x;
        double y;
        double z;
    };
    int dimX;
    int dimY;
    int dimZ;
    Eigen::Tensor<int, 3> map_3d;
    Eigen::Matrix<int, 6, 1> neighbours;
    double groupx_mean;
    double groupy_mean;
    double groupz_mean;
    double group_count;

public:
    Eigen::Tensor<bool, 3> frontier_map;
    Eigen::Tensor<int, 3> group_map;
    std::queue<meanF> frontier_queue;
    Eigen::Tensor<bool, 3> visited;

    // Constructor
    explicit FrontierDetection(const Eigen::Tensor<int, 3> &map_In): map_3d(map_In)
    {
        Eigen::Tensor<int, 3>::Dimensions dims = map_3d.dimensions();
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
        groupx_mean = 0;
        groupy_mean = 0;
        groupz_mean = 0;
        group_count = 0;
    };

    bool is_Frontier(const int &i, const int &j, const int&k);
    int isSafe(const Eigen::Tensor<bool, 3> &,const int &,const int &,const int &);
    void DFS(const Eigen::Tensor<bool, 3> &, const int &, const int &, const int &);
    void FindFrontierGroups(const Eigen::Tensor<bool, 3> &);
    void resetPrevious();

};

void NaiveActiveAreaFrontierDetection(const Eigen::Tensor<int, 3> &map_3d);