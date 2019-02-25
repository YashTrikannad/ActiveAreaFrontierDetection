
#pragma once
#include <queue>

class FrontierDetection{

private:
    Eigen::Tensor<int, 3> map_3d;
    Eigen::Matrix<int, 6, 1> neighbours;
public:
    Eigen::Tensor<bool, 3> frontier_map;
    std::queue<int[3]> frontier_queue;
    Eigen::Tensor<bool, 3> visited;

    // Constructor
    explicit FrontierDetection(const Eigen::Tensor<int, 3> &map_In): map_3d(map_In)
    {
        Eigen::Tensor<int, 3>::Dimensions dims = map_3d.dimensions();
        visited = Eigen::Tensor<bool, 3>(dims[0],dims[1],dims[2]);
        frontier_map = Eigen::Tensor<bool, 3>(dims[0], dims[1], dims[2]);
        frontier_map.setConstant(false);
        neighbours.setZero();
        visited.setConstant(false);
    };

    bool is_Frontier(const int &i, const int &j, const int&k);
    int isSafe(const Eigen::Tensor<bool, 3> &,const int &,const int &,const int &);
    void DFS(const Eigen::Tensor<bool, 3> &, const int &, const int &, const int &);
    int countIslands(const Eigen::Tensor<bool, 3> &);

};

void NaiveActiveAreaFrontierDetection(const Eigen::Tensor<int, 3> &map_3d);