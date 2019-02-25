#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "3d_frontiers.h"
#define ROW 10
#define COL 10
#define DEP 10

bool FrontierDetection::is_Frontier(const int &i, const int &j, const int&k){
    this->neighbours << this->map_3d(i-1, j, k), this->map_3d(i+1, j, k), this->map_3d(i, j-1, k),
            this->map_3d(i, j+1, k), this->map_3d(i, j, k-1), this->map_3d(i, j, k+1);
    return (this->neighbours.array() == 1).any();
}

// A function to check if a given cell (row, col) can be included in DFS
int FrontierDetection::isSafe(const Eigen::Tensor<bool, 3> &M,const int &row,const int &col,const int &dep)
{
    // row number is in range, column number is in range and value is 1
    // and not yet visited
    return (row >= 0) && (row < ROW) &&
           (col >= 0) && (col < COL) &&
           (dep >= 0) && (dep < DEP) &&
           (M(row, col, dep) && !this->visited(row, col, dep));
}

void FrontierDetection::DFS(const Eigen::Tensor<bool, 3> &M, const int &row, const int &col, const int &dep)
{
    // These arrays are used to get row and column numbers of 8 neighbours
    // of a given cell
    static int rowNbr[] = {-1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, 0, -1, -1, -1,  0, 0,  1, 1, 1, 0};
    static int colNbr[] = {-1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0};
    static int depNbr[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,-1, -1, -1, -1, -1, -1, -1, -1};
    // Mark this cell as visited
    this->visited(row, col, dep) = true;

    // Recur for all connected neighbours
    for (int k = 0; k < 8; ++k)
        if (isSafe(M, row + rowNbr[k], col + colNbr[k], dep + depNbr[k]) )
            DFS(M, row + rowNbr[k], col + colNbr[k],dep + depNbr[k]);
}

int FrontierDetection::countIslands(const Eigen::Tensor<bool, 3> &M)
{
    // Make a bool array to mark visited cells.
    // Initially all cells are unvisited
//    Eigen::Tensor<bool, 3> visited(ROW,COL,DEP);
    this->visited.setZero();

    // Initialize count as 0 and travese through the all cells of
    // given matrix
    int count = 0;
    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < COL; ++j)
            for (int k = 0; k < DEP; ++k)
            if (M(i, j, k) && !this->visited(i, j, k)) // If a cell with value 1 is not
            {                         // visited yet, then new island found
                DFS(M, i, j, k);     // Visit all cells in this island.
                ++count;                   // and increment island count
            }

    return count;
}




void NaiveActiveAreaFrontierDetection(const Eigen::Tensor<int, 3> &map_3d){
    FrontierDetection fd(map_3d);
    Eigen::Tensor<int, 3>::Dimensions dims = map_3d.dimensions();

    for (int i = 0; i < dims[0]; i++){
        for (int j = 0; j < dims[1]; j++){
            for (int k =0; k < dims[2]; k++){

                if(map_3d(i, j, k) == 0){

                    if(fd.is_Frontier(i, j, k)){
                        fd.frontier_map(i, j, k) = true;
                    }

//                    if(fd.frontier_map(i, j, k)){
//                        // Remove Frontier
//                        fd.frontier_map(i, j, k) = false;
//                        continue;
//                    }

                }
                else{
                    continue;
                }

            }
        }
    }
//    Eigen::Tensor<bool, 3> frontier_map;
    int numberIslands = fd.countIslands(fd.frontier_map);
    std::cout << numberIslands << std::endl;
//    std::cout << fd.frontier_map << std::endl;
//    std::cout << map_3d;
}


int main()
{
//////////////////////////////////////////////////////////////////////////////////////
////////////////               TEST MAP DEFINITION                   /////////////////
//////////////////////////////////////////////////////////////////////////////////////

    Eigen::Tensor<int, 3> t_3d(10, 10, 10);

    t_3d = t_3d.constant(0);
    Eigen::array<int, 3> offsets1 = {0, 0, 0};
    Eigen::array<int, 3> offsets2 = {8, 0, 0};
    Eigen::array<int, 3> offsets3 = {0, 0, 0};
    Eigen::array<int, 3> offsets4 = {0, 8, 0};
    Eigen::array<int, 3> offsets5 = {2, 2, 0};
    Eigen::array<int, 3> offsets6 = {2, 2, 9};
    Eigen::array<int, 3> extents1 = {2, 10, 10};
    Eigen::array<int, 3> extents2 = {10, 2, 10};
    Eigen::array<int, 3> extents3 = {6, 6, 1};

    Eigen::Tensor<int, 3> slice1 = t_3d.slice(offsets1, extents1).setConstant(2);
    Eigen::Tensor<int, 3> slice2 = t_3d.slice(offsets2, extents1).setConstant(2);
    Eigen::Tensor<int, 3> slice3 = t_3d.slice(offsets3, extents2).setConstant(2);
    Eigen::Tensor<int, 3> slice4 = t_3d.slice(offsets4, extents2).setConstant(2);

    Eigen::Tensor<int, 3> slice5 = t_3d.slice(offsets5, extents3).setConstant(1);
    Eigen::Tensor<int, 3> slice6 = t_3d.slice(offsets6, extents3).setConstant(1);

    // 3D Map pointed by TensorMap
    Eigen::TensorMap<Eigen::Tensor<int, 3>> map_3d(t_3d.data(), 10, 10, 10);

    Eigen::array<int, 3> offsets = {0, 0, 0};
    Eigen::array<int, 3> extents = {10, 10, 10};
    Eigen::Tensor<int, 3> map = map_3d.slice(offsets, extents);
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
    NaiveActiveAreaFrontierDetection(map);

}


