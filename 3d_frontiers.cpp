// Yash Trikannad
// For Pluto (DARPA SubT Challenge UPenn)

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "3d_frontiers.h"

// Function to Find the Frontiers
// Adds to Queue all new centroids of frontiers

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
                    // Write Condition to remove Frontiers
                }
                else{
                    continue;
                }

            }
        }
    }

    fd.FindFrontierGroups(fd.frontier_map);
//    std::cout << fd.frontier_map << std::endl;
//    std::cout << map_3d;
}


// Find Frontier Groups among using the Frontier Map predicted in the previous step
void FrontierDetection::FindFrontierGroups(const Eigen::Tensor<bool, 3> &frontier_map)
{
    // Initialize count as 0 and travese through the all cells of
    // given matrix
    int count = 0;
    meanF mean = {0, 0, 0};
    for (int i = 0; i < this->dimX; ++i)
        for (int j = 0; j < this->dimY; ++j)
            for (int k = 0; k < this->dimZ; ++k)
                if (frontier_map(i, j, k) && !this->visited(i, j, k)) // If a cell with value 1 is not
                {
                    this->resetPrevious();
                    DFS(frontier_map, i, j, k);// Visit all cells in this island.
                    ++count;                   // and increment island count
                    mean.x = this->groupx_mean/this->group_count;
                    mean.y = this->groupy_mean/this->group_count;
                    mean.z = this->groupz_mean/this->group_count;
                    this->frontier_queue.push(mean);
                    std::cout << "Row: " <<mean.x << " Column: " << mean.y << " Depth: " << mean.z <<std::endl;
                }
}

bool FrontierDetection::is_Frontier(const int &i, const int &j, const int&k){
    this->neighbours << this->map_3d(i-1, j, k), this->map_3d(i+1, j, k), this->map_3d(i, j-1, k),
            this->map_3d(i, j+1, k), this->map_3d(i, j, k-1), this->map_3d(i, j, k+1);
    return (this->neighbours.array() == 1).any();
}


// A function to check if a given cell (row, col, dep) can be included in DFS
int FrontierDetection::isSafe(const Eigen::Tensor<bool, 3> &frontier_map,const int &row,const int &col,const int &dep)
{
    // Check for Boundary Conditions
    return (row >= 0) && (row < this->dimX) &&
           (col >= 0) && (col < this->dimY) &&
           (dep >= 0) && (dep < this->dimZ) &&
           (frontier_map(row, col, dep) && !this->visited(row, col, dep));
}

// Recursive Function to Mark all Frontier Points for a single group
void FrontierDetection::DFS(const Eigen::Tensor<bool, 3> &frontier_map, const int &row, const int &col, const int &dep)
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
    this->groupx_mean += row;
    this->groupy_mean += col;
    this->groupz_mean += dep;

    // Recur for all connected neighbours
    for (int k = 0; k < 26; ++k)
        if (isSafe(frontier_map, row + rowNbr[k], col + colNbr[k], dep + depNbr[k]) )
            DFS(frontier_map, row + rowNbr[k], col + colNbr[k],dep + depNbr[k]);
}

void FrontierDetection::resetPrevious(){
    this->group_map.setConstant(0);
    this->groupx_mean = 0;
    this->groupy_mean = 0;
    this->groupz_mean = 0;
    this->group_count = 0;
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


