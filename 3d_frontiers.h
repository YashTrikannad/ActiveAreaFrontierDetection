// Yash Trikannad
// For Pluto (DARPA SubT Challenge UPenn)
#pragma once
#include <vector>

struct meanF{
    double x;
    double y;
    double z;
};

std::vector<meanF> NaiveActiveAreaFrontierDetection(const Eigen::Tensor<int8_t , 3> &map_3d);