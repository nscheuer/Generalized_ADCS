#ifndef TPR_ARMANUMPY_HPP
#define TPR_ARMANUMPY_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

using namespace arma;
namespace py = pybind11;

int add(int i, int j);

cube numpyToArmaCube(py::array_t<double> npCube);

mat numpyToArmaMatrix(py::array_t<double> npMat);

vec numpyToArmaVector(py::array_t<double> npArr);

py::array_t<double> armaVectorToNumpy(vec armaVec);

py::array_t<double> armaMatrixToNumpy(mat armaMat);

py::array_t<double> armaCubeToNumpy(cube armaCube);

py::array_t<double> addArmaMat(py::array_t<double> i, py::array_t<double> j);

py::array_t<double> addArma(py::array_t<double> i, py::array_t<double> j);

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2);


#endif // TPR_ARMAJSON_HPP
