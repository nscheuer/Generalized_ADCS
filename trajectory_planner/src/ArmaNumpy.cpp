#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cstring>

using namespace arma;
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

cube numpyToArmaCube(py::array_t<double> npCube) {
  py::buffer_info buf1 = npCube.request();
  double *ptr1 = (double *) buf1.ptr;
  int X = buf1.shape[1];
  int Y = buf1.shape[2];
  int Z = buf1.shape[0];

  cube armaCube(X, Y, Z);
  for (size_t idx = 0; idx < X; idx++) {
    for (size_t idy = 0; idy < Y; idy++) {
      for (size_t idz = 0; idz < Z; idz++) {
        armaCube(idx, idy, idz) = ptr1[(idz*X+ idx)*Y + idy];
      }
    }
  }
  return armaCube;
}

mat numpyToArmaMatrix(py::array_t<double> npMat) {
  py::buffer_info buf1 = npMat.request();
  double *ptr1 = (double *) buf1.ptr;
  int nrows = buf1.shape[0];
  int ncols = buf1.shape[1];

  // Check if array is Fortran-contiguous (column-major) or C-contiguous (row-major)
  // Armadillo uses column-major storage
  bool is_f_contiguous = (buf1.strides[0] == sizeof(double)) &&
                          (buf1.strides[1] == nrows * sizeof(double));

  if (is_f_contiguous) {
    // Fortran-contiguous: can use direct copy (original approach)
    mat armaMat(ptr1, nrows, ncols, /*copy_aux_mem=*/true, /*strict=*/false);
    return armaMat;
  } else {
    // C-contiguous or other: must copy element-by-element
    mat armaMat(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        // Access via strides for generality
        double* elem_ptr = (double*)((char*)ptr1 + i * buf1.strides[0] + j * buf1.strides[1]);
        armaMat(i, j) = *elem_ptr;
      }
    }
    return armaMat;
  }
}

// vec numpyToArmaVector(py::array_t<double> npArr) {
//   py::buffer_info buf1 = npArr.request();
//   vec armaVec = vec(buf1.size);
//   double *ptr1 = (double *) buf1.ptr;
//   for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//       armaVec[idx] = ptr1[idx];
//   return armaVec;
// }
//

// vec numpyToArmaVector(py::array_t<double> npArr) {
//   // cout<<"np "<<npArr<<"\n";
//   py::buffer_info buf1 = npArr.request();
//   double *ptr1 = (double *) buf1.ptr;
//   int X = buf1.size;
//   vec armaVec(X);
//   for (size_t idx = 0; idx < buf1.size; idx++){
//       cout<<idx<<"\n";
//       cout<<ptr1[idx]<<"\n";
//       armaVec(idx) = ptr1[idx];
//   }
//   return armaVec;
// }

vec numpyToArmaVector(py::array_t<double> npArr) {
  // cout<<"np "<<npArr<<"\n";
  py::buffer_info buf1 = npArr.request();
  double *ptr1 =  (double *) buf1.ptr;//(double *) buf1.ptr;
  // auto shape = npArr.shape();
  size_t size = buf1.size;//shape[0];
  // cout<<size<<"\n";
  vec armaVec(ptr1, size);
  return armaVec;//arma::vec vec(data, size, false, true); // false means don't copy data, true means column-wise

}

py::array_t<double> armaVectorToNumpy(vec armaVec) {
  auto result = py::array_t<double>(armaVec.n_rows);
  py::buffer_info buf3 = result.request();
  double *ptr3 =  (double *)  buf3.ptr;

  for(size_t idx = 0; idx < armaVec.n_rows; idx++) {
      ptr3[idx] = armaVec(idx);
  }
  return result;
}

py::array_t<double> armaMatrixToNumpy(mat armaMat) {
  // Create numpy array with Fortran (column-major) order to match Armadillo's storage
  // This ensures consistency with numpyToArmaMatrix which expects column-major input
  std::vector<ssize_t> shape = {(ssize_t)armaMat.n_rows, (ssize_t)armaMat.n_cols};
  std::vector<ssize_t> strides = {(ssize_t)sizeof(double), (ssize_t)(armaMat.n_rows * sizeof(double))};

  auto result = py::array_t<double>(shape, strides);
  py::buffer_info buf3 = result.request();
  double *ptr3 = (double *) buf3.ptr;

  // Copy data directly from Armadillo's column-major storage
  std::memcpy(ptr3, armaMat.memptr(), armaMat.n_elem * sizeof(double));

  return result;
}

py::array_t<double> armaCubeToNumpy(cube armaCube) {
  int size = armaCube.n_rows*armaCube.n_cols*armaCube.n_slices;
  auto result = py::array_t<double>(size);
  py::buffer_info buf3 = result.request();

  double *ptr3 = (double *) buf3.ptr;

  for (size_t idx = 0; idx < armaCube.n_rows; idx++) {
    for (size_t idy = 0; idy < armaCube.n_cols; idy++) {
      for(size_t idz = 0; idz < armaCube.n_slices; idz++)
      //std::cout<<"Writing i= "<<idx<<", j="<<idy<<"\n";
      ptr3[(idz*armaCube.n_rows+ idx)*armaCube.n_cols + idy] = armaCube(idx, idy, idz);
    }
  }
  // reshape array to match input shape
  //std::cout<<"About to resize\n";
  result.resize({armaCube.n_slices, armaCube.n_rows,armaCube.n_cols});
  //std::cout<<"Time to return\n";
  return result;
}

py::array_t<double> addArmaMat(py::array_t<double> i, py::array_t<double> j) {
  mat mat1 = numpyToArmaMatrix(i);
  mat mat2 = numpyToArmaMatrix(j);
  mat sum = mat1+mat2;
  return armaMatrixToNumpy(sum);
}

py::array_t<double> addArma(py::array_t<double> i, py::array_t<double> j) {
  vec vec1 = numpyToArmaVector(i);
  vec vec2 = numpyToArmaVector(j);
  vec sum = vec1+vec2;
  return armaVectorToNumpy(sum);
}

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}

/*PYBIND11_MODULE(np_arma, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("add_arrays", &addArma, "Add two NumPy arrays");
    m.def("add_matrices", &addArmaMat, "Add two 2d numpy arrays");
}*/
