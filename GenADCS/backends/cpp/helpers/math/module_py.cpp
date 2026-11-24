#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_rotations__parameter_base(py::module_ &module);
void bind_vector(py::module_ &module);
void bind_derivatives(py::module_ &module);
void bind_dynamics(py::module_ &module);
void bind_rotations_quaternion_algebra(py::module_ &module);
void bind_rotations_quaternion_conversions(py::module_ &module);
void bind_rotations_quaternion_interpolation(py::module_ &module);
void bind_rotations_quaternion_quaternion(py::module_ &module);
void bind_rotations_dcm_kernels(py::module_ &module);
void bind_rotations_dcm_dcm(py::module_ &module);
void bind_rotations_mrp_kernels(py::module_ &module);
void bind_rotations_rodrigues_kernels(py::module_ &module);
void bind_rotations_mrp_modified_rodrigues_parameters(py::module_ &module);
void bind_rotations_rodrigues_rodrigues_parameters(py::module_ &module);
void bind_time_conversions(py::module_ &module);
PYBIND11_MODULE(_helpers, module)
{
    module.doc() = "Eigen-backed C++ ADCS Omni mathematical helpers";
    bind_rotations__parameter_base(module);
    bind_vector(module);
    bind_derivatives(module);
    bind_dynamics(module);
    bind_rotations_quaternion_algebra(module);
    bind_rotations_quaternion_conversions(module);
    bind_rotations_quaternion_interpolation(module);
    bind_rotations_quaternion_quaternion(module);
    bind_rotations_dcm_kernels(module);
    bind_rotations_dcm_dcm(module);
    bind_rotations_mrp_kernels(module);
    bind_rotations_rodrigues_kernels(module);
    bind_rotations_mrp_modified_rodrigues_parameters(module);
    bind_rotations_rodrigues_rodrigues_parameters(module);
    bind_time_conversions(module);
}
