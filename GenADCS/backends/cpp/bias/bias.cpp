#include "bias.hpp"

namespace py = pybind11;

namespace omni::bias {

Bias::Bias(py::object implementation) : implementation(std::move(implementation)) {}

Bias Bias::copy() const { return Bias(implementation.attr("copy")()); }
Bias Bias::broadcast_to(int length, const std::string& owner) const {
    return Bias(implementation.attr("broadcast_to")(length, py::arg("owner") = owner));
}
py::object Bias::process_covariance(double dt, const std::string& form) const {
    return implementation.attr("process_covariance")(dt, py::arg("form") = form);
}
py::object Bias::process_psd(const std::string& form) const {
    return implementation.attr("process_psd")(py::arg("form") = form);
}
py::object Bias::update_bias(double j2000, py::object rng) const {
    return implementation.attr("_update_bias")(j2000, py::arg("rng") = rng);
}
py::object Bias::get_bias(double j2000) const { return implementation.attr("get_bias")(j2000); }
py::object Bias::apply(py::object value, double dt, py::object rng) const {
    return implementation.attr("apply")(value, dt, rng);
}

}  // namespace omni::bias
