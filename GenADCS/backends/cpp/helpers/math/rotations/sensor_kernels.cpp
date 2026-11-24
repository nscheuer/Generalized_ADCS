#include "sensor_kernels.hpp"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <string>
namespace py = pybind11;
namespace omni::hardware::sensors::kernels {
template <std::size_t N>
std::array<double, N> read_vector(py::handle value, const char* name) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 1 || array.shape(0) != static_cast<py::ssize_t>(N))
        throw py::value_error(std::string(name) + " has an invalid shape");
    std::array<double, N> result{};
    auto input = array.unchecked<1>();
    for (std::size_t i = 0; i < N; ++i) result[i] = input(static_cast<py::ssize_t>(i));
    return result;
}
M read_matrix(py::handle value, const char* name) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3)
        throw py::value_error(std::string(name) + " must be a 3x3 matrix");
    M result{}; auto input = array.unchecked<2>();
    for (py::ssize_t i=0; i<3; ++i) for (py::ssize_t j=0; j<3; ++j)
        result[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = input(i,j);
    return result;
}
Q normalize(Q q) {
    double norm2=0.0; for (double x:q) norm2 += x*x;
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) throw py::value_error("quaternion must have nonzero finite norm");
    for (double& x:q) x /= std::sqrt(norm2);
    return q;
}
Q multiply(const Q& a, const Q& b) {
    return {{a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
             a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
             a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
             a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]}};
}
using EM = Eigen::Matrix<double,3,3,Eigen::RowMajor>;
using EV = Eigen::Vector3d;
EM em(const M& a) { return Eigen::Map<const EM>(a[0].data()); }
M matrix(const EM& a) { M r{}; Eigen::Map<EM>(r[0].data())=a; return r; }
EV ev(const V& v) { return Eigen::Map<const EV>(v.data()); }
V vector(const EV& v) { V r{}; Eigen::Map<EV>(r.data())=v; return r; }
M identity() { return matrix(EM::Identity()); }
M quaternion_matrix(const Q& q) {
    Eigen::Quaterniond rotation(q[0],q[1],q[2],q[3]);
    return matrix(rotation.toRotationMatrix());
}
V multiply(const M& a, const V& v) { return vector(EV(em(a)*ev(v))); }
M multiply(const M& a, const M& b) { return matrix(EM(em(a)*em(b))); }
M transpose(const M& a) { return matrix(EM(em(a).transpose())); }
M add(const M& a, const M& b) { return matrix(EM(em(a)+em(b))); }
M scale(double s, const M& a) { return matrix(EM(s*em(a))); }
M skew(const V& v) { EM a; a<<0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0; return matrix(a); }
M outer(const V& a, const V& b) { return matrix(EM(ev(a)*ev(b).transpose())); }
M cholesky(const M& a) {
    Eigen::LLT<EM> factor(em(a));
    if (factor.info()!=Eigen::Success) throw py::value_error("noise covariance must be positive definite");
    return matrix(EM(factor.matrixL()));
}
M inverse_from_cholesky(const M& l) {
    const EM lower=em(l);
    return matrix(EM(lower.transpose().triangularView<Eigen::Upper>().solve(
        lower.triangularView<Eigen::Lower>().solve(EM::Identity()))));
}
py::array_t<double> array_vector(const V& v) {
    py::array_t<double> a(3); auto out=a.mutable_unchecked<1>(); for(py::ssize_t i=0;i<3;++i) out(i)=v[static_cast<std::size_t>(i)]; return a;
}
py::array_t<double> array_matrix(const M& m) {
    py::array_t<double> a({3,3}); auto out=a.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < 3; ++i) {
        for (py::ssize_t j = 0; j < 3; ++j)
            out(i,j) = m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
    return a;
}
Q rotation_vector_quaternion(const V& v) {
    const double angle=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if(angle==0.0) return {{1,0,0,0}};
    const double factor=std::sin(0.5*angle)/angle;
    return {{std::cos(0.5*angle),factor*v[0],factor*v[1],factor*v[2]}};
}
V read_vector3(py::handle value, const char* name) { return read_vector<3>(value, name); }
Q read_quaternion(py::handle value, const char* name) { return read_vector<4>(value, name); }
V rotate_inertial_to_sensor(const V& field, const Q& attitude, const Q& orientation, const Q& misalignment) {
    return multiply(quaternion_matrix(multiply(misalignment, orientation)),
                    multiply(quaternion_matrix(normalize(attitude)), field));
}
V rotate_body_to_sensor(const V& body, const Q& orientation, const Q& misalignment) {
    return multiply(quaternion_matrix(multiply(misalignment, orientation)), body);
}
M attitude_vector_jacobian_quaternion(const V& vector, const Q& delta) {
    const Q q=normalize(delta); const V rotated=multiply(quaternion_matrix(q), vector);
    const V v{{q[1],q[2],q[3]}};
    return scale(-2.0,multiply(skew(rotated),add(scale(q[0],identity()),
        add(skew(v),scale(1.0/q[0],outer(v,v))))));
}
M attitude_vector_jacobian_mrp(const V& vector, const V& delta) {
    const double square=delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2], den=1.0+square;
    const Q q{{(1.0-square)/den,2*delta[0]/den,2*delta[1]/den,2*delta[2]/den}};
    const V rotated=multiply(quaternion_matrix(q),vector);
    const M term=add(scale(1.0-square,identity()),add(scale(2.0,skew(delta)),scale(2.0,outer(delta,delta))));
    return scale(-4.0/(den*den),multiply(skew(rotated),term));
}
M attitude_vector_jacobian_angle(const V& vector, const V& delta) {
    const double theta=std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
    const V rotated=multiply(quaternion_matrix(rotation_vector_quaternion(delta)),vector);
    const M cross=skew(delta); double a,b;
    if(theta<1e-5) { a=0.5-theta*theta/24.0; b=1.0/6.0-theta*theta/120.0; }
    else { a=(1.0-std::cos(theta))/(theta*theta); b=(theta-std::sin(theta))/(theta*theta*theta); }
    return scale(-1.0,multiply(skew(rotated),add(identity(),add(scale(a,cross),scale(b,multiply(cross,cross))))));
}
}
