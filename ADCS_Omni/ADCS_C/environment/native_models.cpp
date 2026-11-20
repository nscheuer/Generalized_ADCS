#include "native_models.hpp"
extern "C" {
#include "atmosphere/NRMLSISE00/nrlmsise00_core.h"
}
#include "conversions.hpp"
#include "time/conversions.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omni::environment {
namespace {
constexpr double kJ2000 = 2451545.0;
constexpr double kSecondsPerDay = 86400.0;
constexpr double kMagneticReferenceRadiusKm = 6371.2;
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<int, 12> kTargets{1,2,3,4,5,6,7,8,9,301,10,399};

uint32_t read_u32(const std::vector<unsigned char>& raw, std::size_t offset, bool little) {
    if (offset + 4U > raw.size()) throw std::runtime_error("truncated DAF integer");
    uint32_t value = 0U;
    for (unsigned int i=0; i<4U; ++i) {
        const std::size_t j = little ? i : 3U-i;
        value |= static_cast<uint32_t>(raw[offset+j]) << (8U*i);
    }
    return value;
}

double read_f64(const std::vector<unsigned char>& raw, std::size_t offset, bool little) {
    if (offset + 8U > raw.size()) throw std::runtime_error("truncated DAF double");
    uint64_t bits=0U;
    for (unsigned int i=0; i<8U; ++i) {
        const std::size_t j=little ? i : 7U-i;
        bits |= static_cast<uint64_t>(raw[offset+j]) << (8U*i);
    }
    double value=0.0;
    static_assert(sizeof(value)==sizeof(bits), "unexpected double size");
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

struct HarmonicData { std::vector<double> times; std::vector<Eigen::MatrixXd> g,h; };
HarmonicData read_shc(const std::filesystem::path& path, int nmin, int nmax) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open magnetic coefficient file: "+path.string());
    std::string line;
    do { if (!std::getline(input,line)) throw std::runtime_error("empty SHC file"); }
    while (line.empty() || line[0]=='#');
    int low=0, high=0, count=0, ignored1=0, ignored2=0;
    std::istringstream header(line);
    header >> low >> high >> count >> ignored1 >> ignored2;
    if (!header || low!=nmin || high!=nmax || count<=0) throw std::runtime_error("invalid SHC header");
    do { if (!std::getline(input,line)) throw std::runtime_error("missing SHC epochs"); }
    while (line.empty() || line[0]=='#');
    HarmonicData result;
    std::istringstream epochs(line);
    double epoch=0.0;
    while (epochs >> epoch) result.times.push_back(epoch);
    if (result.times.size()!=static_cast<std::size_t>(count)) throw std::runtime_error("invalid SHC epoch count");
    result.g.assign(static_cast<std::size_t>(count),Eigen::MatrixXd::Zero(nmax+1,nmax+1));
    result.h.assign(static_cast<std::size_t>(count),Eigen::MatrixXd::Zero(nmax+1,nmax+1));
    std::vector<std::vector<bool>> seen(static_cast<std::size_t>(nmax+1),std::vector<bool>(static_cast<std::size_t>(2*nmax+1),false));
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line); int n=0,m=0; row>>n>>m;
        if (!row || n<nmin || n>nmax || std::abs(m)>n) throw std::runtime_error("invalid SHC coefficient row");
        const std::size_t si=static_cast<std::size_t>(m+nmax);
        if (seen[static_cast<std::size_t>(n)][si]) throw std::runtime_error("duplicate SHC coefficient");
        seen[static_cast<std::size_t>(n)][si]=true;
        for (int t=0;t<count;++t) {
            double value=0.0; row>>value;
            if (!row) throw std::runtime_error("malformed SHC values");
            (m>=0 ? result.g[static_cast<std::size_t>(t)] : result.h[static_cast<std::size_t>(t)])(n,std::abs(m))=value;
        }
    }
    Eigen::MatrixXd scale=Eigen::MatrixXd::Zero(nmax+1,nmax+1); scale(0,0)=1.0;
    for (int n=1;n<=nmax;++n) {
        scale(n,0)=scale(n-1,0)*(2.0*n-1.0)/n;
        for(int m=1;m<=n;++m) scale(n,m)=scale(n,m-1)*std::sqrt((n-m+1.0)*(m==1?2.0:1.0)/(n+m));
    }
    for(std::size_t i=0;i<result.g.size();++i) { result.g[i].array()*=scale.array(); result.h[i].array()*=scale.array(); }
    return result;
}

Eigen::Vector3d eval_field(const Eigen::Vector3d& position, double jd,
    const Eigen::MatrixXd& g,const Eigen::MatrixXd& h,int nmax) {
    const double jdtt=jd>=2441317.5 ? omni::time::utc_jd_to_tt_jd(jd) : jd;
    const Eigen::Matrix3d rot=omni::frames::gcrf_to_ecef_matrix(jd,jdtt);
    const Eigen::Vector3d fixed=rot*position;
    const double radius=fixed.norm();
    if(radius<=0.0) throw std::invalid_argument("position must have positive radius");
    const double theta=std::acos(std::clamp(fixed.z()/radius,-1.0,1.0));
    const double phi=std::atan2(fixed.y(),fixed.x());
    const double st=std::sin(theta),ct=std::cos(theta),sp=std::sin(phi),cp=std::cos(phi);
    const double safe_theta=(theta==0.0)?1.0e-7:((theta==kPi)?kPi-1.0e-7:theta);
    const double safe_sin=std::sin(safe_theta);
    std::vector<double> p1(static_cast<std::size_t>(nmax+2),0.0),p2(p1),pc(p1),d1(p1),d2(p1),dc(p1);
    std::vector<double> cm(static_cast<std::size_t>(nmax+1),0.0),sm(cm);
    cm[0]=1.0;
    for(int m=1;m<=nmax;++m) { const double previous_cos=cm[m-1],previous_sin=sm[m-1];cm[m]=previous_cos*cp-previous_sin*sp;sm[m]=previous_sin*cp+previous_cos*sp; }
    p1[0]=1.0; const double ar=kMagneticReferenceRadiusKm/(radius/1000.0); double arn=ar*ar;
    double br=0.0,bt=0.0,bp=0.0;
    for(int n=1;n<=nmax;++n) {
        arn*=ar;
        for(int m=0;m<=n;++m) {
            if(n==m) { pc[n]=safe_sin*p1[m-1]; dc[n]=safe_sin*d1[m-1]+std::cos(safe_theta)*p1[n-1]; }
            else if(n==1) { pc[m]=std::cos(safe_theta)*p1[m]; dc[m]=std::cos(safe_theta)*d1[m]-safe_sin*p1[m]; }
            else { const double q=((n-1.0)*(n-1.0)-m*m)/((2.0*n-1.0)*(2.0*n-3.0)); pc[m]=std::cos(safe_theta)*p1[m]-q*p2[m]; dc[m]=std::cos(safe_theta)*d1[m]-safe_sin*p1[m]-q*d2[m]; }
        }
        pc[n+1]=0.0; dc[n+1]=0.0;
        for(int m=0;m<=n;++m) { const double gn=g(n,m),hn=h(n,m),cc=gn*cm[m]+hn*sm[m]; br+=arn*(n+1.0)*cc*pc[m]; bt-=arn*cc*dc[m]; bp-=arn*m*(hn*cm[m]-gn*sm[m])*pc[m]; }
        std::vector<double> old_p2=std::move(p2); p2=std::move(p1); p1=std::move(pc); pc=std::move(old_p2);
        std::vector<double> old_d2=std::move(d2); d2=std::move(d1); d1=std::move(dc); dc=std::move(old_d2);
    }
    bp/=safe_sin;
    const Eigen::Vector3d bfixed((br*st*cp+bt*ct*cp-bp*sp)*1e-9,
        (br*st*sp+bt*ct*sp+bp*cp)*1e-9,(br*ct-bt*st)*1e-9);
    return rot.transpose()*bfixed;
}
}

std::filesystem::path environment_data_path(const std::filesystem::path& relative) {
    const std::filesystem::path root=std::filesystem::path(__FILE__).parent_path();
    return root/relative;
}
Eigen::MatrixXd schmidt_table(int maximum_degree) {
    if (maximum_degree<0) throw std::invalid_argument("maximum degree must be nonnegative");
    Eigen::MatrixXd scale=Eigen::MatrixXd::Zero(maximum_degree+1,maximum_degree+1); scale(0,0)=1.0;
    for(int n=1;n<=maximum_degree;++n) { scale(n,0)=scale(n-1,0)*(2.0*n-1.0)/n; for(int m=1;m<=n;++m) scale(n,m)=scale(n,m-1)*std::sqrt((n-m+1.0)*(m==1?2.0:1.0)/(n+m)); }
    return scale;
}
Eigen::Vector3d magnetic_field_with_coefficients(const Eigen::Vector3d& position,double jd,const Eigen::MatrixXd& g,const Eigen::MatrixXd& h,int nmax) { return eval_field(position,jd,g,h,nmax); }
std::tuple<std::vector<double>,std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> load_shc(const std::filesystem::path& path,int nmin,int nmax) { auto value=read_shc(path,nmin,nmax);return {std::move(value.times),std::move(value.g),std::move(value.h)}; }

double nrlmsise_density_geodetic(int doy,double sec,double alt,double lat,double lon,double f107,double f107a,double ap,bool effective,const std::array<double,7>& ap_hist) {
    nrlmsise_input input{}; input.doy=doy; input.sec=sec; input.alt=alt; input.g_lat=lat; input.g_long=lon;
    input.lst=std::fmod(std::fmod(sec/3600.0+lon/15.0,24.0)+24.0,24.0); input.f107A=f107a; input.f107=f107; input.ap=ap;
    ap_array history{}; for(std::size_t i=0;i<7U;++i) history.a[i]=ap_hist[i]; input.ap_a=&history;
    nrlmsise_flags flags{}; flags.switches[0]=0; for(int i=1;i<24;++i) flags.switches[i]=1;
    nrlmsise_output output{}; if(effective) gtd7d(&input,&flags,&output); else gtd7(&input,&flags,&output);
    return output.d[5]*1000.0; // g/cm^3 to kg/m^3
}

double nrlmsise_density(const Eigen::Vector3d& p,double jd,double f107,double f107a,double ap,bool effective,const std::array<double,7>& ap_hist) {
    const auto [year,month,day,seconds]=omni::time::jd_to_utc_calendar_fields(jd);
    const int doy=omni::time::day_of_year(year,month,day);
    const double jdtt=jd>=2441317.5 ? omni::time::utc_jd_to_tt_jd(jd) : jd;
    const Eigen::Vector3d geodetic=omni::frames::gcrf_position_to_geodetic(p,jd,jdtt);
    return nrlmsise_density_geodetic(doy,seconds,geodetic.z()/1000.0,geodetic.x(),geodetic.y(),f107,f107a,ap,effective,ap_hist);
}

DE421::DE421(const std::filesystem::path& supplied) {
    filepath_=supplied.empty()?environment_data_path("ephemeris/de421/de421.bsp"):supplied;
    std::ifstream file(filepath_,std::ios::binary); if(!file) throw std::runtime_error("cannot open DE421 kernel: "+filepath_.string());
    std::vector<unsigned char> raw((std::istreambuf_iterator<char>(file)),{});
    if(raw.size()<1024U || std::string(raw.begin(),raw.begin()+8)!="DAF/SPK ") throw std::runtime_error("invalid DAF/SPK kernel");
    const std::string marker(raw.begin()+88,raw.begin()+96); const bool little=marker=="LTL-IEEE";
    if(!little && marker!="BIG-IEEE") throw std::runtime_error("unsupported DAF byte order");
    const int nd=static_cast<int32_t>(read_u32(raw,8,little)),ni=static_cast<int32_t>(read_u32(raw,12,little));
    if(nd!=2||ni!=6) throw std::runtime_error("unsupported DAF summary format");
    int record=static_cast<int32_t>(read_u32(raw,76,little)); const std::size_t summary_size=nd*8U+ni*4U;
    while(record!=0) {
        const std::size_t base=(static_cast<std::size_t>(record)-1U)*1024U;
        if(base+1024U>raw.size()) throw std::runtime_error("invalid DAF summary record");
        const int next=static_cast<int>(read_f64(raw,base,little)); const double countd=read_f64(raw,base+16U,little);
        if(countd<0.0 || countd>25.0) throw std::runtime_error("invalid DAF summary count");
        const int count=static_cast<int>(countd);
        for(int i=0;i<count;++i) {
            const std::size_t off=base+24U+static_cast<std::size_t>(i)*summary_size;
            const double start=read_f64(raw,off,little),end=read_f64(raw,off+8U,little);
            const std::size_t ints=off+16U;
            const int target=static_cast<int32_t>(read_u32(raw,ints,little));
            const int center=static_cast<int32_t>(read_u32(raw,ints+4U,little));
            const int type=static_cast<int32_t>(read_u32(raw,ints+12U,little));
            const int first=static_cast<int32_t>(read_u32(raw,ints+16U,little));
            const int last=static_cast<int32_t>(read_u32(raw,ints+20U,little));
            if(type!=2&&type!=3) continue;
            if(first<1||last<first||static_cast<std::size_t>(last)*8U>raw.size()) throw std::runtime_error("invalid SPK data addresses");
            const std::size_t offset=data_.size(), length=static_cast<std::size_t>(last-first+1);
            for(std::size_t j=0;j<length;++j) data_.push_back(read_f64(raw,(static_cast<std::size_t>(first)-1U+j)*8U,little));
            segments_.push_back(Segment{target,center,type,offset,length,start,end});
        }
        record=next;
    }
    if(segments_.empty()) throw std::runtime_error("no supported SPK segments");
}

DE421::State DE421::segment_state(int target,double et) const {
    const Segment* selected=nullptr;
    for(const auto& s:segments_) if(s.target==target&&s.start<=et&&et<=s.end) selected=&s;
    if(selected==nullptr) throw std::out_of_range("no DE421 segment covers target and epoch");
    const std::size_t trailer=selected->offset+selected->size-4U; const double initial=data_[trailer], length=data_[trailer+1U];
    const int record_size=static_cast<int>(data_[trailer+2U]), count=static_cast<int>(data_[trailer+3U]);
    int rec=static_cast<int>((et-initial)/length); rec=std::clamp(rec,0,count-1);
    const std::size_t base=selected->offset+static_cast<std::size_t>(rec*record_size); const double mid=data_[base], radius=data_[base+1U], x=(et-mid)/radius;
    const int ncf=(record_size-2)/(selected->type==2?3:6); State result{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()};
    for(int axis=0;axis<3;++axis) {
        const std::size_t po=base+2U+static_cast<std::size_t>(axis*ncf); double t0=1.0,t1=x,d0=0.0,d1=1.0;
        result.position[axis]=data_[po]; if(ncf>1) result.position[axis]+=data_[po+1U]*x;
        if(selected->type==3) { const std::size_t vo=base+2U+static_cast<std::size_t>((axis+3)*ncf); result.velocity[axis]=data_[vo]+data_[vo+1U]*x; }
        for(int k=2;k<ncf;++k) { const double t2=2.0*x*t1-t0, d2=2.0*t1+2.0*x*d1-d0; result.position[axis]+=data_[po+static_cast<std::size_t>(k)]*t2; if(selected->type==3) result.velocity[axis]+=data_[base+2U+static_cast<std::size_t>((axis+3)*ncf)+static_cast<std::size_t>(k)]*t2; t0=t1;t1=t2;d0=d1;d1=d2; }
        if(selected->type==2) { double derivative=ncf>1?data_[po+1U]:0.0; t0=1.0;t1=x;d0=0.0;d1=1.0; for(int k=2;k<ncf;++k) { const double t2=2.0*x*t1-t0,d2=2.0*t1+2.0*x*d1-d0; derivative+=data_[po+static_cast<std::size_t>(k)]*d2;t0=t1;t1=t2;d0=d1;d1=d2; } result.velocity[axis]=derivative/radius; }
    }
    return result;
}
DE421::State DE421::barycentric_state(int target,double et) const { State sum{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()}; int node=target; for(int depth=0;depth<8&&node!=0;++depth) { const State s=segment_state(node,et); sum.position+=s.position;sum.velocity+=s.velocity; const Segment* found=nullptr;for(const auto& seg:segments_)if(seg.target==node&&seg.start<=et&&et<=seg.end)found=&seg;if(found==nullptr)throw std::out_of_range("missing DE421 center");node=found->center; } if(node!=0)throw std::runtime_error("DE421 center chain too deep");return sum; }
int DE421::body_target(const std::string& name) const { static const std::map<std::string,int> names{{"mercury",1},{"venus",2},{"earth-moon barycenter",3},{"emb",3},{"mars",4},{"jupiter",5},{"saturn",6},{"uranus",7},{"neptune",8},{"pluto",9},{"moon",301},{"sun",10},{"earth",399}};auto it=names.find(name);if(it==names.end())throw std::invalid_argument("unsupported DE421 body: "+name);return it->second; }
std::pair<Eigen::Vector3d,Eigen::Vector3d> DE421::state_eci(const std::string& target,const double jd,const std::string& observer) const { const int t=body_target(target),o=body_target(observer);const double et=(jd-kJ2000)*kSecondsPerDay;State ts=t==399?State{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()}:barycentric_state(t,et);State os=o==399?State{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()}:barycentric_state(o,et); if(t==399){ts=barycentric_state(399,et); } if(o==399){os=barycentric_state(399,et);} return {ts.position-os.position,ts.velocity-os.velocity}; }
Eigen::Vector3d DE421::position_eci(const std::string& t,double jd,const std::string& o) const { return state_eci(t,jd,o).first; }
Eigen::Vector3d DE421::velocity_eci(const std::string& t,double jd,const std::string& o) const { return state_eci(t,jd,o).second; }
Eigen::Vector3d DE421::position(int id,double jd) const { if(id==1)return Eigen::Vector3d::Zero();if(id==0)return position_eci("sun",jd);if(id==2)return position_eci("moon",jd);throw std::invalid_argument("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON"); }
Eigen::Vector3d DE421::velocity(int id,double jd) const { if(id==1)return Eigen::Vector3d::Zero();if(id==0)return velocity_eci("sun",jd);if(id==2)return velocity_eci("moon",jd);throw std::invalid_argument("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON"); }
std::vector<std::string> DE421::available_bodies() const { return {"earth-moon barycenter","emb","jupiter","mars","mercury","moon","neptune","pluto","saturn","sun","uranus","venus"}; }

Eigen::Vector3d MagneticFieldModel::field(const Eigen::Vector3d& p,double jd) const { return field_inertial_tesla(p,jd); }
IGRF14::IGRF14() { auto d=read_shc(environment_data_path("magnetic_field_model/igrf14/kernels/IGRF14.shc"),1,13);for(double year:d.times)times_.push_back(omni::time::calendar_to_jd(static_cast<int>(year),1,1));g_=std::move(d.g);h_=std::move(d.h); }
Eigen::Vector3d IGRF14::field_inertial_tesla(const Eigen::Vector3d& p,double jd) const { if(jd<times_.front()||jd>times_.back())throw std::out_of_range("IGRF-14 coefficients cover 1900-01-01 to 2030-01-01");auto upper=std::upper_bound(times_.begin(),times_.end(),jd);std::size_t i=upper==times_.end()?times_.size()-2U:static_cast<std::size_t>(upper-times_.begin()-1);const double w=(jd-times_[i])/(times_[i+1U]-times_[i]);return eval_field(p,jd,g_[i]+w*(g_[i+1U]-g_[i]),h_[i]+w*(h_[i+1U]-h_[i]),13); }
CHAOS8::CHAOS8(bool include_static):include_static_(include_static) { auto core=read_shc(environment_data_path("magnetic_field_model/chaos8/kernels/CHAOS-8.7_core.shc"),1,20);times_=std::move(core.times);g_=std::move(core.g);h_=std::move(core.h);static_g_=Eigen::MatrixXd::Zero(186,186);static_h_=static_g_;if(include_static_){auto stat=read_shc(environment_data_path("magnetic_field_model/chaos8/kernels/CHAOS-8.7_static.shc"),21,185);static_g_=stat.g[0];static_h_=stat.h[0];} }
Eigen::Vector3d CHAOS8::field_inertial_tesla(const Eigen::Vector3d& p,double jd) const { const double year=2000.0+(jd-2451544.5)/365.25;if(year<times_.front()||year>times_.back())throw std::out_of_range("Epoch outside CHAOS-8.7 core coefficient range");const auto up=std::upper_bound(times_.begin(),times_.end(),year);std::size_t start=(up==times_.begin()?0U:static_cast<std::size_t>(up-times_.begin()-1));start=(start/5U)*5U;start=std::min(start,times_.size()-6U);Eigen::MatrixXd g=Eigen::MatrixXd::Zero(186,186),h=Eigen::MatrixXd::Zero(186,186);if(include_static_){g=static_g_;h=static_h_;}for(std::size_t j=0;j<6U;++j){double w=1.0;for(std::size_t k=0;k<6U;++k)if(j!=k)w*= (year-times_[start+k])/(times_[start+j]-times_[start+k]);g.block(1,0,20,21)+=w*g_[start+j].block(1,0,20,21);h.block(1,0,20,21)+=w*h_[start+j].block(1,0,20,21);}return eval_field(p,jd,g,h,include_static_?185:20); }
Eigen::Vector3d WMM::field_inertial_tesla(const Eigen::Vector3d&,double) const { throw std::runtime_error("WMM coefficients are not implemented"); }

std::tuple<double,bool,bool> eclipse_state(const Eigen::Vector3d& r,const Eigen::Vector3d& sun,double body_radius,double sun_radius) { const Eigen::Vector3d s=sun-r,b=-r;const double ds=s.norm(),db=b.norm();if(ds<=sun_radius||db<=body_radius)return {0.0,true,false};const double as=std::asin(std::min(1.0,sun_radius/ds)),ab=std::asin(std::min(1.0,body_radius/db));const double sep=std::acos(std::clamp(s.dot(b)/(ds*db),-1.0,1.0));if(sep>=as+ab)return {1.0,false,false};if(ab>=as&&sep<=ab-as)return {0.0,true,false};double area=0.0;if(sep<=std::abs(ab-as))area=kPi*std::min(as,ab)*std::min(as,ab);else {const double x=(sep*sep+as*as-ab*ab)/(2.0*sep*as),y=(sep*sep+ab*ab-as*as)/(2.0*sep*ab);const double rad=std::max(0.0,(-sep+as+ab)*(sep+as-ab)*(sep-as+ab)*(sep+as+ab));area=as*as*std::acos(std::clamp(x,-1.0,1.0))+ab*ab*std::acos(std::clamp(y,-1.0,1.0))-0.5*std::sqrt(rad);}return {std::clamp(1.0-area/(kPi*as*as),0.0,1.0),false,true}; }
} // namespace omni::environment
