// An executable for testing all the bits and pieces of trajectory planning
// #include <catch2/extras/catch_amalgamated.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <iostream>
#include <Armadillo>
#include <boost/math/tools/numerical_differentiation.hpp>
/*#include <armadillo>
#include <rapidcsv.h>
#include <picojson.h>
#include <ArmaCSV.hpp>
#include <ArmaJSON.hpp>
#include <ArmaNumpy.hpp>
#include <vector>
#include <string>
#include <sstream>

#include <fstream>
#include <planner/OldPlanner.hpp>
#include <planner/PlannerUtil.hpp>
#include <string>
#include <tuple>*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Satellite.hpp"
#include "PlannerUtil.hpp"
// #include "OldPlanner.hpp"
// #include "PlannerUtil.hpp"
// #include "../ArmaNumpy.hpp"

// namespace py = pybind11;

TEST_CASE("print hello world") {
	std::cout<<"hello world";
}

TEST_CASE("big matrix mult") {
	arma::mat fake_clearvel = arma::mat(7, 7).eye();
  arma::mat fake_Xset = arma::mat(7, 10000).zeros();
  arma::mat test = fake_clearvel*fake_Xset;
	std::cout<<"test ncols"<<test.n_cols<<"\n";

}

TEST_CASE("Test rotMat", "[armadillo]") {
	//Set input
	arma::vec4 q;
  q(0, 0) = -0.1104;
  q(1, 0) = 0.4417;
  q(2, 0) = 0.7730;
  q(3, 0) = -0.4417;
	arma::mat matrix_out = rotMat(q);
	//Set expected output
	arma::mat33 matrix_expected;
  matrix_expected(0, 0) = -0.5853;
  matrix_expected(0, 1) = 0.5853;
  matrix_expected(0, 2) = -0.5609;
  matrix_expected(1, 0) = 0.7804;
  matrix_expected(1, 1) = 0.2195;
  matrix_expected(1, 2) = -0.5853;
  matrix_expected(2, 0) = -0.2195;
  matrix_expected(2, 1) = -0.7804;
  matrix_expected(2, 2) = -0.5853;
	//Assert output == expected output within 1e-2
	REQUIRE(arma::approx_equal(matrix_out.row(0), matrix_expected.row(0), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(1), matrix_expected.row(1), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(2), matrix_expected.row(2), "absdiff", 1e-02));
}

TEST_CASE("Test skewSymmetric", "[armadillo]") {
	//Set input
	arma::vec vk = arma::vec(3);
  vk(0) = 1;
  vk(1) = 2;
  vk(2) = 3;
	arma::mat matrix_out = skewSymmetric(vk);
	//Set expected output
  const std::initializer_list<std::initializer_list<double>> skewSymexpected_contents = {{0, -3, 2}, {3, 0, -1}, {-2, 1, 0}};
  arma::mat matrix_expected = arma::mat(skewSymexpected_contents);
	//Assert output == expected output within 1e-2
	REQUIRE(arma::approx_equal(matrix_out.row(0), matrix_expected.row(0), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(1), matrix_expected.row(1), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(2), matrix_expected.row(2), "absdiff", 1e-02));
}
//
TEST_CASE("Test findWMat", "[armadillo]") {
	//Set input
	arma::vec4 qk = arma::vec({4, 1, 2, 3});
	arma::mat::fixed<4,3> matrix_out = findWMat(qk);
	//Set expected output
	const std::initializer_list<std::initializer_list<double>> wMatexpected_contents = {{-1, -2, -3}, {4, -3, 2}, {3, 4, -1}, {-2, 1, 4}};
	arma::mat matrix_expected = arma::mat(wMatexpected_contents);
	//Assert output == expected output within 1e-2
	REQUIRE(arma::approx_equal(matrix_out.row(0), matrix_expected.row(0), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(1), matrix_expected.row(1), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(2), matrix_expected.row(2), "absdiff", 1e-02));
	REQUIRE(arma::approx_equal(matrix_out.row(3), matrix_expected.row(3), "absdiff", 1e-02));
}

//
TEST_CASE("Test quatcostJac", "[armadillo]") {
		//Set input
			//TODO tests of final step, RW, magic
	for(int mode = 0; mode<6; mode++){
		cout<<"mode "<<mode<<"\n";
		cout<<"quatcost\n";

		arma::arma_rng::set_seed_random();
		Satellite sat = Satellite();
		arma::mat33 vecmat = arma::mat33().eye();
		sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
		sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
		sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
		sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);

		arma::vec3 z3 = arma::vec3().zeros();
		arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
		arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
		arma::vec xk = join_cols(wk,qk);
		arma::vec3 uk = 0.5*arma::normalise(arma::vec(3,fill::randn));
		arma::vec3 satvec_k = arma::vec({1,0,0});
		arma::vec4 ECIvec_k = arma::normalise(arma::vec(4,fill::randn));
		arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));

		for(int k = 8;k<10;k++){
			int N = 10;
			COST_SETTINGS_FORM costset_tmp = std::make_tuple(1.0e3,1.0e0,1.0e0,0.1,0.33,3.0,1.0e6,1.0e3,1.0e-2,1.43,mode,1);
			// double w_ang = get<0>(costSettings_tmp);
			// double w_av = get<1>(costSettings_tmp);
			// double w_u_mult = get<2>(costSettings_tmp);
			// double w_umag = get<3>(costSettings_tmp);
			// double w_avmag = get<4>(costSettings_tmp);
			// double w_avang = get<5>(costSettings_tmp);

			double cost = sat.stepcost_quat(k, N, xk, uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);
			cost_jacs costJac = sat.quatcostJacobians(k, N, xk, uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
			//Set expected output
			arma::vec lkx = costJac.lx;
			arma::mat lkxx = costJac.lxx;
			arma::mat lkux = costJac.lux;
			arma::vec lku = costJac.lu;
			arma::mat lkuu = costJac.luu;
			arma::vec ee = xk*0;
		  arma::vec df__dx = arma::vec(xk.n_elem).zeros();
			arma::vec df__dx_errest = arma::vec(xk.n_elem).zeros();

			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				double errest = 0;
				auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_quat(k,N,xk + ee*(xi-x0i), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
				df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i,&errest);
				df__dx_errest(i) = errest;
			}
			cout<<"quatcost lkx\n";
			arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
			cout<<df__dxQ.t()<<"\n";
			cout<<lkx.t()<<"\n";
			cout<<(df__dxQ-lkx).t()<<"\n";
			cout<<df__dx_errest.t()<<"\n";
			CHECK(arma::approx_equal(df__dxQ,lkx , "absdiff", 1e-04));


			ee = uk*0;
		  arma::vec df__du = arma::vec(uk.n_elem).zeros();
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_quat(k,N,xk, uk + ee*(ui-u0i),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
				df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
			}
			cout<<"quatcost lku\n";
			cout<<df__du.t()<<"\n";
			cout<<lku.t()<<"\n";
			cout<<(df__du-lku).t()<<"\n";
			CHECK(arma::approx_equal(df__du,lku , "absdiff", 1e-04));


			arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
			arma::vec er = arma::vec(xk.n_elem).zeros();
			ee = xk*0;
			for(int j = 0; j<xk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double x0j = xk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);
					if(i==j)
					{
						auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_quat(k,N,xk + ee*(xi-x0i), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
						auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
						ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

					}
					else
					{
						auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_quat(k,N,xk + ee*(xi-x0i) + er*(xj-x0j), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																													return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
						ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<"quatcost lkxx\n";
			cout<<ddf__dxdx<<"\n";
			arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
			ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
			cout<<ddf__dxdxQ<<"\n";
			cout<<lkxx<<"\n";
			cout<<(ddf__dxdxQ-lkxx)<<"\n";
			CHECK(arma::approx_equal(ddf__dxdxQ,lkxx , "absdiff", 1e-04));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);
					if(i==j)
					{
						auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_quat(k,N,xk, uk+ ee*(ui-u0i),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
						auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}
					else
					{
						auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_quat(k,N,xk, uk + ee*(ui-u0i) + er*(uj-u0j),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																													return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<"quatcost lkuu\n";
			cout<<ddf__dudu<<"\n";
			cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			CHECK(arma::approx_equal(ddf__dudu,lkuu , "absdiff", 1e-04));



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				cout<<"j "<<j<<"\n";
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);
					auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return sat.stepcost_quat(k,N,xk + ee*(xi-x0i) , uk+ er*(uj-u0j),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																												return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<"quatcost lkux\n";
			cout<<ddf__dudx<<"\n";
			arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			cout<<lkux<<"\n";
			cout<<(ddf__dudxQ-lkux)<<"\n";
			CHECK(arma::approx_equal(ddf__dudxQ,lkux , "absdiff", 1e-04));
		}
	}
}

//
TEST_CASE("Test veccostJac", "[armadillo]") {
	//Set input
		//TODO tests of final step, RW, magic
for(int mode = 0; mode<4; mode++){
	cout<<"mode "<<mode<<"\n";
	cout<<"quatcost\n";
		cout<<"veccost\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	// qk = arma::vec({1,0,0,0});
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk);
	arma::vec3 uk = 0.5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));

	for(int k = 8;k<10;k++){
		int N = 10;

		COST_SETTINGS_FORM costset_tmp = std::make_tuple(1.0e3,1.0e0,1.0e0,1.2,0.33,3.0,1.0e6,1.0e3,1.0e-1,3.0,mode,1);
		// costset_tmp = std::make_tuple(0.0e3,0.0e0,0.0e0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,0,1);
		double cost = sat.stepcost_vec(k, N, xk, uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);
		cost_jacs costJac = sat.veccostJacobians(k, N, xk, uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
		//Set expected output
		arma::vec lkx = costJac.lx;
		arma::mat lkxx = costJac.lxx;
		arma::mat lkux = costJac.lux;
		arma::vec lku = costJac.lu;
		arma::mat lkuu = costJac.luu;
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();
		arma::vec df__dx_errest = arma::vec(xk.n_elem).zeros();

		double x0i;
		double errest;

		for(int iii = 0; iii<xk.n_elem;iii++){
			ee.zeros();
			ee(iii) = 1;
			errest = 0;
			x0i = xk(iii);
			auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_vec(k,N,xk + ee*(xi-x0i), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i,&errest);
			df__dx_errest(iii) = errest;
		}
		cout<<"veccost lkx\n";
		cout<<df__dx.t()<<"\n";
		arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
		// cout<<normalise(BECI_k).t()<<"\n";
		// cout<<join_cols(rotMat(qk).t()*normalise(BECI_k),(wk.t()*dRTBdqQ(qk,normalise(BECI_k))).t()).t()<<"\n";
		cout<<df__dxQ.t()<<"\n";
		cout<<lkx.t()<<"\n";
		cout<<(df__dxQ-lkx).t()<<"\n";
		cout<<df__dx_errest.t()<<"\n";

		CHECK(arma::approx_equal(df__dxQ,lkx , "absdiff", 1e-04));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_vec(k,N,xk, uk + ee*(ui-u0i),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<"veccost lku\n";
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		CHECK(arma::approx_equal(df__du,lku , "absdiff", 1e-04));


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;

		// cout<<"veccost lkxx steps\n";
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_vec(k,N,xk + ee*(xi-x0i), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return sat.stepcost_vec(k,N,xk + ee*(xi-x0i) + er*(xj-x0j), uk,z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}


				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		cout<<"veccost lkxx\n";
		cout<<ddf__dxdx<<"\n";
		arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		cout<<ddf__dxdxQ<<"\n";
		cout<<lkxx<<"\n";
		cout<<(ddf__dxdxQ-lkxx)<<"\n";
		CHECK(arma::approx_equal(ddf__dxdxQ,lkxx , "absdiff", 1e-04));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);
					if(i==j)
					{
						auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_vec(k,N,xk, uk+ ee*(ui-u0i),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
						auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}
					else
					{
						auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return sat.stepcost_vec(k,N,xk, uk + ee*(ui-u0i) + er*(uj-u0j),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																													return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<"veccost lkuu\n";
			cout<<ddf__dudu<<"\n";
			cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			CHECK(arma::approx_equal(ddf__dudu,lkuu , "absdiff", 1e-04));



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				cout<<"j "<<j<<"\n";
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);
					auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return sat.stepcost_vec(k,N,xk + ee*(xi-x0i) , uk+ er*(uj-u0j),z3, satvec_k,  ECIvec_k,BECI_k,  &costset_tmp);};
																												return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<"veccost lkux\n";
			cout<<ddf__dudx<<"\n";
			arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			cout<<lkux<<"\n";
			cout<<(ddf__dudxQ-lkux)<<"\n";
			CHECK(arma::approx_equal(ddf__dudxQ,lkux , "absdiff", 1e-04));
		}
	}
}


//
TEST_CASE("Test constraint jacobians & Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"CONSTRAINTS\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec cnstr = sat.getConstraints(k,N,uk,xk,sunk);
	std::tuple<arma::mat,arma::mat> cjs = sat.constraintJacobians(k,N,uk,xk,sunk);
	std::tuple<arma::cube,arma::cube,arma::cube> chs = sat.constraintHessians(k,N,uk,xk,sunk);
	arma::mat cku = std::get<0>(cjs);
	arma::mat ckx = std::get<1>(cjs);
	arma::cube ckuu = std::get<0>(chs);
	arma::cube ckux = std::get<1>(chs);
	arma::cube ckxx = std::get<2>(chs);

	for(int ind = 0; ind<sat.constraint_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.constraint_N()).zeros();
		eind(ind) = 1.0;
		arma::vec lku = cku.row(ind).t();
		arma::vec lkx = ckx.row(ind).t();
		arma::mat lkuu = ckuu.slice(ind);
		arma::mat lkux = ckux.slice(ind);
		arma::mat lkxx = ckxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.getConstraints(k,N,uk,xk + ee*(xi-x0i),sunk));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<df__dx.t()<<"\n";
		arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
		cout<<df__dxQ.t()<<"\n";
		cout<<lkx.t()<<"\n";
		cout<<(df__dxQ-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dxQ,lkx , "absdiff", 1e-04));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,sat.getConstraints(k,N,uk + ee*(ui-u0i),xk,sunk));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "absdiff", 1e-04));


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.getConstraints(k,N,uk ,xk+ ee*(xi-x0i),sunk));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.getConstraints(k,N,uk ,xk+ ee*(xi-x0i)+ er*(xj-x0j),sunk));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		cout<<ddf__dxdx<<"\n";
		arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		cout<<ddf__dxdxQ<<"\n";
		cout<<lkxx<<"\n";
		cout<<(ddf__dxdxQ-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdxQ,lkxx , "absdiff", 1e-04));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);
					if(i==j)
					{
						auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,sat.getConstraints(k,N,uk+ ee*(ui-u0i),xk,sunk));};
						auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}
					else
					{
						auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,sat.getConstraints(k,N,uk+ ee*(ui-u0i)+ er*(uj-u0j) ,xk,sunk));};
																													return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<ddf__dudu<<"\n";
			cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			cout<<xk.t()<<"\n";
			cout<<uk.t()<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudu,lkuu , "absdiff", 1e-04));



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);
					auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.getConstraints(k,N,uk+ er*(uj-u0j) ,xk+ ee*(xi-x0i),sunk));};
																												return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<ddf__dudx<<"\n";
			arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			cout<<lkux<<"\n";
			cout<<(ddf__dudxQ-lkux)<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudxQ,lkux , "absdiff", 1e-04));
		}
}


TEST_CASE("Test norm, jacobians, & Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"norm\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk0 = 1.1*normalise(arma::vec(4,fill::randn));
	cout<<qk0.t()<<"\n";
	arma::vec4 qk = normalise(qk0);
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec xk0 = join_cols(wk,qk0,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xn = sat.state_norm(xk0);
	REQUIRE(arma::approx_equal(xn,xk , "absdiff", 1e-08));


	arma::mat jac = sat.state_norm_jacobian(xk0);
	arma::cube hess = sat.state_norm_hessian(xk0);


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;

		arma::vec ee = xk0*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk0(i);
			auto fxi = [=,&costset_tmp] (double xi) {return dot(eind,sat.state_norm(xk0 + ee*(xi-x0i)));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<df__dx.t()<<"\n";
		cout<<jac.row(ind)<<"\n";
		REQUIRE(arma::approx_equal(df__dx,jac.row(ind).t() , "absdiff", 1e-010));



		arma::mat ddf__dxdx = arma::mat(xk0.n_elem,xk0.n_elem).zeros();
		arma::vec er = arma::vec(xk0.n_elem).zeros();
		ee = xk0*0;
		for(int j = 0; j<xk0.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk0(j);
			for(int i = 0; i<xk0.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk0(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.state_norm(xk0 + ee*(xi-x0i)));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.state_norm(xk0+ee*(xi-x0i)+ er*(xj-x0j)));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		cout<<ddf__dxdx<<"\n";
		cout<<hess.slice(ind)<<"\n";
		cout<<ddf__dxdx-hess.slice(ind)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,hess.slice(ind) , "absdiff", 1e-010));
	}

	//TODO tests of final step, magic
	cout<<"norm 2\n";
	arma::arma_rng::set_seed_random();
	sat = Satellite();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	torqs = arma::vec({1e-4,2e-4,5e-5});
	ams = 3e-3*arma::vec3().ones();
	js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	qk0 = 0.9*normalise(arma::vec(4,fill::randn));
	cout<<qk0.t()<<"\n";
	qk = normalise(qk0);
	wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	xk = join_cols(wk,qk,hk);
	xk0 = join_cols(wk,qk0,hk);

	xn = sat.state_norm(xk0);
	REQUIRE(arma::approx_equal(xn,xk , "absdiff", 1e-08));


	jac = sat.state_norm_jacobian(xk0);
	hess = sat.state_norm_hessian(xk0);


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;

		arma::vec ee = xk0*0;
		arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk0(i);
			auto fxi = [=,&costset_tmp] (double xi) {return dot(eind,sat.state_norm(xk0 + ee*(xi-x0i)));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<df__dx.t()<<"\n";
		cout<<jac.row(ind)<<"\n";
		REQUIRE(arma::approx_equal(df__dx,jac.row(ind).t() , "absdiff", 1e-010));



		arma::mat ddf__dxdx = arma::mat(xk0.n_elem,xk0.n_elem).zeros();
		arma::vec er = arma::vec(xk0.n_elem).zeros();
		ee = xk0*0;
		for(int j = 0; j<xk0.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk0(j);
			for(int i = 0; i<xk0.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk0(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.state_norm(xk0 + ee*(xi-x0i)));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.state_norm(xk0+ee*(xi-x0i)+ er*(xj-x0j)));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		cout<<ddf__dxdx<<"\n";
		cout<<hess.slice(ind)<<"\n";
		cout<<ddf__dxdx-hess.slice(ind)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,hess.slice(ind) , "absdiff", 1e-010));
	}

}


//
TEST_CASE("Test dynamics Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"Dynamics Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.08})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-2*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-2*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;


	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);


	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	std::tuple<arma::vec,arma::vec> out = sat.dynamics(xk,uk,dynamics_info_k);
	arma::vec xd =std::get<0>(out);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = sat.dynamicsJacobians(xk,uk,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = sat.dynamicsHessians(xk,uk,dynamics_info_k);
	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);



	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.dynamics_pure(xk+ ee*(xi-x0i),uk,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.dynamics_pure(xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-09));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);
					if(i==j)
					{
						auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,sat.dynamics_pure(xk,uk+ ee*(ui-u0i),dynamics_info_k));};
						auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}
					else
					{
						auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,sat.dynamics_pure(xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),dynamics_info_k));};
																													return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<ddf__dudu<<"\n";
			cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			// cout<<xk.t()<<"\n";
			// cout<<uk.t()<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudu,lkuu , "absdiff", 1e-09));



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);

					auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.dynamics_pure(xk+ ee*(xi-x0i),uk+ er*(uj-u0j),dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			cout<<ddf__dudx<<"\n";
			arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			cout<<lkux<<"\n";
			cout<<(ddf__dudx-lkux)<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-09));
		}
}

//
TEST_CASE("Test dynamics jacobians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic,torque
	cout<<"DYNAMICS\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-2*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-3*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);


	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	// arma::vec xd = sat.dynamics(xk,uk,dynamics_info_k);

	std::tuple<arma::vec,arma::vec> out = sat.dynamics(xk,uk,dynamics_info_k);
	arma::vec xd =std::get<0>(out);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = sat.dynamicsJacobians(xk,uk,dynamics_info_k);
	arma::mat jx = std::get<0>(jacs);
	arma::mat ju = std::get<1>(jacs);
	arma::mat jt = std::get<2>(jacs);

	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<"ind "<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		arma::vec lkt = jt.row(ind).t();
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);

			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,sat.dynamics_pure(xk + ee*(xi-x0i),uk,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<"DYNAMICS lx,ind: "<<ind<<"\n";
		cout<<df__dx.t()<<"\n";
		arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
		// cout<<df__dxQ.t()<<"\n";
		cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx ,"both", 1e-08,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,sat.dynamics_pure(xk,uk + ee*(ui-u0i),dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<"DYNAMICS lu,ind: "<<ind<<"\n";
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-08,1e-10));



		}
}


//
TEST_CASE("Test rk4 xd0 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 xd0 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zxd0(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zxd0Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;



	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd0(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd0(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zxd0(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zxd0(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd0(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
	}
}


//
TEST_CASE("Test rk4 xd1 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 xd1 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zxd1(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zxd1Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd1(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd1(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zxd1(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zxd1(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;


		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd1(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
	}

		arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
		arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
			cout<<"xd1 reduced cube check\n";
		for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
		{
			cout<<ind2<<"\n";
			// cout<<ddfQ__dudu.slice(ind2)<<"\n";
			// cout<<lkuuRedCube.slice(ind2)<<"\n";
			cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
			CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
		}

}

//
TEST_CASE("Test rk4 x1 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x1 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx1(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx1Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;



	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx1(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx1(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
	}


	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x1 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
	}
}

//
//
TEST_CASE("Test rk4 x2 Jacobians&Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x2 J&Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk =  arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx2(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx2Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::mat jx = std::get<3>(hess);
	arma::mat ju = std::get<4>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  // arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-06));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx2(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx2(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-06));

		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		//Set expected output
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		// cout<<"RK4x2 lx,ind: "<<ind<<"\n";
		// cout<<df__dx.t()<<"\n";
		// // cout<<df__dxQ.t()<<"\n";
		// cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx2(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		// cout<<"RK4x2 lu,ind: "<<ind<<"\n";
		// cout<<df__du.t()<<"\n";
		// cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));

	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x2 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
	}


}

//
//
TEST_CASE("Test rk4 x2r Jacobians&Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x2r J&Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk =  arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx2r(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx2rHessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::mat jx = std::get<3>(hess);
	arma::mat ju = std::get<4>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  // arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2r(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2r(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-06));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx2r(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx2r(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2r(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-06));

		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		//Set expected output
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx2r(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		// cout<<"RK4x2r lx,ind: "<<ind<<"\n";
		// cout<<df__dx.t()<<"\n";
		// // cout<<df__dxQ.t()<<"\n";
		// cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx2r(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		// cout<<"RK4x2r lu,ind: "<<ind<<"\n";
		// cout<<df__du.t()<<"\n";
		// cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));

	}


		arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
		arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
			cout<<"x2r reduced cube check\n";
		for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
		{
			cout<<ind2<<"\n";
			// cout<<ddfQ__dudu.slice(ind2)<<"\n";
			// cout<<lkuuRedCube.slice(ind2)<<"\n";
			cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
			CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
		}

}

//
//
TEST_CASE("Test rk4 x1 Jacobians&Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x1 J&Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk =  arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx1(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx1Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::mat jx = std::get<3>(hess);
	arma::mat ju = std::get<4>(hess);

	arma::cube ddf__duduCube = 0*huu;

	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  // arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-09));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx1(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx1(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-09));

		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		//Set expected output
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx1(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		// cout<<"RK4x2 lx,ind: "<<ind<<"\n";
		// cout<<df__dx.t()<<"\n";
		// // cout<<df__dxQ.t()<<"\n";
		// cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-09,1e-12));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx1(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		// cout<<"RK4x1 lu,ind: "<<ind<<"\n";
		// cout<<df__du.t()<<"\n";
		// cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-09,1e-12));

	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x1 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
	}

}

TEST_CASE("Test rk4 xkp1r Jacobians&Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 xkp1r Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.05,0.05,0.01})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({5e-5,2e-4,1e-4});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.5,0.02,0.01});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zxkp1r(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat,arma::cube,arma::cube,arma::cube,arma::cube,arma::cube> hess = rk4zxkp1rHessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::mat jx = std::get<3>(hess);
	arma::mat ju = std::get<4>(hess);
	arma::cube ddf__duduCube = 0*huu;


	arma::cube x0ddx0r = std::get<5>(hess);
	arma::cube xd0ddx0r = std::get<6>(hess);
	arma::cube xd1ddx0r = std::get<7>(hess);
	arma::cube xd2ddx0r = std::get<8>(hess);
	arma::cube xd3ddx0r = std::get<9>(hess);


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  // arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		//
		// ddf__dxdx.zeros();
		// er.zeros();
		// ee = xk*0;
		// for(int j = 0; j<xk.n_elem;j++){
		// 	er.zeros();
		// 	er(j) = 1;
		// 	double x0j = xk(j);
		// 	for(int i = 0; i<xk.n_elem;i++){
		// 		ee.zeros();
		// 		ee(i) = 1;
		// 		double x0i = xk(i);
		// 		if(i==j)
		// 		{
		// 			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd0(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 			auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		// 		else
		// 		{
		// 			auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd0(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 																										return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		//
		// 		// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
		// 		// 																						arma::vec lx = cj.lx;
		// 		// 																						return lx(j);
		// 		// 																					};
		// 	}
		// }
		// // cout<<"rk4zxd0\n";
		// // cout<<ddf__dxdx<<"\n";
		// // // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // // cout<<ddf__dxdxQ<<"\n";
		// // cout<<xd0ddx0r.slice(ind)<<"\n";
		// cout<<(ddf__dxdx-xd0ddx0r.slice(ind))<<"\n";
		// REQUIRE(arma::approx_equal(ddf__dxdx,xd0ddx0r.slice(ind) , "absdiff", 1e-04));
		//
		// ddf__dxdx.zeros();
		// er.zeros();
		// ee = xk*0;
		// for(int j = 0; j<xk.n_elem;j++){
		// 	er.zeros();
		// 	er(j) = 1;
		// 	double x0j = xk(j);
		// 	for(int i = 0; i<xk.n_elem;i++){
		// 		ee.zeros();
		// 		ee(i) = 1;
		// 		double x0i = xk(i);
		// 		if(i==j)
		// 		{
		// 			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd1(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 			auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		// 		else
		// 		{
		// 			auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd1(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 																										return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		//
		// 		// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
		// 		// 																						arma::vec lx = cj.lx;
		// 		// 																						return lx(j);
		// 		// 																					};
		// 	}
		// }
		// // cout<<"rk4zxd1\n";
		// // cout<<ddf__dxdx<<"\n";
		// // // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // // cout<<ddf__dxdxQ<<"\n";
		// // cout<<xd1ddx0r.slice(ind)<<"\n";
		// cout<<(ddf__dxdx-xd1ddx0r.slice(ind))<<"\n";
		// REQUIRE(arma::approx_equal(ddf__dxdx,xd1ddx0r.slice(ind) , "absdiff", 1e-04));
		//
		//
		//
		//
		// ddf__dxdx.zeros();
		// er.zeros();
		// ee = xk*0;
		// for(int j = 0; j<xk.n_elem;j++){
		// 	er.zeros();
		// 	er(j) = 1;
		// 	double x0j = xk(j);
		// 	for(int i = 0; i<xk.n_elem;i++){
		// 		ee.zeros();
		// 		ee(i) = 1;
		// 		double x0i = xk(i);
		// 		if(i==j)
		// 		{
		// 			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd2(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 			auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		// 		else
		// 		{
		// 			auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd2(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 																										return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		//
		// 		// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
		// 		// 																						arma::vec lx = cj.lx;
		// 		// 																						return lx(j);
		// 		// 																					};
		// 	}
		// }
		// // cout<<"rk4zxd2\n";
		// // cout<<ddf__dxdx<<"\n";
		// // // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // // cout<<ddf__dxdxQ<<"\n";
		// // cout<<xd2ddx0r.slice(ind)<<"\n";
		// cout<<(ddf__dxdx-xd2ddx0r.slice(ind))<<"\n";
		// REQUIRE(arma::approx_equal(ddf__dxdx,xd2ddx0r.slice(ind) , "absdiff", 1e-04));
		//

		//
		// ddf__dxdx.zeros();
		// er.zeros();
		// ee = xk*0;
		// for(int j = 0; j<xk.n_elem;j++){
		// 	er.zeros();
		// 	er(j) = 1;
		// 	double x0j = xk(j);
		// 	for(int i = 0; i<xk.n_elem;i++){
		// 		ee.zeros();
		// 		ee(i) = 1;
		// 		double x0i = xk(i);
		// 		if(i==j)
		// 		{
		// 			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd3(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 			auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		// 		else
		// 		{
		// 			auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd3(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
		// 																										return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
		// 			ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);
		//
		// 		}
		//
		// 		// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
		// 		// 																						arma::vec lx = cj.lx;
		// 		// 																						return lx(j);
		// 		// 																					};
		// 	}
		// }
		// // cout<<"rk4zxd3\n";
		// // cout<<ddf__dxdx<<"\n";
		// // // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // // cout<<ddf__dxdxQ<<"\n";
		// // cout<<xd3ddx0r.slice(ind)<<"\n";
		// cout<<(ddf__dxdx-xd3ddx0r.slice(ind))<<"\n";
		// REQUIRE(arma::approx_equal(ddf__dxdx,xd3ddx0r.slice(ind) , "absdiff", 1e-04));

		ddf__dxdx.zeros();
		er.zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxkp1r(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxkp1r(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<"rk4zxkp1r\n";
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zxkp1r(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zxkp1r(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<"uu\n";
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxkp1r(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<"ux\n";
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));

		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		//Set expected output
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxkp1r(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		// cout<<"RK4xkp1r lx,ind: "<<ind<<"\n";
		// cout<<df__dx.t()<<"\n";
		// // cout<<df__dxQ.t()<<"\n";
		// cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zxkp1r(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		// cout<<"RK4xkp1r lu,ind: "<<ind<<"\n";
		// cout<<df__du.t()<<"\n";
		// cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));

	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"xkp1r reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 2e-06));
	}

}

//
TEST_CASE("Test rk4 x3 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x3 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx3(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zx3Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		// cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx3(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx3(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		// cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		// cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x3 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		// cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 2e-06));
	}
}


//
TEST_CASE("Test rk4 x3r Jacobians&Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x3r Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx3r(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx3rHessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::mat jx = std::get<3>(hess);
	arma::mat ju = std::get<4>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  // arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3r(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3r(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zx3r(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx3r(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;


		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3r(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));

		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		//Set expected output
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);
			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zx3r(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		// cout<<"RK4x3r lx,ind: "<<ind<<"\n";
		// cout<<df__dx.t()<<"\n";
		// // cout<<df__dxQ.t()<<"\n";
		// cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);
			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zx3r(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<"RK4x3r lu,ind: "<<ind<<"\n";
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));

	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x3r reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
	}

}


//
TEST_CASE("Test rk4 xd2 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 xd2 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.1});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zxd2(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	// std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zxd2Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd2(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd2(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


		arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
		er = 0*uk;
		ee = uk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<uk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double u0i = uk(i);
				if(i==j)
				{
					auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4zxd2(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}
				else
				{
					auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4zxd2(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudu<<"\n";
		// cout<<lkuu<<"\n";
		cout<<(ddf__dudu-lkuu)<<"\n";
		// cout<<xk.t()<<"\n";
		// cout<<uk.t()<<"\n";
		CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
		ddf__duduCube.slice(ind) = ddf__dudu;



		arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
		er = uk*0;
		ee = xk*0;
		for(int j = 0; j<uk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double u0j = uk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4zxd2(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																											return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
				ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dudx<<"\n";
		// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
		// cout<<lkux<<"\n";
		cout<<(ddf__dudx-lkux)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
	}
	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"xd2 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
	}
}

//
TEST_CASE("Test rk4 Hessians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 Hessians\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 =rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	std::tuple<vec,vec> rk4zout = rk4z(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
  arma::vec xd =std::get<0>(rk4zout);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zHessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;

	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);
				if(i==j)
				{
					auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4z_pure(1.0,xk+ ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
					auto dfxi = [=,&costset_tmp] (double xj) {return boost::math::differentiation::finite_difference_derivative(fxi,xj);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}
				else
				{
					auto dfxi = [=,&costset_tmp] (double xj) { 		auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4z_pure(1.0,xk+ ee*(xi-x0i)+ er*(xj-x0j),uk,sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fxi,x0i);};
					ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0j);

				}

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);
					if(i==j)
					{
						auto fui = [=,&costset_tmp] (double ui) {return  arma::dot(eind,rk4z_pure(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
						auto dfui = [=,&costset_tmp] (double uj) {return boost::math::differentiation::finite_difference_derivative(fui,uj);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}
					else
					{
						auto dfui = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4z_pure(1.0,xk,uk+ ee*(ui-u0i)+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																													return boost::math::differentiation::finite_difference_derivative(fui,u0i);};
						ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0j);

					}

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudu<<"\n";
			// cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			// cout<<xk.t()<<"\n";
			// cout<<uk.t()<<"\n";
			CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
			ddf__duduCube.slice(ind) = ddf__dudu;



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);
					auto dfxi = [=,&costset_tmp] (double uj) { 		auto fui = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4z_pure(1.0,xk+ ee*(xi-x0i),uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k));};
																												return boost::math::differentiation::finite_difference_derivative(fui,x0i);};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);



					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudx<<"\n";
			// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			// cout<<lkux<<"\n";
			cout<<(ddf__dudx-lkux)<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
		}
		arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
		arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
			cout<<"rk4z reduced cube check\n";
		for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
		{
			cout<<ind2<<"\n";
			// cout<<ddfQ__dudu.slice(ind2)<<"\n";
			// cout<<lkuuRedCube.slice(ind2)<<"\n";
			cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
			CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 2e-06));
		}
}



TEST_CASE("Test rk4 Hessians 2", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 Hessians 2\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	// for(int k = 0;k<3;k++){
	// 	sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	// }

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk);//,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = mk;//arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*mat33().eye().col(0);//arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = 1e-5*mat33().eye().col(0);//rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	std::tuple<vec,vec> rk4zout = rk4z(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
  arma::vec xd =std::get<0>(rk4zout);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube> hess = rk4zHessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);
	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);

				auto dfxi = [=,&costset_tmp] (double xi) {
																									std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zJacobians(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k);
																									arma::mat jfx = std::get<0>(jacsf);
																									return arma::as_scalar(eind.t()*jfx*er);
																								};
				ddf__dxdx += ee*er.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0i);

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// // arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// // ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// // cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-04));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);



					auto dfui = [=,&costset_tmp] (double ui) {
																										std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zJacobians(1.0,xk,uk+ ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k);
																										arma::mat jfu = std::get<1>(jacsf);
																										return arma::as_scalar(eind.t()*jfu*er);
																									};
					ddf__dudu += ee*er.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0i);
					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudu<<"\n";
			// cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			// cout<<xk.t()<<"\n";
			// cout<<uk.t()<<"\n";
			CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
			ddf__duduCube.slice(ind) = ddf__dudu;

			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);


					auto dfxi = [=,&costset_tmp] (double uj) {
																										std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zJacobians(1.0,xk,uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k);
																										arma::mat jfx = std::get<0>(jacsf);
																										return arma::as_scalar(eind.t()*jfx*ee);
																									};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudx<<"\n";
			// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			// cout<<lkux<<"\n";
			cout<<(ddf__dudx-lkux)<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-04));
		}
		arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
		arma::cube huuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
			cout<<"rk4z method 2 reduced cube check\n";
		for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
		{
			cout<<ind2<<"\n";
			// cout<<ddfQ__dudu.slice(ind2)<<"\n";
			// cout<<lkuuRedCube.slice(ind2)<<"\n";
			cout<<(ddfQ__dudu.slice(ind2)-huuRedCube.slice(ind2))<<"\n";
			CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  huuRedCube.slice(ind2), "absdiff", 1e-06));
		}
}



TEST_CASE("Test rk4 x2 Hessians 2", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic
	cout<<"rk4 x2 Hessians 2\n";



	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	// sat.plan_for_gg = false;
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.01,0.05,0.05})));
	// sat.change_Jcom(vecmat);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);

	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	arma::vec xd = rk4zx2(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zx2Jacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	std::tuple<arma::cube,arma::cube,arma::cube,arma::mat,arma::mat> hess = rk4zx2Hessians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);

	arma::cube hxx = std::get<0>(hess);
	arma::cube hux = std::get<1>(hess);
	arma::cube huu = std::get<2>(hess);

	arma::cube ddf__duduCube = 0*huu;


	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::mat lkuu = huu.slice(ind);
		arma::mat lkux = hux.slice(ind);
		arma::mat lkxx = hxx.slice(ind);
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();


		arma::mat ddf__dxdx = arma::mat(xk.n_elem,xk.n_elem).zeros();
		arma::vec er = arma::vec(xk.n_elem).zeros();
		ee = xk*0;
		for(int j = 0; j<xk.n_elem;j++){
			er.zeros();
			er(j) = 1;
			double x0j = xk(j);
			for(int i = 0; i<xk.n_elem;i++){
				ee.zeros();
				ee(i) = 1;
				double x0i = xk(i);

				auto dfxi = [=,&costset_tmp] (double xi) {
																									std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zx2Jacobians(1.0,xk + er*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k);
																									arma::mat jfx = std::get<0>(jacsf);
																									return arma::as_scalar(eind.t()*jfx*ee);
																								};
				ddf__dxdx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,x0i);

				// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
				// 																						arma::vec lx = cj.lx;
				// 																						return lx(j);
				// 																					};
			}
		}
		// cout<<ddf__dxdx<<"\n";
		// arma::mat ddf__dxdxQ = sat.findGMat(qk)*ddf__dxdx*sat.findGMat(qk).t();
		// ddf__dxdxQ(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);
		// cout<<ddf__dxdxQ<<"\n";
		// cout<<lkxx<<"\n";
		cout<<(ddf__dxdx-lkxx)<<"\n";
		REQUIRE(arma::approx_equal(ddf__dxdx,lkxx , "absdiff", 1e-06));


			arma::mat ddf__dudu = arma::mat(uk.n_elem,uk.n_elem).zeros();
			er = 0*uk;
			ee = uk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<uk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double u0i = uk(i);



					auto dfui = [=,&costset_tmp] (double ui) {
																										std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zx2Jacobians(1.0,xk,uk+ er*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k);
																										arma::mat jfu = std::get<1>(jacsf);
																										return arma::as_scalar(eind.t()*jfu*ee);
																									};
					ddf__dudu += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfui,u0i);
					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudu<<"\n";
			// cout<<lkuu<<"\n";
			cout<<(ddf__dudu-lkuu)<<"\n";
			// cout<<xk.t()<<"\n";
			// cout<<uk.t()<<"\n";
			CHECK(arma::approx_equal(ddf__dudu,lkuu , "both", 1e-03,1e-05));
			ddf__duduCube.slice(ind) = ddf__dudu;

			// ddfQ__dxdx(3,3,arma::size(3,3)) += arma::mat33().eye()*arma::dot(df__dx(arma::span(3,6)),qk);



			arma::mat ddf__dudx = arma::mat(uk.n_elem,xk.n_elem).zeros();
			er = uk*0;
			ee = xk*0;
			for(int j = 0; j<uk.n_elem;j++){
				er.zeros();
				er(j) = 1;
				double u0j = uk(j);
				for(int i = 0; i<xk.n_elem;i++){
					ee.zeros();
					ee(i) = 1;
					double x0i = xk(i);


					auto dfxi = [=,&costset_tmp] (double uj) {
																										std::tuple<arma::mat,arma::mat,arma::mat> jacsf = rk4zx2Jacobians(1.0,xk,uk+ er*(uj-u0j),sat,dynamics_info_kn1,dynamics_info_k);
																										arma::mat jfx = std::get<0>(jacsf);
																										return arma::as_scalar(eind.t()*jfx*ee);
																									};
					ddf__dudx += er*ee.t()*boost::math::differentiation::finite_difference_derivative(dfxi,u0j);

					// auto dfxi = [=,&costset_tmp] (double xi) {	cost_jacs cj = sat.quatcostJacobians(k, N, xk +  ee*(xi-x0i), uk, z3,satvec_k,ECIvec_k,BECI_k, &costset_tmp);
					// 																						arma::vec lx = cj.lx;
					// 																						return lx(j);
					// 																					};
				}
			}
			// cout<<ddf__dudx<<"\n";
			// arma::mat ddf__dudxQ = ddf__dudx*sat.findGMat(qk).t();
			// cout<<lkux<<"\n";
			cout<<(ddf__dudx-lkux)<<"\n";
			REQUIRE(arma::approx_equal(ddf__dudx,lkux , "absdiff", 1e-06));
		}

	arma::cube ddfQ__dudu = matOverCube(sat.findGMat(xd(span(3,6))),ddf__duduCube);
	arma::cube lkuuRedCube = matOverCube(sat.findGMat(xd(span(3,6))),huu);
		cout<<"x2 method 2 reduced cube check\n";
	for(int ind2 = 0; ind2<sat.state_N()-1;ind2++)
	{
		cout<<ind2<<"\n";
		// cout<<ddfQ__dudu.slice(ind2)<<"\n";
		// cout<<lkuuRedCube.slice(ind2)<<"\n";
		cout<<(ddfQ__dudu.slice(ind2)-lkuuRedCube.slice(ind2))<<"\n";
		CHECK(arma::approx_equal(ddfQ__dudu.slice(ind2),  lkuuRedCube.slice(ind2), "absdiff", 1e-06));
	}
}


TEST_CASE("Test RK4 jacobians", "[armadillo]") {
	//Set input
	//TODO tests of final step, magic,torque
	cout<<"RK4\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
	sat.change_Jcom(arma::mat33().eye());
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	sat.set_AV_constraint(1.0);
	sat.plan_for_gg = false;
	sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);


	COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	std::tuple<vec,vec> rk4zout = rk4z(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
  arma::vec xp1 =std::get<0>(rk4zout);
	std::tuple<arma::mat,arma::mat,arma::mat> jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	arma::mat jx = std::get<0>(jacs);
	arma::mat ju = std::get<1>(jacs);
	arma::mat jt = std::get<2>(jacs);

	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<"ind "<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		arma::vec lkt = jt.row(ind).t();
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);

			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4z_pure(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<"RK4 lx,ind: "<<ind<<"\n";
		cout<<df__dx.t()<<"\n";
		arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
		// cout<<df__dxQ.t()<<"\n";
		cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);

			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4z_pure(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<"RK4 lu,ind: "<<ind<<"\n";
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));



		}


		arma::arma_rng::set_seed_random();
		sat = Satellite();
		sat.change_Jcom(arma::diagmat(arma::vec({0.005,0.05,0.05})));
		sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
		sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
		sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
		sat.set_AV_constraint(1.0);
		sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
		sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
		// arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
		// arma::vec3 ams = 3e-3*arma::vec3().ones();
		// arma::vec3 js = 1e-4*arma::vec({0.01,0.02,0.5});
		// for(int k = 0;k<3;k++){
		// 	sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
		// }

	z3 = arma::vec3().zeros();
	qk = arma::normalise(arma::vec(4,fill::randn));
	wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	// hk = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	xk = join_cols(wk,qk);//,hk);
	mk = 0.1*arma::normalise(arma::vec(3,fill::randn));
	// arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	uk = mk;// arma::join_cols(mk,rwk);
	satvec_k = arma::vec({1,0,0});
	ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	BECI_k =  1e-4*arma::mat33().eye().col(0);// 1e-4*arma::normalise(arma::vec(3,fill::randn));
	brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	BECI_kp1 =   1e-4*arma::mat33().eye().col(0);//rotMat(brot)*BECI_k;
	R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));
	sunk = arma::normalise(arma::vec(3,fill::randn));
	// sat.plan_for_gg = false;
	k = 1;
	N = 10;

	dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	dynamics_info_k = std::make_tuple(BECI_kp1,R_k+1*V_k,0,V_k,sunk,1);


	costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);

	rk4zout = rk4z(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	xp1 =std::get<0>(rk4zout);
  // xp1 = rk4z(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	jacs = rk4zJacobians(1.0,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	jx = std::get<0>(jacs);
	ju = std::get<1>(jacs);
	jt = std::get<2>(jacs);

	for(int ind = 0; ind<sat.state_N();ind++)
	{
		cout<<"ind "<<ind<<"\n";
		arma::vec eind = arma::vec(sat.state_N()).zeros();
		eind(ind) = 1.0;
		arma::vec lku = ju.row(ind).t();
		arma::vec lkx = jx.row(ind).t();
		arma::vec lkt = jt.row(ind).t();
		//Set expected output
		arma::vec ee = xk*0;
	  arma::vec df__dx = arma::vec(xk.n_elem).zeros();

		for(int i = 0; i<xk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double x0i = xk(i);

			auto fxi = [=,&costset_tmp] (double xi) {return arma::dot(eind,rk4z_pure(1.0,xk + ee*(xi-x0i),uk,sat,dynamics_info_kn1,dynamics_info_k));};
			df__dx += ee*boost::math::differentiation::finite_difference_derivative(fxi,x0i);
		}
		cout<<"RK4 lx,ind: "<<ind<<"\n";
		cout<<df__dx.t()<<"\n";
		arma::vec df__dxQ = sat.findGMat(qk)*df__dx;
		// cout<<df__dxQ.t()<<"\n";
		cout<<lkx.t()<<"\n";
		cout<<(df__dx-lkx).t()<<"\n";
		REQUIRE(arma::approx_equal(df__dx,lkx , "both", 1e-06,1e-10));


		ee = uk*0;
	  arma::vec df__du = arma::vec(uk.n_elem).zeros();
		for(int i = 0; i<uk.n_elem;i++){
			ee.zeros();
			ee(i) = 1;
			double u0i = uk(i);

			auto fui = [=,&costset_tmp] (double ui) {return arma::dot(eind,rk4z_pure(1.0,xk,uk + ee*(ui-u0i),sat,dynamics_info_kn1,dynamics_info_k));};
			df__du += ee*boost::math::differentiation::finite_difference_derivative(fui,u0i);
		}
		cout<<"RK4 lu,ind: "<<ind<<"\n";
		cout<<df__du.t()<<"\n";
		cout<<lku.t()<<"\n";
		cout<<(df__du-lku).t()<<"\n";
		REQUIRE(arma::approx_equal(df__du,lku , "both", 1e-06,1e-10));



		}
}



TEST_CASE("Test test_J_update_w_RW", "[Satellite]") {
	//Set input
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	sat.change_Jcom(arma::diagmat(arma::vec({0.1,100,5})));
	arma::vec3 torqs = arma::vec({0.01,0.05,0.02});
	arma::vec3 ams = 0.1*arma::vec3().ones();
	arma::vec3 js = arma::vec({0.001,0.002,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::mat J = {{0.1,0,0},{0,100,0},{0,0,5}};
  arma::mat invJ ={{10,0,0},{0,0.01,0},{0,0,0.2}};
  arma::mat J_noRW ={{0.099,0,0},{0,99.998,0},{0,0,4.5}};
  arma::mat invJ_noRW ={{1/0.099,0,0},{0,1/99.998,0},{0,0,2.0/9.0}};
	REQUIRE(arma::approx_equal(sat.Jcom, J, "absdiff", 1e-10));
	REQUIRE(arma::approx_equal(sat.invJcom, invJ, "absdiff", 1e-10));
	REQUIRE(arma::approx_equal(sat.Jcom_noRW, J_noRW, "absdiff", 1e-10));
	REQUIRE(arma::approx_equal(sat.invJcom_noRW, invJ_noRW, "absdiff", 1e-10));
}

TEST_CASE("Test dynamics", "[armadillo]") {
	cout<<"DYNAMICS\n";
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	arma::mat33 Jcom = arma::diagmat(arma::vec({0.005,0.05,0.05}));
	sat.change_Jcom(Jcom);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	// sat.set_AV_constraint(1.0);
	// sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	// sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 prop_torq = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 gd_torq = 0.01*arma::normalise(arma::vec(3,fill::randn));

	// cout<<prop_torq.t()<<"\n";
	// cout<<gd_torq.t()<<"\n";
	// cout<<(prop_torq+gd_torq).t()<<"\n";

	sat.add_prop_torq(prop_torq);
	sat.add_gendist_torq(gd_torq);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-2*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::mat33 invJ = (Jcom-arma::diagmat(js)).i();

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = 0.01*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = 1e-3*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = arma::vec3().zeros();//0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = 2e-5*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	// DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,0,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k,1,V_k,sunk,1);


	// COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	// arma::vec xd = sat.dynamics(xk,uk,dynamics_info_k);

	std::tuple<arma::vec,arma::vec> out = sat.dynamics(xk,uk,dynamics_info_k);
	arma::vec xd =std::get<0>(out);
	arma::vec tqd = std::get<1>(out);

	arma::vec3 wd_exp = (invJ)*(MAGRW_TORQ_MULT*rwk + -1.0*cross(wk, (Jcom)*wk + hk)  + prop_torq+gd_torq );
	arma::vec4 qd_exp = 0.5*findWMat(qk)*wk;
	arma::vec3 hd_exp = -MAGRW_TORQ_MULT*rwk - diagmat(js)*wd_exp;

	arma::vec xd_exp =  arma::join_cols(wd_exp,qd_exp,hd_exp);

	cout<<tqd.t()<<"\n";
	cout<<prop_torq.t()<<"\n";
	cout<<gd_torq.t()<<"\n";
	cout<<(prop_torq+gd_torq).t()<<"\n";

	cout<<(prop_torq+gd_torq-tqd).t()<<"\n";


	cout<<xd_exp.t()<<"\n";
	cout<<xd.t()<<"\n";
	cout<<(xd-xd_exp).t()<<"\n";

	CHECK(arma::approx_equal(tqd,prop_torq+gd_torq ,"both", 1e-08,1e-10));
	CHECK(arma::approx_equal(xd,xd_exp ,"both", 1e-08,1e-10));


}



TEST_CASE("Test RK4z simple", "[armadillo]") {
	arma::arma_rng::set_seed_random();
	Satellite sat = Satellite();
	arma::mat33 vecmat = arma::mat33().eye();
	arma::mat33 Jcom = arma::diagmat(arma::vec({0.005,0.05,0.05}));
	arma::mat33 invJ = arma::diagmat(arma::vec({1.0/0.005,1.0/0.05,1.0/0.05}));
	sat.change_Jcom(Jcom);
	sat.add_MTQ(arma::vec({1,0,0}), 0.2, 0.1);
	sat.add_MTQ(arma::vec({0,1,0}), 0.5, 0.1);
	sat.add_MTQ(arma::vec({0,0,1}), 0.5, 0.1);
	// sat.set_AV_constraint(1.0);
	// sat.add_sunpoint_constraint(vecmat.col(0),20.0*datum::pi/180.0,false);
	// sat.add_sunpoint_constraint(vecmat.col(2),10.0*datum::pi/180.0,true);
	arma::vec3 prop_torq = 0.0001*arma::vec({1,0,0});;
	arma::vec3 gd_torq = 0.0001*arma::vec({-3,0,0});;
	sat.add_prop_torq(prop_torq);
	sat.add_gendist_torq(gd_torq);
	arma::vec3 torqs = arma::vec({1e-4,2e-4,5e-5});
	arma::vec3 ams = 3e-3*arma::vec3().ones();
	arma::vec3 js = 1e-2*arma::vec({0.01,0.02,0.5});
	for(int k = 0;k<3;k++){
		sat.add_RW(vecmat.col(k),js(k),torqs(k),ams(k),1,1,10,0,0.01);
	}

	arma::vec3 z3 = arma::vec3().zeros();
	arma::vec4 qk = arma::normalise(arma::vec(4,fill::randn));
	arma::vec3 wk = z3;//0.001*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 hk = z3;//1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec xk = join_cols(wk,qk,hk);
	arma::vec3 mk = arma::vec3().zeros();//0.1*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 rwk = z3;//2e-6*arma::normalise(arma::vec(3,fill::randn));
	arma::vec uk = arma::join_cols(mk,rwk);
	arma::vec3 satvec_k = arma::vec({1,0,0});
	arma::vec3 ECIvec_k = arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 BECI_k = 1e-4*arma::normalise(arma::vec(3,fill::randn));
	arma::vec4 brot = arma::normalise(arma::join_cols(vec(1).ones()*cos(0.001),arma::normalise(arma::vec(3,fill::randn))*sin(0.001)));
	arma::vec3 BECI_kp1 = rotMat(brot)*BECI_k;
	arma::vec3 R_k = 7000*arma::normalise(arma::vec(3,fill::randn));
	arma::vec3 V_k = 7000*arma::normalise(arma::cross(R_k,arma::vec(3,fill::randn)));
	arma::vec3 sunk = arma::normalise(arma::vec(3,fill::randn));

	int k = 1;
	int N = 10;

	double dt = 0.00001;
	DYNAMICS_INFO_FORM dynamics_info_kn1 = std::make_tuple(BECI_k,R_k,1,V_k,sunk,1);

	DYNAMICS_INFO_FORM dynamics_info_k = std::make_tuple(BECI_kp1,R_k+dt*V_k,1,V_k,sunk,1);


	// COST_SETTINGS_FORM costset_tmp = std::make_tuple(1e3,1e0,1e-1,3,5,1.0,1e6,1e3,1e1,1e1,0,1);
	// arma::vec xd = sat.dynamics(xk,uk,dynamics_info_k);

	std::tuple<arma::vec,arma::vec> out = rk4z(dt,xk,uk,sat,dynamics_info_kn1,dynamics_info_k);
	arma::vec xkp1 =std::get<0>(out);
	arma::vec tqd = std::get<1>(out);

	arma::vec3 wd_exp = invJ*(rwk + -1.0*cross(wk, Jcom*wk + hk)  + prop_torq+gd_torq );
	arma::vec4 qd_exp = 0.5*findWMat(qk)*wk;
	arma::vec3 hd_exp = -MAGRW_TORQ_MULT*rwk - diagmat(js)*wd_exp;

	arma::vec dx = arma::join_cols(wd_exp,qd_exp,hd_exp);

	arma::vec xd_exp =  xk+dt*dx;

	xd_exp(arma::span(3,6)) = normalise(xd_exp(arma::span(3,6)));

	cout<<"RK4 TORQ\n";
	cout<<tqd.t()<<"\n";
	cout<<prop_torq.t()<<"\n";
	cout<<gd_torq.t()<<"\n";
	cout<<(prop_torq+gd_torq).t()<<"\n";

	cout<<(prop_torq+gd_torq-tqd).t()<<"\n";


	cout<<xkp1.t()<<"\n";
	cout<<xd_exp.t()<<"\n";
	cout<<(xkp1-xd_exp).t()<<"\n";
	cout<<dt*dx.t()<<"\n";


	REQUIRE(arma::approx_equal(tqd,prop_torq+gd_torq ,"both", 1e-08,1e-10));
	REQUIRE(arma::approx_equal(xkp1,xd_exp ,"both", 1e-08,1e-10));


}

/*TEST_CASE("Test dynamicsJacobians", "[csv][armadillo]") {
	//Read inputs
	rapidcsv::Document docU0("../test_io/dynamicsJacobiansTest_51121_input_U0.csv", rapidcsv::LabelParams(-1, -1));
	arma::vec3 u = csvToArma(docU0);
	rapidcsv::Document docx0("../test_io/dynamicsJacobiansTest_51121_input_x0.csv", rapidcsv::LabelParams(-1, -1));
	arma::vec7 x = csvToArma(docx0);
	rapidcsv::Document docB0("../test_io/dynamicsJacobiansTest_51121_input_B0.csv", rapidcsv::LabelParams(-1, -1));
	arma::vec3 Bk = csvToArma(docB0);
	rapidcsv::Document docJ("../test_io/dynamicsJacobiansTest_51121_input_J.csv", rapidcsv::LabelParams(-1, -1));
	arma::mat33 J = csvToArma(docJ);
  arma::mat33 invJ = inv(J);
  arma::mat33 skewSymU = 2*invJ*skewSymmetric(u);
	//Call dynamicsJacobians
    arma::vec3 Rk = arma::vec({0,0,1e12});
    arma::vec3 pt = arma::vec({0,0,0});
  std::tuple<arma::mat, arma::mat> jac = dynamicsJacobians(x, u, J, invJ, Bk,Rk,pt);
  arma::mat jxx = std::get<0>(jac);
  arma::mat jxu = std::get<1>(jac);
	//Read outputs
	rapidcsv::Document docjacX("../test_io/dynamicsJacobiansTest_51121_output_jacX.csv", rapidcsv::LabelParams(-1, -1));
	arma::mat jxx_expected = csvToArma(docjacX);
	rapidcsv::Document docjacU("../test_io/dynamicsJacobiansTest_51121_output_jacU.csv", rapidcsv::LabelParams(-1, -1));
	arma::mat jxu_expected = csvToArma(docjacU);
	//Assert output == expected output to machine precision
	for(int i = 0; i < jxx.n_rows; i++){
		REQUIRE(arma::approx_equal(jxx.row(i), jxx_expected.row(i), "absdiff", arma::datum::eps));
	}
	for(int i = 0; i < jxu.n_rows; i++){
		REQUIRE(arma::approx_equal(jxu.row(i), jxu_expected.row(i), "absdiff", arma::datum::eps));
	}
}*/
//
// // TEST_CASE("Test findQ (in point mode, as that is what we primarily use)", "[armadillo]") {
// // 	//Set input
// // 	int Nslew = 1800;
// //   double sv1 = 100000.0;
// //   double swpoint = 1000.0;
// //   double swslew = 0.05;
// //   double sratioslew = 0.01;
// //   std::tuple<int, double, double, double, double> qSettingsArma = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
// // 	arma::mat matrix_out = OldPlanner::findQ(1801,  &qSettingsArma);
// // 	//Set expected
// // 	arma::mat66 matrix_expected = arma::mat66().zeros();
// // 	matrix_expected(0, 0) = swpoint;
// //   matrix_expected(1, 1) = swpoint;
// //   matrix_expected(2, 2) = swpoint;
// //   matrix_expected(3, 3) = sv1;
// //   matrix_expected(4, 4) = sv1;
// //   matrix_expected(5, 5) = sv1;
// // 	//Assert output == expected output to machine precision
// // 	for(int i = 0; i < matrix_out.n_rows; i++){
// // 		REQUIRE(arma::approx_equal(matrix_out.row(i), matrix_expected.row(i), "absdiff", arma::datum::eps));
// // 	}
// // }
//
// TEST_CASE("Test bdot", "[armadillo][csv]") {
// 	//Read inputs
// 	rapidcsv::Document docB("../test_io/bdotTest_5821_input.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Bset = trans(csvToArma(docB));
//   arma::vec x0 = {0.000188538709250411, 0.000445091331190917, 0.00133589165953042, -0.86430183415298, 0.446928929910215, 0.224735597608838, 0.0522568871681439};
//   double bdotgain = 100000.0;
//   arma::vec umax = {0.15, 0.15, 0.15};
//   double dt = 1.0;
//   arma::mat33 J = arma::mat({{0.03136490806,            -0.00671361357,               5.88304e-05},
//             {-0.00671361357,             0.01004091997,           -0.00012334756},
//                {5.88304e-05,            -0.00012334756,             0.03409127827}});
//
//   arma::mat33 invJ = inv(J);
//
//   int N = 3600;
//   arma::vec3 Rv = arma::vec({0,0,1e15});
//   arma::mat Rset = arma::repmat(Rv,1,N);
//   arma::vec3 V = arma::vec({0,0,1e15});
//   arma::mat Vset = arma::repmat(V,1,N);
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = arma::repmat(s,1,N);
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//   double wmax  = 0.1;
//   double mu0 = 0;
//   arma::mat33 R = arma::mat33().eye();
//   R = R*500;
//
//   int Nslew = 0;
//   double sv1 = 500.0;
//   double swpoint = 0.328280635001174;
//   double swslew = pow(10,-6);
//   double sratioslew = pow(10, -3);
//
//   arma::mat77 QN = arma::mat77().eye();
//   QN = QN*swpoint;
//   QN(4, 4) = sv1;
//   QN(5, 5) = sv1;
//   QN(6, 6) = sv1;
//
//   std::tuple<int, double, double, double, double> qSettings = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
//   arma::mat ECIvec = arma::normalise(Vset);
//   arma::mat satvec = ECIvec*0;
//   arma::vec3 pt = arma::vec({0,0,0});
//   std::tuple<arma::mat,double> bdotout = OldPlanner::bdot(Bset,Rset, N, x0, bdotgain, umax, dt, J, invJ,Vset,sunset,R,QN,wmax,Nslew,qSettings,mu0,ECIvec,satvec,pt,c,sunang);
//   arma::mat Uset = std::get<0>(bdotout);
// 	//Read expected output
// 	rapidcsv::Document docU("../test_io/bdotTest_5821_output.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset_expected = csvToArma(docU);
// 	//Assert output == expected output to machine precision
// 	for(int i = 0; i < Uset.n_rows; i++){
// 		REQUIRE(arma::approx_equal(Uset.row(i), Uset_expected.row(i), "absdiff", 1e-8));
// 	}
// }
//
// TEST_CASE("Test backwardPass (with zero lambdaset)", "[armadillo][csv]") {
// 	//Set inputs
// 	rapidcsv::Document docJ("../test_io/backwardPassTest_5821_input_J.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat33 J = csvToArma(docJ);
//   double rho = 0.0;
//   double drho = 0.0;
//   double mu = 40.0;
//   double regMin = pow(10, -8);
//   double regScale = 1.6;
//
//   int Nslew = 0;
//   double sv1 = 500.0;
//   double swpoint = 0.328280635001174;
//   double swslew = pow(10,-6);
//   double sratioslew = pow(10, -3);
//   std::tuple<int, double, double, double, double> qSettings = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
//
//   arma::mat33 R = arma::mat33().eye();
//   R = R*500;
//
//   arma::mat77 QN = arma::mat77().eye();
//   QN = QN*swpoint;
//   QN(4, 4) = sv1;
//   QN(5, 5) = sv1;
//   QN(6, 6) = sv1;
//
//   arma::vec3 umax_arma = arma::vec3().ones();
//   umax_arma = umax_arma*0.15;
//
//   int length_slew = 3600;
// 	rapidcsv::Document docXset("../test_io/backwardPassTest_5821_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset = csvToArma(docXset);
// 	rapidcsv::Document docUset("../test_io/backwardPassTest_5821_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset = csvToArma(docUset);
// 	rapidcsv::Document docRset("../test_io/backwardPassTest_5821_input_Rset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Rset = trans(csvToArma(docRset));
// 	rapidcsv::Document docBset("../test_io/backwardPassTest_5821_input_Bset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Bset = trans(csvToArma(docBset));
// 	rapidcsv::Document docVset("../test_io/backwardPassTest_5821_input_Vset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Vset = trans(csvToArma(docVset));
// 	//rapidcsv::Document docXdesset("../test_io/backwardPassTest_5821_input_Xdesset.csv", rapidcsv::LabelParams(-1, -1));
// 	//arma::mat Xdesset = csvToArma(docXdesset);
// 	rapidcsv::Document doclambdaSet("../test_io/backwardPassTest_5821_input_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat lambdaSet = csvToArma(doclambdaSet);
// 	//rapidcsv::Document docmuSet("../test_io/backwardPassTest_5821_input_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat muSet = 0*lambdaSet + mu;
//   Nslew = 0;
//   arma::mat ECIvec = arma::normalise(Vset);
//   arma::mat satvec = ECIvec*0;
//   satvec.each_row() += arma::vec({0, 0, -1});
//   double dt = 1.0;
//
//
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = ECIvec*0;
//   sunset.each_row() += s;
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//
//   arma::mat33 invJ = inv(J);
//   double wmax = 0.00872664625997165;
// 	//Call backwardPass
//     arma::vec3 pt = arma::vec({0,0,0});
// 	std::tuple<arma::cube, arma::mat, arma::mat, double, double> backwardPassOut = OldPlanner::backwardPass(Xset, Uset, Vset,Rset, Bset,sunset, lambdaSet, rho, drho, mu, muSet, dt, regScale, regMin, R, QN, umax_arma, wmax, J, &invJ, Nslew, satvec, ECIvec, &qSettings,pt,c,sunang);
// 	arma::cube Kset_arma = std::get<0>(backwardPassOut);
//   arma::mat dset_arma = std::get<1>(backwardPassOut);
//   arma::mat delV_arma = std::get<2>(backwardPassOut);
//   double rho_arma = std::get<3>(backwardPassOut);
//   double drho_arma = std::get<4>(backwardPassOut);
// 	//Reshape Kset for comparison
//   arma::mat Kset_arma_matrix = arma::mat(18, Kset_arma.n_slices).zeros();
//   for(int k = 0; k < Kset_arma.n_slices; k++)
//   {
//     arma::mat Kmatrix = Kset_arma.slice(k);
//     for (size_t rowtest=0; rowtest < 6; rowtest++)
//     {
//       for (size_t coltest=0; coltest < 3; coltest++)
//       {
//         size_t i = rowtest*3+coltest;
//         Kset_arma_matrix(i, k) = Kmatrix(coltest, rowtest);
//       }
//     }
//   }
// 	//Set expected outputs
// 	rapidcsv::Document docKset("../test_io/backwardPassTest_5821_output_Kset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Kset_expected = csvToArma(docKset);
// 	rapidcsv::Document docdset("../test_io/backwardPassTest_5821_output_dset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat dset_expected = csvToArma(docdset);
// 	rapidcsv::Document docdelV("../test_io/backwardPassTest_5821_output_delV.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat delV_expected = csvToArma(docdelV);
//   double rho_expected = 0.0;
//   double drho_expected = 0.0;
// 	//Assert equality within 1e-8 to 1e-10 depending on output
// 	for(int i = 0; i < Kset_arma_matrix.n_cols; i++){
// 		REQUIRE(arma::approx_equal(Kset_arma_matrix.col(i), Kset_expected.col(i), "absdiff", 1e-9));
// 	}
// 	for(int i = 0; i < dset_arma.n_cols; i++){
// 		REQUIRE(arma::approx_equal(dset_arma.col(i), dset_expected.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < delV_arma.n_cols; i++){
// 		REQUIRE(arma::approx_equal(delV_arma.col(i), delV_expected.col(i), "absdiff", 1e-8));
// 	}
// 	REQUIRE(pow(rho_arma-rho_expected,2)<1e-8);
// 	REQUIRE(pow(drho_arma-drho_expected,2)<1e-8);
// }
//
// TEST_CASE("Test backwardPass (with nonzero lambdaset)", "[armadillo][csv]") {
// 	//Set inputs
// 	rapidcsv::Document docJ("../test_io/backwardPassNonzeroLambdasetTest_51121_input_J.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat33 J = csvToArma(docJ);
//   double rho = 0.0;
//   double drho = 0.0;
//   //double mu = 40.0;
// 	rapidcsv::Document docmu("../test_io/backwardPassNonzeroLambdasetTest_51121_input_mu.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat muMat = csvToArma(docmu);
// 	double mu = muMat(0,0);
//   double regMin = pow(10, -8);
//   double regScale = 1.6;
//
//   int Nslew = 0;
//   double sv1 = 500.0;
//   double swpoint = 0.328280635001174;
//   double swslew = pow(10,-6);
//   double sratioslew = pow(10, -3);
//   std::tuple<int, double, double, double, double> qSettings = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
//
//   arma::mat33 R = arma::mat33().eye();
//   R = R*500;
//
//   arma::mat77 QN = arma::mat77().eye();
//   QN = QN*swpoint;
//   QN(4, 4) = sv1;
//   QN(5, 5) = sv1;
//   QN(6, 6) = sv1;
//
//   arma::vec3 umax_arma = arma::vec3().ones();
//   umax_arma = umax_arma*0.15;
//
//   int length_slew = 3600;
// 	rapidcsv::Document docXset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset = csvToArma(docXset);
// 	rapidcsv::Document docUset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset = csvToArma(docUset);
// 	rapidcsv::Document docRset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Rset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Rset = trans(csvToArma(docRset));
// 	rapidcsv::Document docBset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Bset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Bset = trans(csvToArma(docBset));
// 	rapidcsv::Document docVset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Vset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Vset = trans(csvToArma(docVset));
// 	//rapidcsv::Document docXdesset("../test_io/backwardPassNonzeroLambdasetTest_51121_input_Xdesset.csv", rapidcsv::LabelParams(-1, -1));
// 	//arma::mat Xdesset = csvToArma(docXdesset);
// 	rapidcsv::Document doclambdaSet("../test_io/backwardPassNonzeroLambdasetTest_51121_input_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat lambdaSet = csvToArma(doclambdaSet);
//
// 	arma::mat muSet = 0*lambdaSet + mu;
//   Nslew = 0;
//   arma::vec satAlignVector = arma::vec({0, 0, -1});
//   arma::mat ECIvec = arma::normalise(Vset);
//   arma::mat satvec = ECIvec*0;
//   satvec.each_row() += arma::vec({0, 0, -1});
//   double dt = 1.0;
//
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = ECIvec*0;
//   sunset.each_row() += s;
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//
//   arma::mat33 invJ = inv(J);
//   double wmax = 0.00872664625997165;
// 	//Call backwardPass
//     arma::vec3 pt = arma::vec({0,0,0});
// 	std::tuple<arma::cube, arma::mat, arma::mat, double, double> backwardPassOut = OldPlanner::backwardPass(Xset, Uset, Vset, Rset,Bset, sunset,lambdaSet, rho, drho, mu, muSet,dt, regScale, regMin, R, QN, umax_arma, wmax, J, &invJ, Nslew, satvec,ECIvec, &qSettings,pt,c,sunang);
// 	arma::cube Kset_arma = std::get<0>(backwardPassOut);
//   arma::mat dset_arma = std::get<1>(backwardPassOut);
//   arma::mat delV_arma = std::get<2>(backwardPassOut);
//   double rho_arma = std::get<3>(backwardPassOut);
//   double drho_arma = std::get<4>(backwardPassOut);
// 	//Reshape Kset for comparison
//   arma::mat Kset_arma_matrix = arma::mat(18, Kset_arma.n_slices).zeros();
//   for(int k = 0; k < Kset_arma.n_slices; k++)
//   {
//     arma::mat Kmatrix = Kset_arma.slice(k);
//     for (size_t rowtest=0; rowtest < 6; rowtest++)
//     {
//       for (size_t coltest=0; coltest < 3; coltest++)
//       {
//         size_t i = rowtest*3+coltest;
//         Kset_arma_matrix(i, k) = Kmatrix(coltest, rowtest);
//       }
//     }
//   }
// 	//Set expected outputs
// 	rapidcsv::Document docKset("../test_io/backwardPassNonzeroLambdasetTest_51121_output_Kset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Kset_expected = csvToArma(docKset);
// 	rapidcsv::Document docdset("../test_io/backwardPassNonzeroLambdasetTest_51121_output_dset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat dset_expected = csvToArma(docdset);
// 	rapidcsv::Document docdelV("../test_io/backwardPassNonzeroLambdasetTest_51121_output_delV.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat delV_expected = csvToArma(docdelV);
//   double rho_expected = 0.0;
//   double drho_expected = 0.0;
// 	//Assert equality within 1e-8 to 1e-10 depending on output
// 	for(int i = 0; i < Kset_arma_matrix.n_cols; i++){
// 		REQUIRE(arma::approx_equal(Kset_arma_matrix.col(i), Kset_expected.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < dset_arma.n_cols; i++){
// 		REQUIRE(arma::approx_equal(dset_arma.col(i), dset_expected.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < delV_arma.n_cols; i++){
// 		REQUIRE(arma::approx_equal(delV_arma.col(i), delV_expected.col(i), "absdiff", 1e-8));
// 	}
// 	REQUIRE(pow(rho_arma-rho_expected,2)<1e-8);
// 	REQUIRE(pow(drho_arma-drho_expected,2)<1e-8);
// }
//
// /*
// TEST_CASE("Test cost function (with nonzero lambdaSet)", "[csv][armadillo]") {
// 	int Nslew = 0;
//   double sv1 = 500;//10000.0;
//   double su = 500;
//   double swpoint = 0.328280635001174;//3.2828*pow(10,5);
//   double swslew =  0.000001;//1.0000*pow(10, -4);
//   double sratioslew = 0.0001;
//   std::tuple<int, double, double, double, double> qSettingsArma = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
//   arma::vec satAlignVector = arma::vec({0, 0, -1});
//   arma::vec vNslew = arma::vec({-3.6894, -3.0127, 6.0019});
//   arma::vec umax = arma::vec({0.15, 0.15, 0.15});
//   arma::mat R = arma::mat(3,3).eye()*su;
//
//   double dt = 1.0;
//   double mu = 40.0;
//
// 	rapidcsv::Document docXset("../test_io/costTest_51021_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset = csvToArma(docXset);
//   int N = Xset.n_cols;
//   arma::mat QN_velCon = OldPlanner::findQ(N-1, &qSettingsArma);
//
// 	rapidcsv::Document docUset("../test_io/costTest_51021_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset = csvToArma(docUset);
//
// 	rapidcsv::Document docRset("../test_io/costTest_51021_input_Rset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Rset = trans(csvToArma(docRset));
//
// 	rapidcsv::Document docVset("../test_io/costTest_51021_input_Vset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Vset = trans(csvToArma(docVset));
//
// 	arma::mat Bset = 0.0*Vset;
//
// 	//rapidcsv::Document docXdesset("../test_io/costTest_51021_input_Xdesset.csv", rapidcsv::LabelParams(-1, -1));
// 	//arma::mat Xdesset = csvToArma(docXdesset);
//
//   double wmax = 0.00872664625997165;
//
// 	rapidcsv::Document docLambdaset("../test_io/costTest_51021_input_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat lambdaSetVelCon = csvToArma(docLambdaset);
//
// 	arma::mat muSet = 0*lambdaSetVelCon + mu;
//
//       arma::mat ECIvec = arma::normalise(Vset);
//       arma::mat satvec = ECIvec*0;
//       satvec.each_row() += arma::vec({0, 0, -1});
//
//
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = arma::repmat(s,1,N);
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//
// 	//Call veccostFunc
// 	//std::cout<<"about to call veccostfunc...\n";
//   double LA = OldPlanner::veccostFunc(Xset, Uset, Vset,Bset,sunset, lambdaSetVelCon, mu,muSet, dt, QN_velCon, R, umax, wmax, vNslew, satvec,ECIvec, Nslew, &qSettingsArma,c,sunang);
//
//   //Define expected output
//   double LA_ex = 3503940.61112495;//48013831107.0956;
// 	//std::cout<<"LA "<<LA<<"\n";
// 	//Assert equality to 1e-10
// 	REQUIRE(pow(LA_ex-LA, 2)<1e-10);
// }
// */
// // TEST_CASE("Test forwardPass", "[csv][armadillo]") {
// // 	//Set inputs
// // 	int length_slew = 3600;
// // 	rapidcsv::Document docXset("../test_io/forwardPassTest_51021_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat Xset = csvToArma(docXset);
// // 	rapidcsv::Document docdset("../test_io/forwardPassTest_51021_input_dset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat dset = csvToArma(docdset);
// // 	rapidcsv::Document docKset("../test_io/forwardPassTest_51021_input_Kset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat K_reshape = csvToArma(docKset);
// // 	rapidcsv::Document docUset("../test_io/forwardPassTest_51021_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat Uset = csvToArma(docUset);
// // 	rapidcsv::Document docBset("../test_io/forwardPassTest_51021_input_Bset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat Bset = trans(csvToArma(docBset));
// // 	rapidcsv::Document docRset("../test_io/forwardPassTest_51021_input_Rset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat Rset = trans(csvToArma(docRset));
// // 	rapidcsv::Document docVset("../test_io/forwardPassTest_51021_input_Vset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat Vset = trans(csvToArma(docVset));
// //   arma::mat lambdaSet = arma::mat(13, length_slew+1).zeros();
// //
// // 	arma::mat muSet = 0*lambdaSet + 40.0;
// // 	rapidcsv::Document docdelV("../test_io/forwardPassTest_51021_input_delV.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat delV = csvToArma(docdelV);
// // 	//rapidcsv::Document docXdesset("../test_io/forwardPassTest_51021_input_Xdesset.csv", rapidcsv::LabelParams(-1, -1));
// // 	//arma::mat Xdesset = csvToArma(docXdesset);
// // 	rapidcsv::Document docLA("../test_io/forwardPassTest_51021_input_LA.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat LAmat = csvToArma(docLA);
// // 	double LA = LAmat(0,0);
// //   double alpha = 0.01;
// //   double dt = 1.0;
// // 	rapidcsv::Document docJ("../test_io/forwardPassTest_51021_input_J.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat33 J = csvToArma(docJ);
// //   int Nslew = 0;
// //   double sv1 = 500.0;
// //   double swpoint = 0.328280635001174;
// //   double swslew = pow(10,-4);
// //   double sratioslew = pow(10, -3);
// //   double su = 500.0;
// //   double wmax = 0.00872664625997165;
// //   std::tuple<int, double, double, double, double> qSettings = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
// //
// //   arma::mat33 R = arma::mat33().eye();
// //   R = R*su;
// //
// //   arma::mat77 QN = arma::mat77().eye();
// //   QN = QN*swpoint;
// //   QN(4, 4) = sv1;
// //   QN(5, 5) = sv1;
// //   QN(6, 6) = sv1;
// //
// //
// //   int maxLsIter = 10;
// //   double beta1 = pow(10, -8);
// //   double beta2 = 10;
// //   double regScale = 1.6;
// //   double regMin = pow(10, -8);
// //   double regBump = 1000;
// //   arma::vec umax = {0.15, 0.15, 0.15};
// //   arma::vec xmax = arma::vec7().ones()*10;
// //   double eps = arma::datum::eps;
// //   arma::vec vNslew = {-3.6894, -3.0127, 6.0019};
// //   arma::vec satAlignVector = {0, 0, -1};
// //
// //   arma::mat ECIvec = arma::normalise(Vset);
// //   arma::mat satvec = ECIvec*0;
// //   satvec.each_row() += arma::vec({0, 0, -1});
// //   std::tuple<int, double, double, double, double, double, arma::vec, arma::vec, double, arma::vec, arma::mat,  arma::mat,int, double> forwardPassSettings = std::make_tuple(maxLsIter, beta1, beta2, regScale, regMin, regBump, umax, xmax, eps, vNslew, satvec,ECIvec, Nslew, wmax);
// //   //Reshape Kset
// //   arma::cube Kset = arma::cube(3, 6, 3599);
// //   for(int k = 0; k < 3599; k++)
// //   {
// //     arma::vec Kcol = K_reshape.col(k);
// //     arma::mat Kmat = arma::mat(3, 6);
// //     for (size_t rowtest=0; rowtest < 6; rowtest++)
// //     {
// //       for (size_t coltest=0; coltest < 3; coltest++)
// //       {
// //         size_t i = rowtest*3+coltest;
// //         Kmat(coltest, rowtest) = Kcol(i);
// //       }
// //     }
// //     Kset.slice(k) = Kmat;
// //   }
// //   arma::mat33 invJ = inv(J);
// //
// //   //Call forwardpass
// //   arma::vec3 s = arma::vec({0,0.5,0.5});
// //   arma::mat sunset = ECIvec*0;
// //   sunset.each_row() += s;
// //   arma::vec c = arma::vec({1,0,0});
// //   double sunang = 0;
// //  arma::vec3 pt = arma::vec({0,0,0});
// //   std::tuple<arma::mat, arma::mat, double, double, double> forwardPassOut = OldPlanner::forwardPass(Xset, Uset, Kset, Vset, Rset, Bset, sunset,lambdaSet, dset, delV, J, &invJ, R, QN, LA, 0.0, 0.0, 40.0, muSet, 1.0, &qSettings, forwardPassSettings,pt,c,sunang,1e20);
// //
// //   arma::mat newX = std::get<0>(forwardPassOut);
// //   arma::mat newU = std::get<1>(forwardPassOut);
// //   double newLA = std::get<2>(forwardPassOut);
// // 	//Get expected output
// // 	rapidcsv::Document docNewx("../test_io/forwardPassTest_51021_output_Xset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat newX_expected = csvToArma(docNewx);
// // 	rapidcsv::Document docNewu("../test_io/forwardPassTest_51021_output_Uset.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat newU_expected = csvToArma(docNewu);
// // 	rapidcsv::Document docNewLA("../test_io/forwardPassTest_51021_output_newLA.csv", rapidcsv::LabelParams(-1, -1));
// // 	arma::mat newLAmat = csvToArma(docNewLA);
// //   double newLA_expected = newLAmat(0,0);
// // 	//Assert equality to 1e-5 to 1e-10 depending on output
// // 	REQUIRE(pow(newLA_expected-newLA, 2)<1e-5);
// // 	for(int i = 0; i < newX.n_cols; i++){
// // 		REQUIRE(arma::approx_equal(newX.col(i), newX_expected.col(i), "absdiff", 1e-10));
// // 	}
// // 	for(int i = 0; i < newU.n_cols; i++){
// // 		REQUIRE(arma::approx_equal(newU.col(i), newU_expected.col(i), "absdiff", 1e-10));
// // 	}
// // }
//
// /*TEST_CASE("Test maxViol", "[csv][armadillo]") {
// 	//Read in inputs
// 	rapidcsv::Document docXset("../test_io/maxViolTest_51021_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset = csvToArma(docXset);
// 	rapidcsv::Document docUset("../test_io/maxViolTest_51021_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset = csvToArma(docUset);
// 	rapidcsv::Document doclambdaSet("../test_io/maxViolTest_51021_input_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat lambdaSet = csvToArma(doclambdaSet);
//
// 	rapidcsv::Document docMu("../test_io/maxViolTest_51021_input_mu.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat muMat = csvToArma(docMu);
// 	double mu = muMat(0,0);
//   arma::vec umax = {0.15, 0.15, 0.15};
//   double wmax = 0.00872664625997165;
// 	arma::mat muSet = 0*lambdaSet + mu;
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = Uset*0;
//   sunset.each_row() += s;
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//
//   //Call maxViol
//   std::tuple<arma::mat, double> viol = OldPlanner::maxViol(Xset, Uset,sunset, lambdaSet, mu, muSet,umax, wmax,c,sunang);
//
//   arma::mat clist = std::get<0>(viol);
//   double cmaxtmp = std::get<1>(viol);
// 	//Define expected output
// 	rapidcsv::Document docClist("../test_io/maxViolTest_51021_output_clist.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat clist_ex = csvToArma(docClist);
// 	rapidcsv::Document docCmaxtmp("../test_io/maxViolTest_51021_output_cmaxtmp.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat cmaxtmpMat = csvToArma(docCmaxtmp);
// 	double cmaxtmp_ex = cmaxtmpMat(0,0);
// 	//Assert equality to machine precision
// 	REQUIRE(fabs(cmaxtmp-cmaxtmp_ex) < arma::datum::eps);
// 	for(int i = 0; i < clist.n_cols; i++){
// 		REQUIRE(arma::approx_equal(clist.col(i), clist_ex.col(i), "absdiff", arma::datum::eps));
// 	}
// }*/
// /*
// TEST_CASE("Test alilqr (with fewer iterations)", "[csv][armadillo]") {
// 	//Assign inputs
// 	int length_slew = 3600;
// 	rapidcsv::Document docXset("../test_io/alilqrTest_51021_input_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset = csvToArma(docXset);
// 	rapidcsv::Document docUset("../test_io/alilqrTest_51021_input_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset = csvToArma(docUset);
// 	rapidcsv::Document docBset("../test_io/alilqrTest_51021_input_Bset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Bset = trans(csvToArma(docBset));
// 	rapidcsv::Document docRset("../test_io/alilqrTest_51021_input_Rset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Rset = trans(csvToArma(docRset));
// 	rapidcsv::Document docVset("../test_io/alilqrTest_51021_input_Vset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Vset = trans(csvToArma(docVset));
//   arma::mat lambdaSet = arma::mat(13, length_slew+1).zeros();
// 	//rapidcsv::Document docXdesset("../test_io/alilqrTest_51021_input_Xdesset.csv", rapidcsv::LabelParams(-1, -1));
// 	//arma::mat Xdesset = csvToArma(docXdesset);
//
//   double dt = 1.0;
// 	rapidcsv::Document docJ("../test_io/alilqrTest_51021_input_J.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat33 J = csvToArma(docJ);
// 	int Nslew = 0;
//   double sv1 = 500.0;
//   double swpoint = 0.328280635001174;
//   double su = 500.0;
//   double swslew = pow(10,-4);
//   double sratioslew = pow(10, -3);
//   std::tuple<int, double, double, double, double> qSettings = std::make_tuple(Nslew, sv1, swpoint, swslew, sratioslew);
//
//   arma::mat33 R = arma::mat33().eye();
//   R = R*su;
//
//   arma::mat77 QN = arma::mat77().eye();
//   QN = QN*swpoint;
//   QN(4, 4) = sv1;
//   QN(5, 5) = sv1;
//   QN(6, 6) = sv1;
//   double QNmult = 1;
//
//
//   int maxLsIter = 10;
//   double beta1 = pow(10, -8);
//   double beta2 = 10;
//   double regScale = 1.6;
//   double regMin = pow(10, -8);
//   double regBump = 1000.0;
//   arma::vec umax = {0.15, 0.15, 0.15};
//   arma::vec xmax = arma::vec7().ones()*10;
//   double eps = 2.2204e-16;
//   arma::vec vNslew = {-3.6894, -3.0127, 6.0019};
//   arma::vec satAlignVector = {0, 0, -1};
//   double wmax = 0.5*arma::datum::pi/180;
//
//   arma::mat ECIvec = arma::normalise(Vset);
//   arma::mat satvec = ECIvec*0;
//   satvec.each_row() += arma::vec({0, 0, -1});
//   std::tuple<int, double, double, double, double, double, arma::vec, arma::vec, double, arma::vec,  arma::mat, arma::mat, int, double> forwardPassSettingsWmax = std::make_tuple(maxLsIter, beta1, beta2, regScale, regMin, regBump, umax, xmax, eps, vNslew,satvec,ECIvec, Nslew, wmax);
//
//   int lagMultInit = 0;
//   double penInit = 100.0;
//   int regInit = 0;
//   int maxOuterIter = 2;
//   int maxIlqrIter = 1;
//   double gradTol = 1e-05;
//   double costTol = 0.0001;
//   double cmax = 0.001;
//   int zCountLim = 10;//10;
//   int maxIter = 700;
//   double penMax = 1.0e+18;
//   double penScale = 20;
//   double lagMultMax = 1e+10;
//   double ilqrCostTol = 0.001;
//   std::tuple<int, double, int, int, int, double, double, double, int, int, double, double, double, double> alilqrSettings = std::make_tuple(lagMultInit, penInit, regInit, maxOuterIter, maxIlqrIter, gradTol, costTol, cmax, zCountLim, maxIter, penMax, penScale, lagMultMax, ilqrCostTol);
//
//   //Call alilqr
//     arma::vec3 pt = arma::vec({0,0,0});
//   arma::vec3 s = arma::vec({0,0.5,0.5});
//   arma::mat sunset = ECIvec*0;
//   sunset.each_row() += s;
//   arma::vec c = arma::vec({1,0,0});
//   double sunang = 0;
//   std::tuple<arma::mat, arma::mat, arma::mat, arma::cube, double, double> alilqrOut = OldPlanner::alilqr(Xset, Uset, Rset, Vset, Bset, sunset, dt, J, R, QN, QNmult, qSettings, forwardPassSettingsWmax, alilqrSettings,pt,c,sunang);
//
//   arma::mat Xset_out = std::get<0>(alilqrOut);
//   arma::mat Uset_out = std::get<1>(alilqrOut);
//   arma::mat lambdaSet_out = std::get<2>(alilqrOut);
//   arma::cube Kset_arma = std::get<3>(alilqrOut);
//   double mu_out = std::get<4>(alilqrOut);
// 	//Assign expected outputs
// 	rapidcsv::Document docXset_ex("../test_io/alilqrTest_51021_output_Xset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Xset_ex = csvToArma(docXset_ex);
// 	rapidcsv::Document docUset_ex("../test_io/alilqrTest_51021_output_Uset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Uset_ex = csvToArma(docUset_ex);
// 	rapidcsv::Document doclambdaSet_ex("../test_io/alilqrTest_51021_output_lambdaSet.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat lambdaSet_ex = csvToArma(doclambdaSet_ex);
// 	rapidcsv::Document docKset("../test_io/alilqrTest_51021_output_Kset.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat Kset_reshape_ex = csvToArma(docKset);
// 	rapidcsv::Document docMumat("../test_io/alilqrTest_51021_output_mu.csv", rapidcsv::LabelParams(-1, -1));
// 	arma::mat muMat = csvToArma(docMumat);
//   double mu_ex = muMat(0,0);
//   //Reshape Kset for comparison
//   arma::mat Kset_arma_matrix = arma::mat(18, Kset_arma.n_slices).zeros();
//   for(int k = 0; k < Kset_arma.n_slices; k++)
//   {
//     arma::mat Kmatrix = Kset_arma.slice(k);
//     for (size_t rowtest=0; rowtest < 6; rowtest++)
//     {
//       for (size_t coltest=0; coltest < 3; coltest++)
//       {
//         size_t i = rowtest*3+coltest;
//         Kset_arma_matrix(i, k) = Kmatrix(coltest, rowtest);
//       }
//     }
//   }
// 	//Assert equality to 1 to 1e-10 depending on output
// 	REQUIRE(pow(mu_ex-mu_out,2)<1e1);
// 	for(int i = 0; i < lambdaSet_ex.n_cols; i++){
// 		REQUIRE(arma::approx_equal(lambdaSet_ex.col(i), lambdaSet_out.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < Xset_ex.n_cols; i++){
// 		REQUIRE(arma::approx_equal(Xset_ex.col(i), Xset_out.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < Uset_ex.n_cols; i++){
// 		REQUIRE(arma::approx_equal(Uset_ex.col(i), Uset_out.col(i), "absdiff", 1e-10));
// 	}
// 	for(int i = 0; i < Kset_arma_matrix.n_cols; i++){
// 		REQUIRE(arma::approx_equal(Kset_arma_matrix.col(i), Kset_reshape_ex.col(i), "absdiff", 1e-9));
// 	}
// }
// */
//
// /*TEST_CASE("Run trajectory planner without checking output", "[json][armadillo]") {
// 	OldPlanner::trajOpt("../../trajOptSettings.json", 3600, 10.0, 10, 11, 2022, 13.4, 10, 11, 2022, arma::vec({0, 0, 0, 0.5, 0.5, 0.5, 0.5}), 1);
// 	REQUIRE(1==1);
// }*/
