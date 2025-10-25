#include "ArmaCSV.hpp"
#include <cstdint>

void armaToCsv(const arma::mat& matrix, rapidcsv::Document& document) {
	arma::uword rows = matrix.n_rows;
	arma::uword cols = matrix.n_cols;
	document.Clear();
	for(arma::uword i = 0; i < rows; i++) {
		for(arma::uword j = 0; j < cols; j++) {
			document.SetCell<double>(j, i, matrix(i, j));
		}
	}
}

void armaToCsvWithTimes(const arma::mat& matrix, const arma::mat& timeMatrix, rapidcsv::Document& document) {
	int rows = matrix.n_rows+timeMatrix.n_rows;
	arma::mat combinedMat(rows, matrix.n_cols);
	for(size_t i = 0; i < rows; i++) {
		for(size_t j = 0; j < matrix.n_cols; j++) {
			if(i < 4) {
				combinedMat(i, j) = timeMatrix(i, j);
			} else {
				combinedMat(i, j) = matrix(i-4, j);
			}
		}
	}
	return armaToCsv(combinedMat, document);
}

arma::mat addTimesToArma(const arma::mat& matrix, const arma::mat& timeMatrix) {
	int rows = matrix.n_rows+timeMatrix.n_rows;
	arma::mat combinedMat(rows, matrix.n_cols);
	for(size_t i = 0; i < rows; i++) {
		for(size_t j = 0; j < matrix.n_cols; j++) {
			if(i < timeMatrix.n_rows) {
				combinedMat(i, j) = timeMatrix(i, j);
			} else {
				combinedMat(i, j) = matrix(i-timeMatrix.n_rows, j);
			}
		}
	}
	return combinedMat;
}

arma::mat csvToArma(const rapidcsv::Document& document) {
	size_t cols = document.GetColumnCount();
	size_t rows = document.GetRowCount();
	arma::mat newMatrix(rows, cols);
	for(size_t i = 0; i < rows; i++) {
		for(size_t j = 0; j < cols; j++) {
			// rapidcsv specifies column-first, unlike armadillo
			newMatrix(i, j) = document.GetCell<double>(j, i);
		}
	}
	return newMatrix;
}

arma::mat timeTruncateArma(arma::mat mat_long, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end) {
	//Extract time params
	double dh_start = std::get<0>(t_start);
	int day_start = std::get<1>(t_start);
	int mon_start = std::get<2>(t_start);
	int yr_start = std::get<3>(t_start);
	double dh_end = std::get<0>(t_end);
	int day_end = std::get<1>(t_end);
	int mon_end = std::get<2>(t_end);
	int yr_end = std::get<3>(t_end);
	//Find start and end index in matrix
	int ind_tstart = 0;
	int ind_tend = 0;
	for(int i = 0; i < mat_long.n_rows; i++){
		if(mat_long(i,0) <= dh_start && mat_long(i,1) <= day_start && mat_long(i,2) <= mon_start && mat_long(i,3) <= yr_start) {
			ind_tstart = i;
		}
	}
	for(int i = 0; i < mat_long.n_rows; i++) {
		if(mat_long(i, 0) <= dh_end && mat_long(i, 1) <= day_end && mat_long(i, 2) <= mon_end && mat_long(i, 3) <= yr_end) {
			ind_tend = i;
		}
	}
	//Find dt from read-in array (take delta time in decimal hours + days and then convert to seconds)
	//Assumes dt_readin is always less than one day. Should make this safe with J2000 but ok for now.
	//TODO ^^^^^ vvvv fix this
	int dt_readin = std::round((mat_long(1, 0)-mat_long(0, 0))*3600) +std::round((mat_long(1, 1)-mat_long(0, 1))*86400);
	//N = # of seconds between t_start and t_end, adjust by dt to get array size
	int traj_length = floor(N/dt);
	//Adjust dt to what we want (INVARIANT: dt is a whole-number mulitple of dt_readin)
	//arma::mat mat_times = arma::mat(ceil(traj_length/dt), mat_long.n_cols);
	arma::mat mat_times = arma::mat(traj_length, mat_long.n_cols);
	arma::uvec inds = arma::regspace<arma::uvec>(ind_tstart,dt/dt_readin,ind_tend);
	mat_times = mat_long.rows(inds);
	//Return
	return mat_times;
}

arma::mat timeTruncateCsvToArma(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end) {
		arma::mat mat_long = trans(csvToArma(document));
		return timeTruncateArma(mat_long, dt, N, t_start, t_end);
}

arma::mat timeAwareCsvToArma(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end) {
		arma::mat mat_times = timeTruncateCsvToArma(document, dt, N, t_start, t_end);
		arma::mat final_mat = mat_times.rows(4, mat_times.n_rows-1);
		return final_mat;
}

arma::mat timeAwareArma(arma::mat mat_long, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end) {
		arma::mat mat_times = timeTruncateArma(mat_long, dt, N, t_start, t_end);
		arma::mat final_mat = mat_times.cols(4, mat_times.n_cols-1);
		return final_mat;
}

arma::mat extractRelevantTimes(arma::mat mat_long, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end) {
		arma::mat mat_times = timeTruncateArma(mat_long, dt, N, t_start, t_end);
		arma::mat final_mat = mat_times.cols(0, 3);
		return final_mat;
}

arma::mat extractRelevantTimesCsv(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end)
{
	arma::mat mat_times = timeTruncateCsvToArma(document, dt, N, t_start, t_end);
	//arma::mat final_mat = mat_times.rows(4, mat_times.n_cols-1)
	return mat_times.rows(0, 3);
}
