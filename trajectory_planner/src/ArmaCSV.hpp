#ifndef TPR_ARMACSV_HPP
#define TPR_ARMACSV_HPP

#include <armadillo>
#include "rapidcsv.h"

// document is an out parameter (All data in the document will be CLEARED.)
// Also, one must manually save the changes to the document, this function
// will not do that for you.
void armaToCsv(const arma::mat& matrix, rapidcsv::Document& document);

arma::mat csvToArma(const rapidcsv::Document& document);

arma::mat timeTruncateArma(arma::mat, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

arma::mat addTimesToArma(const arma::mat& matrix, const arma::mat& timeMatrix);

arma::mat timeTruncateCsvToArma(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

arma::mat timeAwareCsvToArma(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

arma::mat timeAwareArma(arma::mat mat_long, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

arma::mat extractRelevantTimes(arma::mat mat_long, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

arma::mat extractRelevantTimesCsv(const rapidcsv::Document& document, int dt, int N, std::tuple<double, int, int, int> t_start, std::tuple<double, int, int, int> t_end);

void armaToCsvWithTimes(const arma::mat& matrix, const arma::mat& timeMatrix, rapidcsv::Document& document);
#endif // TPR_ARMACSV_HPP
