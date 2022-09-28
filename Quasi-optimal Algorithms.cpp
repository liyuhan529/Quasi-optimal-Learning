#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <RcppClock.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppClock)]]

using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]

double Q_func(double a, double alpha1, double alpha2, double alpha3){
	double Q = alpha1 * pow(a,2) + alpha2 * a + alpha3;
	return Q;
}

// [[Rcpp::export]]
double Q_int(double alpha1, double alpha2, double alpha3, double lb, double ub){
	double ub_val = alpha1 * pow(ub,3)/3 + alpha2 * pow(ub,2)/2 + alpha3 * ub;
	double lb_val = alpha1 * pow(lb,3)/3 + alpha2 * pow(lb,2)/2 + alpha3 * lb;
	double result = ub_val - lb_val;
	return result;
}

// [[Rcpp::export]]
double pi_policy(double a, double lambda, double alpha1, double alpha2, double alpha3, double lb, double ub){
	double K = ub - lb;
	double pi_a;
	if (a >= lb & a <= ub){
		pi_a = Q_func(a,alpha1,alpha2,alpha3)/(2*lambda) - Q_int(alpha1,alpha2,alpha3,lb,ub)/(2*lambda*K)+1/K;
	}
	else{
		pi_a = 0;
	}
	return pi_a;
}

// [[Rcpp::export]]
double V_func(double lambda, double alpha1, double alpha2, double alpha3, double lb, double ub){
	double K = ub - lb;
	// arma::vec X = linspace<vec>(lb, ub, 50);
	// arma::vec Y(50);
	// for (int i = 0; i < 50; i++){
	// 	Y[i] = pow((Q_int(alpha1,alpha2,alpha3,lb,ub)/(2*lambda*K)-1/K),2)-pow((Q_func(X[i],alpha1,alpha2,alpha3)/(2*lambda)),2);
	// }
	// 
	// arma::mat Z = arma::trapz(X,Y);
	// double result = Z(0,0);
	// double V = lambda * (1-result);
	// return V;
	double tmp1 = pow((Q_int(alpha1,alpha2,alpha3,lb,ub)/(2*lambda*K)-1/K),2)*K;
	double tmp2 = pow(alpha1,2)*(pow(ub,5)-pow(lb,5))/5+alpha1*alpha2*(pow(ub,4)-pow(lb,4))/2+
		            (pow(alpha2,2)+2*alpha1*alpha3)*(pow(ub,3)-pow(lb,3))/3+alpha2*alpha3*(pow(ub,2)-pow(lb,2))+pow(alpha3,2)*(ub-lb);
	double result = lambda*(1-tmp1+tmp2/(4*pow(lambda,2)));
	return result;
}


// [[Rcpp::export]]
arma::mat gau_kernel(const arma::mat& S, 
                     const arma::vec& A,
                     double sigma, double sigma_a)
{
	int N = S.n_rows;
	double diag = 0;
	arma::mat kernel_matrix(N, N);
	
	for (int i = 0; i < N; i++)
	{
		kernel_matrix(i, i) = diag;
		for (int j = 0; j < i; j ++)
		{
			kernel_matrix(j,i) = exp(-sum(pow(S.row(i)-S.row(j),2))/sigma/sigma-sum(pow(A(i)-A(j),2))/sigma_a/sigma_a);
			kernel_matrix(i,j) = kernel_matrix(j,i);
		}
	}
	
	return kernel_matrix;
}

// [[Rcpp::export]]
double big_psi(arma::vec w, arma::vec s, double lambda){
	double k0 = 0.5;
	double b0 = 0.5;
	double z = -(5*lambda)/(1+exp(-k0*(sum(w % s))-b0));
	return z;
}


// [[Rcpp::export]]
double small_psi(double a, double lambda, double alpha1, double alpha2, double alpha3, double lb, double ub){
	double K = ub - lb;
	double psi_a;
	if (a < lb || a > ub){
		psi_a = -Q_func(a,alpha1,alpha2,alpha3)/(2*lambda) + Q_int(alpha1,alpha2,alpha3,lb,ub)/(2*lambda*K) - 1/K;
	}
	else{
		psi_a = 0;
	}
	return psi_a;
}


// [[Rcpp::export]]


arma::mat basis_func(arma::mat S,arma::vec nodes){
	int nn = S.n_rows;
	int p = S.n_cols;
	arma::mat basis(nn, 9*p+1, fill::ones);
	for (int i = 0; i < nn; i++){
		for (int j=0; j < p; j++){
			basis(i, 9*j+1) = S(i,j);
			basis(i, 9*j+2) = pow(S(i,j),2);
			basis(i, 9*j+3) = pow(S(i,j),3);
			basis(i, 9*j+4) = pow(std::max(0.0,S(i,j)-nodes(0)),  3);
			basis(i, 9*j+5) = pow(std::max(0.0,S(i,j)-nodes(1)),  3);
			basis(i, 9*j+6) = pow(std::max(0.0,S(i,j)-nodes(2)),  3);
			basis(i, 9*j+7) = pow(std::max(0.0,S(i,j)-nodes(3)),  3);
			basis(i, 9*j+8) = pow(std::max(0.0,S(i,j)-nodes(4)),  3);
			basis(i, 9*j+9) = pow(std::max(0.0,S(i,j)-nodes(5)),  3);
		}
	}
	return basis;
}

// [[Rcpp::export]]
arma::mat bounds(arma::vec alpha1_vec, arma::vec alpha2_vec, double lambda){
	double n = alpha1_vec.n_elem;
	arma::vec tmp = -alpha2_vec/(2*alpha1_vec);
	arma::vec lb = (-alpha2_vec+pow((12*pow(alpha1_vec,2)*lambda),1/3.))/(2*alpha1_vec);
	arma::vec ub = (-alpha2_vec-pow((12*pow(alpha1_vec,2)*lambda),1/3.))/(2*alpha1_vec);
	arma::mat result(n,2);
	result.col(0) = lb;
	result.col(1) = ub;
	return result;
	
}


// [[Rcpp::export]]
double loss_func(arma::vec theta, arma::mat S, arma::vec A, arma::vec R,  double lambda, double gamma, arma::mat basis,
                 double n, double stage_num, double sigma, double sigma_a, arma::vec nodes){
	
	//theta is the initial value of the params, with length of 28*p+3
	int p  = S.n_cols;
	arma::vec theta1 = theta.subvec(0,9*p);
	arma::vec theta2 = theta.subvec(9*p+1,18*p+1);
	arma::vec theta3 = theta.subvec(18*p+2,27*p+2);
	arma::vec w = theta.subvec(27*p+3,28*p+2);
	
	//normalize the data
	
	//for (int i = 0; i < S.n_cols; i++){
	//	for (int j=0; j < S.n_rows; j++){
	//		S(j,i) = (S(j,i)-S.col(i).min()) / (S.col(i).max()-S.col(i).min());
	//	}
	//}
	
	//A = (A - A.min()) / (A.max() - A.min());
	
	// find the ub, lb for each sample
	arma::vec alpha1_vec = -exp(basis * theta1);
	arma::vec alpha2_vec = basis * theta2;
	arma::vec alpha3_vec = basis * theta3;
	arma::vec tmp = -alpha2_vec / (2*alpha1_vec);
	

  arma::mat res = bounds(alpha1_vec,alpha2_vec,lambda);
	arma::vec lb = res.col(0);
	arma::vec ub = res.col(1);
	
	
	arma::vec lu(n);
	for (int i = 0; i < n; i++){
		arma::uvec ind = arma::regspace<uvec>(i,n,i+(stage_num-1)*n);
		arma::uvec ind1 = arma::regspace<uvec>(i,n,i+(stage_num-2)*n);
		arma::mat S_tmp = S.rows(ind);
		arma::vec A_tmp = A(ind);
		arma::vec R_tmp = R(ind1);
		arma::vec result_tmp1(stage_num-1);
		
		for (int j = 0; j < stage_num-1; j++){
			result_tmp1(j) = R_tmp(j)+gamma*V_func( lambda, alpha1_vec(ind(j+1)), alpha2_vec(ind(j+1)), alpha3_vec(ind(j+1)), lb(ind(j+1)), ub(ind(j+1)))+lambda-
				2*lambda*pi_policy(A_tmp(j), lambda, alpha1_vec(ind(j)), alpha2_vec(ind(j)), alpha3_vec(ind(j)), lb(ind(j)), ub(ind(j)))-
				big_psi(w, S_tmp.row(j).t(), lambda)+small_psi(A_tmp(j),lambda,alpha1_vec(ind(j)),alpha2_vec(ind(j)),alpha3_vec(ind(j)),lb(ind(j)),ub(ind(j)))-
				V_func( lambda, alpha1_vec(ind(j)), alpha2_vec(ind(j)), alpha3_vec(ind(j)), lb(ind(j)), ub(ind(j)));
		}
		// arma::mat result_tmp2= result_tmp1 * result_tmp1.t();
		// arma::uvec ind_tmp = arma::trimatl_ind(size(result_tmp2),-1);
		// arma::vec  result_tmp3 = result_tmp2(ind_tmp);
		// arma::vec  gau_dist = gau_kernel(S.rows(ind1), A(ind1), sigma, sigma_a);
		// lu (i) = mean(result_tmp3 % gau_dist);
		arma::mat omega = gau_kernel(S.rows(ind1), A(ind1), sigma, sigma_a);
		double result_tmp = arma::as_scalar(result_tmp1.t()*omega*result_tmp1);
    lu(i) = result_tmp/((stage_num-1)*(stage_num-2));
	}
	double result = mean(lu);
	return result;
}

// [[Rcpp::export]]

double loss_func_samp(arma::vec theta, arma::mat S, arma::vec A, arma::vec R, double lambda, double gamma, arma::mat basis,
                      double n, double stage_num, double sigma, double sigma_a, arma::vec nodes, int index1){
	// index1 is the index of trajectory
	//Rcpp::Clock clock;
	//clock.tick("base");
	int p  = S.n_cols;
	arma::vec theta1 = theta.subvec(0,9*p);
	arma::vec theta2 = theta.subvec(9*p+1,18*p+1);
	arma::vec theta3 = theta.subvec(18*p+2,27*p+2);
	arma::vec w = theta.subvec(27*p+3,28*p+2);

	//normalize the data
	
	//	for (int i = 0; i < S.n_cols; i++){
	//		for (int j=0; j < S.n_rows; j++){
	//			S(j,i) = (S(j,i)-S.col(i).min()) / (S.col(i).max()-S.col(i).min());
	//		}
	//	}
	
	//	A = (A - A.min()) / (A.max() - A.min());
	
	// find the ub, lb for each sample

	arma::uvec ind1_tmp1 = arma::regspace<uvec>(index1,n,index1+(stage_num-1)*n);
	arma::uvec ind1_tmp2 = arma::regspace<uvec>(index1,n,index1+(stage_num-2)*n);
	
	arma::mat S_tmp = S.rows(ind1_tmp1);
	arma::vec A_tmp = A(ind1_tmp1);
	arma::vec R_tmp = R(ind1_tmp2);
	
	arma::mat base = basis.rows(ind1_tmp1);
	
	arma::vec alpha1_tmp = -exp(base * theta1);
	arma::vec alpha2_tmp = base * theta2;
	arma::vec alpha3_tmp = base * theta3;
	
	//clock.tock("base");
	//clock.tick("error");
	
	// arma::vec alpha1_tmp = alpha1_vec(ind1_tmp1);
	// arma::vec alpha2_tmp = alpha2_vec(ind1_tmp1);
	// arma::vec alpha3_tmp = alpha3_vec(ind1_tmp1);
	// 
	arma::mat res = bounds(alpha1_tmp,alpha2_tmp,lambda);
	arma::vec lb_tmp = res.col(0);
	arma::vec ub_tmp = res.col(1);
	
	arma::vec result_tmp1(stage_num-1);
	
	for (int j = 0; j < stage_num-1; j++){
		result_tmp1(j) = R_tmp(j)+gamma*V_func(lambda, alpha1_tmp(j+1), alpha2_tmp(j+1), alpha3_tmp(j+1), lb_tmp(j+1), ub_tmp(j+1))+lambda-
			2*lambda*pi_policy(A_tmp(j), lambda, alpha1_tmp(j), alpha2_tmp(j), alpha3_tmp(j), lb_tmp(j), ub_tmp(j))-
			big_psi(w, S_tmp.row(j).t(), lambda)+small_psi(A_tmp(j),lambda,alpha1_tmp(j),alpha2_tmp(j),alpha3_tmp(j),lb_tmp(j), ub_tmp(j))-
			V_func(lambda, alpha1_tmp(j), alpha2_tmp(j), alpha3_tmp(j), lb_tmp(j), ub_tmp(j));
	}
	arma::mat omega = gau_kernel(S.rows(ind1_tmp2), A(ind1_tmp2), sigma, sigma_a);
	double result_tmp = arma::as_scalar(result_tmp1.t()*omega*result_tmp1);
	double lu = result_tmp/((stage_num-1)*(stage_num-2));
	//clock.tock("error");
	//clock.stop("ss");
	return lu;
	
}

// [[Rcpp::export]]

arma::vec loss_func_grad(arma::vec theta, arma::mat S, arma::vec A, arma::vec R, double lambda, double gamma, arma::mat basis, 
                         double n, double stage_num, double sigma, double sigma_a, arma::vec nodes, int index1, double epsilon){
	
	int nn = theta.n_elem;
	arma::vec grad(nn);
	for (int i = 0; i < nn; i++){
		arma::vec theta_tmp1 = theta;
		arma::vec theta_tmp2 = theta;	
		theta_tmp1(i) = theta(i)+epsilon;
		theta_tmp2(i) = theta(i)-epsilon;
		double tmp = loss_func_samp(theta_tmp1, S, A, R, lambda, gamma, basis,n, stage_num, sigma, sigma_a, nodes, index1)-
			           loss_func_samp(theta_tmp2, S, A, R, lambda, gamma, basis,n, stage_num, sigma, sigma_a, nodes, index1);
		grad(i) = tmp/(2*epsilon);
	}
	return grad;
	
}

// [[Rcpp::export]]

arma::vec loss_func_bg(arma::vec theta, arma::mat S, arma::vec A, arma::vec R,double lambda, double gamma, arma::mat basis,
                       double n, double stage_num, double sigma, double sigma_a, arma::vec nodes, int batch1, double epsilon){
	//batch1 is the number of trajectory to be selected
	//batch2 is the number of time points pair to be selected from each trajectory
	
	// randomly select the trajectory
	
	arma::uvec ind = randperm(n, batch1);
	
	arma::mat grad_tmp;
	for (int i = 0; i < batch1; i++){
		arma::vec grad_tmp1 = loss_func_grad(theta, S, A, R,  lambda, gamma, basis, n, stage_num, sigma, sigma_a, nodes, ind(i), epsilon);
		grad_tmp = join_rows(grad_tmp,grad_tmp1);
	}
	arma::vec grad = mean(grad_tmp,1);
	return(grad);
	
}


// [[Rcpp::export]]

arma::vec loss_func_bgd(arma::vec theta, arma::mat S, arma::vec A, arma::vec R,  double lambda, double gamma, arma::mat basis,
                        double n, double stage_num, double sigma, double sigma_a, arma::vec nodes, int batch1, double epsilon,int maxit, double tol,double lr,double desc_rate){
	arma::vec theta_old = theta;
	for (int i = 0; i < maxit; i++){
		double lr_tmp = lr/(1+desc_rate*sqrt(i));
		arma::vec theta_new = theta_old - lr_tmp*loss_func_bg(theta_old, S, A, R, lambda, gamma, basis,
                                                    n, stage_num, sigma, sigma_a, nodes, batch1, epsilon);
		if(sqrt(sum(pow(theta_old-theta_new,2)))<=tol){
			Rcout<<"Algorithm stopped at ieration"<<i<<endl;
			break;
		}
		if (i % 100 == 0) {
			Rcout<< loss_func(theta_new,S, A, R, lambda, gamma, basis, n, stage_num, sigma, sigma_a, nodes)<<endl;
		}
		theta_old = theta_new;
	}
	return theta_old;
}

