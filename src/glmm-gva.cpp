// [[Rcpp::plugins(openmp, cpp14)]]
// [[Rcpp::depends(RcppArmadillo, psqn, RcppEigen)]]
#define PSQN_USE_EIGEN
#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include "psqn-Rcpp-wrapper.h"
#include "psqn.h"
#include "psqn-reporter.h"
#include <cmath>
#include <array>
#include <limits>
#include <memory.h>
#include <algorithm>
#include <numeric>
#include <R_ext/RS.h> // F77_... macros

#ifdef FC_LEN_T
# define FCLEN ,FC_LEN_T
# define FCONE ,(FC_LEN_T)1
#else
# define FCLEN
# define FCONE
#endif

#ifdef DO_PROF
#include <gperftools/profiler.h>
#endif

/* contains functions to computed the needed expectations of the log partition 
 * functions */
#include "log-partition.h"

// implement the lower bound 

/// simple function to avoid copying a vector. You can ignore this
inline arma::vec vec_no_cp(double const * x, arma::uword const n_ele){
  return arma::vec(const_cast<double *>(x), n_ele, false);
}

/** Computes LL^T where L is a lower triangular matrix. The argument is a
 a vector with the non-zero elements in column major order. The diagonal 
 entries are on the log scale. The method computes both L and LL^T.  */
void get_pd_mat(double const *theta, arma::mat &L, arma::mat &res){
  unsigned const dim{L.n_rows};
  L.zeros();
  for(unsigned j = 0; j < dim; ++j){
    L.at(j, j) = std::exp(*theta++);
    for(unsigned i = j + 1; i < dim; ++i)
      L.at(i, j) = *theta++;
  }
  
  res = L * L.t();
}

struct dpd_mat {
  /**
   * return the required memory to get the derivative as part of the chain
   * rule  */
  static size_t n_wmem(unsigned const dim){
    return dim * dim;
  }

  /**
   * computes the derivative w.r.t. theta as part of the chain rule. That is,
   * the derivatives of f(LL^T) where d/dX f(X) evaluated at X = LL^T
   * is supplied.
   */
  static void get(arma::mat const &L, double * __restrict__ res,
                  double const * derivs, double * __restrict__ wk_mem){
    unsigned const dim{L.n_rows};
    arma::mat D(const_cast<double*>(derivs), dim, dim, false);
    arma::mat jac(wk_mem, dim, dim, false);
    jac = D * L;

    double * __restrict__ r = res;
    for(unsigned j = 0; j < dim; ++j){ 
      *r++ = 2 * L.at(j, j) * jac.at(j, j);
      for(unsigned i = j + 1; i < dim; ++i)
        *r++ = 2 * jac.at(i, j);
    }
  }
};

inline double vec_dot(arma::vec const &x, double const *y) noexcept {
  return std::inner_product(x.begin(), x.end(), y, 0.);
}

inline double quad_form(arma::mat const &X, double const *x) noexcept {
  arma::uword const n = X.n_rows;
  double out(0.);
  for(arma::uword j = 0; j < n; ++j){
    double const *Xij = X.colptr(j) + j, 
                 *xi  = x + j + 1;
    out += *Xij++ * x[j] * x[j];
    for(arma::uword i = j + 1; i < n; ++i)
      out += 2 * *Xij++ * x[j] * *xi++;
  }
  
  return out;
}

/** 
 * compute the log of the determinant of X = LL^T given L where L is a lower 
 * triangular matrix.
 */
inline double log_deter(arma::mat const &L){
  double out{};
  unsigned const n = L.n_cols;
  double const *l = L.begin();
  for(unsigned i = 0; i < n; ++i, l += n + 1)
    out += log(*l);
  return 2 * out;
}

extern "C" {
  /// either performs a forward or backward solve
  void F77_NAME(dtpsv)
    (const char * /* uplo */, const char * /* trans */, const char * /* diag */,
     const int * /* n */, const double * /* ap */, double * /* x */, 
     const int * /* incx */ FCLEN FCLEN FCLEN);

  /// computes the inverse from the Choleksy factorization
  void F77_NAME(dpptri)
    (const char * /* uplo */, const int * /* n */, double *ap, 
     int * /* info */ FCLEN);
}

// copies a lower triangular matrix of a n x n matrix into a n(n + 1) / 2 vector
inline void copy_lower_tri(arma::mat const &X, double * __restrict__ res){
  arma::uword const n = X.n_cols;
  for(arma::uword j = 0; j < n; ++j)
    for(arma::uword i = j; i < n; ++i)
      *res++ = X.at(i, j);
}

/** 
 * given a n(n + 1) / 2 vector containing a lower triangular matrix X
 * and a n x k matrix A, this function solves XY = A or X^TY = A
 */
inline void tri_solve(double const *x, double * A, int const n, 
                      int const k, bool const transpose){
  char const uplo{'L'}, 
             trans = transpose ? 'T' : 'N', 
             diag{'N'};
  int const incx{1};
  
  for(int i = 0; i < k; ++i, A += n)
    F77_CALL(dtpsv)
    (&uplo, &trans, &diag, &n, x, A, &incx FCONE FCONE FCONE);
}

/// same as tri_solve but finds XX^TY = A 
inline void tri_solve_original
  (double const *x, double *A, int const n, int const k){
  tri_solve(x, A, n, k, false);
  tri_solve(x, A, n, k, true);
}  
  
// needs to be forward declared due for lower_bound_caller 
class lower_bound_term;

/**
 * Class used to compute the unconditional random effect covariance matrix
 * it Cholesky decomposition, and its inverse once. 
 */  
struct lower_bound_caller {
  /**
   * the first element is the number of fixed effects and the second
   * element is the number of random effects.
   */
  std::array<unsigned, 2> dims;
  arma::mat Sig, Sig_L, Sig_inv;
  double Sig_log_deter;
  // stores Sig_L in compact form 
  std::unique_ptr<double> Sig_L_compact;
  
  lower_bound_caller(std::vector<lower_bound_term const*>&);
  void setup(double const *val, bool const comp_grad);
  double eval_func(lower_bound_term const &obj, double const * val);
  double eval_grad(lower_bound_term const &obj, double const * val, 
                   double *gr);
};

class lower_bound_term {
public:
  enum models { binomial_logit, Poisson_log };
  // the model to use
  models const model;
  // outcomes and size variables for the binomial distribution
  arma::vec const y, nis;
  // design matrices
  arma::mat const X, Z;
  
  unsigned const n_beta = X.n_rows, 
                  n_rng = Z.n_rows,
                  n_sig = (n_rng * (n_rng + 1L)) / 2L,
                  n_obs = y.n_elem;
  
  // normalization constant
  double const norm_constant;
  
private:
  // members and functions handle working memory
  static unsigned mem_per_thread,
                  n_mem_alloc;
  static std::unique_ptr<double[]> wk_mem;
  static double * get_thread_mem() noexcept {
#ifdef _OPENMP
    int const thread_num(omp_get_thread_num());
#else
    int const thread_num(0L));
#endif
    
    return wk_mem.get() + thread_num * mem_per_thread;
  }
  
public:
  lower_bound_term(std::string const &smodel, arma::vec const &y, 
                   arma::vec const &nis, arma::mat const &X, 
                   arma::mat const &Z): 
  model(([&]() -> models {
    if(smodel == "binomial_logit")
      return models::binomial_logit;
    else if(smodel == "Poisson_log")
      return models::Poisson_log;
    
    throw std::invalid_argument("model not implemented");
    return models::binomial_logit;
  })()),
  y(y), nis(nis), X(X), Z(Z), 
  norm_constant(([&]() -> double {
    if(model == models::binomial_logit){
      if(static_cast<size_t>(n_obs) != nis.n_elem)
        // have to check this now
        throw std::invalid_argument("invalid nis");
      
      double out(n_rng / 2.);
      for(unsigned i = 0; i < n_obs; ++i)
        out += std::lgamma(nis[i] + 1) - std::lgamma(y[i] + 1) - 
          std::lgamma(nis[i] - y[i] + 1);
      return -out;
      
    } else if(model == models::Poisson_log){
      double out(n_rng / 2.);
      for(unsigned i = 0; i < n_obs; ++i)
        out -= std::lgamma(y[i] + 1);
      return -out;
      
    } else
      throw std::runtime_error("normalization constant not implemented");
  })()) {
    // checks
    if(X.n_cols != static_cast<size_t>(n_obs))
      throw std::invalid_argument("invalid X");
    if(Z.n_cols != static_cast<size_t>(n_obs))
      throw std::invalid_argument("invalid X");
  }
  
  lower_bound_term(Rcpp::List dat):
  lower_bound_term(Rcpp::as<std::string>(dat["model"]),
                   Rcpp::as<arma::vec>  (dat["y"]),
                   Rcpp::as<arma::vec>  (dat["nis"]),
                   Rcpp::as<arma::mat>  (dat["X"]),
                   Rcpp::as<arma::mat>  (dat["Z"])) { }
  
  /// sets the working memory.
  static void set_wk_mem(unsigned const max_n_beta, unsigned const max_n_rng, 
                         unsigned const max_n_obs, unsigned const max_threads){
    constexpr size_t const mult = cacheline_size() / sizeof(double),
                       min_size = 2L * mult;
    
    size_t n_ele = std::max<size_t>
      (8L * max_n_rng * max_n_rng + max_n_rng, min_size);
    n_ele = (n_ele + mult - 1L) / mult;
    n_ele *= mult;
    n_ele += dpd_mat::n_wmem(max_n_rng);
    
    mem_per_thread = n_ele;
    n_ele *= max_threads;
    
    if(n_mem_alloc < n_ele){
      n_mem_alloc = n_ele;
      wk_mem.reset(new double[n_ele]);
    }
  }
  
  // the rest is the member functions which are needed for the psqn package.
  size_t global_dim() const {
    return n_beta + n_sig;
  }
  size_t private_dim() const {
    return n_rng + n_sig;
  }
  
  double comp(double const *p, double *gr, 
              lower_bound_caller const &caller, bool const comp_grad) const {
    // setup the objects we need
    unsigned const beta_start = 0, 
                    Sig_start = beta_start + n_beta, 
                  va_mu_start = Sig_start + n_sig, 
                    Lam_start = va_mu_start + n_rng;
    arma::vec const beta = vec_no_cp(p + beta_start , n_beta), 
                   va_mu = vec_no_cp(p + va_mu_start, n_rng); 
    
    // the working memory and function to get working memory.
    double * w = get_thread_mem();
    auto get_wk_mem = [&](unsigned const n_ele){
      double * out = w;
      w += n_ele;
      return out;
    };
    
    unsigned const n_rng_sq = n_rng * n_rng;
    arma::mat Lam_L(get_wk_mem(n_rng_sq), n_rng, n_rng, false), 
              Lam  (get_wk_mem(n_rng_sq), n_rng, n_rng, false);
    get_pd_mat(p + Lam_start, Lam_L, Lam);
    
    arma::mat const &Sig_L = caller.Sig_L, 
                  &Sig_inv = caller.Sig_inv;
    double const * const Sig_L_compact{caller.Sig_L_compact.get()};
    
    // objects for partial derivatives
    arma::vec dbeta(gr + beta_start , comp_grad ? n_beta : 0, false), 
             dva_mu(gr + va_mu_start, comp_grad ? n_rng  : 0, false);
    arma::mat dSig(get_wk_mem(n_rng_sq), n_rng, n_rng, false), 
              dLam(get_wk_mem(n_rng_sq), n_rng, n_rng, false);
    if(comp_grad){
      dbeta.zeros();
      dva_mu.zeros();
      
      dSig.zeros(); 
      dLam.zeros(); 
    }
    
    // evaluate the lower bound. Start with the terms from the conditional 
    // density of the outcomes
    double out(norm_constant);
    for(unsigned i = 0; i < n_obs; ++i){
      double const eta = 
        vec_dot(beta, X.colptr(i)) + vec_dot(va_mu, Z.colptr(i)), 
               cond_sd = std::sqrt(std::abs(
                quad_form(Lam, Z.colptr(i))));
      
      if(!comp_grad){
        double B(0);
        if(model == models::binomial_logit)
          B = partition::logit::B(eta, cond_sd, true);
        else if(model == models::Poisson_log)
          B = partition::poisson::B(eta, cond_sd);
        else
          throw std::runtime_error("partition function not implemented");
        out += -y[i] * eta + nis[i] * B;
        
      } else {
        std::array<double, 3L> const Bp = ([&]() -> std::array<double, 3L> {
          if(model == models::binomial_logit)
            return partition::logit::Bp(eta, cond_sd, true);
          else if (model != models::Poisson_log)
            throw std::runtime_error("partition function not implemented");
          
          return   partition::poisson::Bp(eta, cond_sd);
        })();
        
        // the lower bound terms
        out += -y[i] * eta + nis[i] * Bp[2];
        
        // the derivatives
        double const d_eta = -y[i] + nis[i] * Bp[0];
        dbeta  += d_eta * X.col(i);
        dva_mu += d_eta * Z.col(i);
        
        double const mult = nis[i] * Bp[1] / cond_sd * .5;
        for(unsigned k = 0; k < n_rng; ++k){
          for(unsigned j = 0; j < k; ++j){
            double const term = mult * Z.at(k, i) * Z.at(j, i); 
            dLam.at(j, k) += term;
            dLam.at(k, j) += term;
          }
          dLam.at(k, k) += mult * Z.at(k, i) * Z.at(k, i); 
        }
      }
    }
    
    // terms from the KL divergence of the unconditional random effect 
    // density and the variational distribution density
    double half_term{};
    
    // determinant terms
    half_term -= caller.Sig_log_deter;
    half_term += log_deter(Lam_L);
    
    arma::vec sig_inv_va_mu(get_wk_mem(n_rng), n_rng, false);
    std::copy(va_mu.begin(), va_mu.end(), sig_inv_va_mu.begin());
    tri_solve(Sig_L_compact, sig_inv_va_mu.begin(), n_rng, 1, false);
    half_term -= vec_dot(sig_inv_va_mu, sig_inv_va_mu.begin());
    tri_solve(Sig_L_compact, sig_inv_va_mu.begin(), n_rng, 1, true);
    
    arma::mat Sig_inv_Lam(get_wk_mem(n_rng_sq), n_rng, n_rng, false);
    std::copy(Lam.begin(), Lam.end(), Sig_inv_Lam.begin());
    tri_solve_original(Sig_L_compact, Sig_inv_Lam.begin(), n_rng, n_rng);
    
    for(unsigned i = 0; i < n_rng; ++i){
      for(unsigned j = 0; j < i; ++j){
        if(comp_grad){
          double const d_term = .5 * Sig_inv.at(j, i);
          dLam.at(j, i) += d_term;
          dLam.at(i, j) += d_term;
          dSig.at(j, i) += d_term;
          dSig.at(i, j) += d_term;
        }
      }
      half_term -= Sig_inv_Lam.at(i, i);
      if(comp_grad){
        dLam.at(i, i) += .5 * Sig_inv.at(i, i);
        dSig.at(i, i) += .5 * Sig_inv.at(i, i);
      }
    }
    
    out -= .5 * half_term;
    if(comp_grad){
      dva_mu += sig_inv_va_mu;
      
      {
        // TODO: we can compute this from the Cholesky decomposition
        arma::mat lam_inv(get_wk_mem(n_rng_sq), n_rng, n_rng, false); 
        if(!arma::inv_sympd(lam_inv, Lam))
          half_term = std::numeric_limits<double>::quiet_NaN();
        else
          dLam -= .5 * lam_inv; 
      }
      
      // modify Lam as we do not need it anymore
      double * const d_pd_mem = // last working memory we can use
        get_wk_mem(2 * n_rng_sq);
      for(unsigned i = 0; i < n_rng; ++i)
        for(unsigned j = 0; j < n_rng; ++j)
          Lam.at(j, i) += va_mu[i] * va_mu[j];
      
      tri_solve_original(Sig_L_compact, Lam.begin(), n_rng, n_rng);
      arma::mat tmp(d_pd_mem, n_rng, n_rng, false);
      tmp = Lam.t();
      tri_solve_original(Sig_L_compact, tmp.begin(), n_rng, n_rng);
      dSig -= .5 * tmp; 
      
      // copy the result
      dpd_mat::get(Sig_L, gr + Sig_start, dSig.begin(), d_pd_mem);
      dpd_mat::get(Lam_L, gr + Lam_start, dLam.begin(), d_pd_mem);
    }
    
    return out;
  }
  
  double func(double const *point, lower_bound_caller const &caller) const {
    return comp(point, nullptr, caller, false);
  }
  
  double grad
  (double const * point, double * gr, 
   lower_bound_caller const &caller) const {
    return comp(point, gr, caller, true);
  }
  
  bool thread_safe() const {
    return true;
  }
};

unsigned lower_bound_term::mem_per_thread = 0L, 
         lower_bound_term::n_mem_alloc    = 0L;
std::unique_ptr<double[]> lower_bound_term::wk_mem = 
  std::unique_ptr<double[]>();

// definitions of lower_bound_caller's member functions
lower_bound_caller::lower_bound_caller
  (std::vector<lower_bound_term const*>& funcs):
  dims(([&](){
    // get the dimension of the random effects
    if(funcs.size() < 1 or !funcs[0])
      throw std::invalid_argument(
          "lower_bound_caller::lower_bound_caller: invalid funcs");
    unsigned const n_rng = funcs[0]->n_rng, 
                  n_beta = funcs[0]->n_beta;
             
    // checks 
    for(auto &f: funcs)
      // check that n_rng is identical for each func
      if(!f or f->n_rng != n_rng or f->n_beta != n_beta)
        throw std::invalid_argument(
            "lower_bound_caller::lower_bound_caller: n_rng or n_beta do not match");
    
    
    return std::array<unsigned, 2L>({ n_beta, n_rng });
  })()), 
  Sig(dims[1], dims[1]), Sig_L(dims[1], dims[1]), 
  Sig_inv(dims[1], dims[1]), 
  Sig_log_deter(std::numeric_limits<double>::quiet_NaN()), 
  Sig_L_compact(new double[(dims[1] * (dims[1] + 1)) / 2]){ }

void lower_bound_caller::setup(double const *val, bool const comp_grad){
  // compute Sigma and setup the Cholesky decomposition 
  get_pd_mat(val + dims[0], Sig_L, Sig);
  
  copy_lower_tri(Sig_L, Sig_L_compact.get());
  
  if(!arma::inv_sympd(Sig_inv, Sig)){
    // inversion failed
    Sig_inv.zeros(dims[1], dims[1]);
    Sig_log_deter = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  
  // compute the determinant
  Sig_log_deter = log_deter(Sig_L);
}

double lower_bound_caller::eval_func
  (lower_bound_term const &obj, double const * val){
  return obj.func(val, *this);
}

double lower_bound_caller::eval_grad(
    lower_bound_term const &obj, double const * val, double *gr){
  return obj.grad(val, gr, *this);
}

// psqn interface 
using lb_optim = PSQN::optimizer
  <lower_bound_term, PSQN::R_reporter, PSQN::R_interrupter, 
   lower_bound_caller>;

// [[Rcpp::export(rng = false)]]
SEXP get_lb_optimizer(Rcpp::List data, unsigned const max_threads){
  size_t const n_elem_funcs = data.size();
  std::vector<lower_bound_term> funcs;
  funcs.reserve(n_elem_funcs);
  
  unsigned max_n_beta(0L), 
           max_n_rng (0L), 
           max_n_obs(0L);
  for(auto dat : data){
    funcs.emplace_back(Rcpp::List(dat));
    lower_bound_term const &obj = funcs.back();
    max_n_beta = std::max(max_n_beta, obj.n_beta);
    max_n_rng  = std::max(max_n_rng , obj.n_rng);
    max_n_obs  = std::max(max_n_obs , obj.n_obs);
  }
  lower_bound_term::set_wk_mem(
    max_n_beta, max_n_rng, max_n_obs, max_threads);
  
  // create an XPtr to the object we will need
  Rcpp::XPtr<lb_optim> ptr(new lb_optim(funcs, max_threads));
  
  // return the pointer to be used later
  return ptr;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List opt_lb
  (Rcpp::NumericVector val, SEXP ptr, double const rel_eps, unsigned const max_it,
   unsigned const n_threads, double const c1,
   double const c2, bool const use_bfgs = true, unsigned const trace = 0L,
   double const cg_tol = .5, bool const strong_wolfe = true,
   size_t const max_cg = 0L, unsigned const pre_method = 1L){
  Rcpp::XPtr<lb_optim> optim(ptr);
  
  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("optim_lb: invalid parameter size");
  
  Rcpp::NumericVector par = clone(val);
  optim->set_n_threads(n_threads);
#ifdef DO_PROF
  ProfilerStart("optim");
#endif
  auto res = optim->optim(&par[0], rel_eps, max_it, c1, c2,
                          use_bfgs, trace, cg_tol, strong_wolfe, max_cg,
                          static_cast<PSQN::precondition>(pre_method));
#ifdef DO_PROF
  ProfilerStop();
#endif
  Rcpp::NumericVector counts = Rcpp::NumericVector::create(
    res.n_eval, res.n_grad,  res.n_cg);
  counts.names() = 
    Rcpp::CharacterVector::create("function", "gradient", "n_cg");
  
  int const info = static_cast<int>(res.info);
  return Rcpp::List::create(
    Rcpp::_["par"] = par, Rcpp::_["value"] = res.value, 
    Rcpp::_["info"] = info, Rcpp::_["counts"] = counts,
    Rcpp::_["convergence"] =  res.info == PSQN::info_code::converged);
}

// [[Rcpp::export(rng = false)]]
double eval_lb(Rcpp::NumericVector val, SEXP ptr, unsigned const n_threads){
  Rcpp::XPtr<lb_optim> optim(ptr);
  
  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("eval_lb: invalid parameter size");
  
  optim->set_n_threads(n_threads);
  return optim->eval(&val[0], nullptr, false);
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector eval_lb_gr(Rcpp::NumericVector val, SEXP ptr,
                               unsigned const n_threads){
  Rcpp::XPtr<lb_optim> optim(ptr);
  
  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("eval_lb_gr: invalid parameter size");
  
  Rcpp::NumericVector grad(val.size());
  optim->set_n_threads(n_threads);
  grad.attr("value") = optim->eval(&val[0], &grad[0], true);
  
  return grad;
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector opt_priv
  (Rcpp::NumericVector val, SEXP ptr, 
   double const rel_eps, unsigned const max_it, unsigned const n_threads, 
   double const c1, double const c2){
  Rcpp::XPtr<lb_optim> optim(ptr);

  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("opt_priv: invalid parameter size");
  
  Rcpp::NumericVector par = clone(val);
  optim->set_n_threads(n_threads);
  double const res = optim->optim_priv(&par[0], rel_eps, max_it, c1, c2);
  par.attr("value") = res;
  return par;
}

/// function used to get starting values
// [[Rcpp::export(rng = false)]]
arma::vec get_start_vals(SEXP ptr, arma::vec const &par, 
                         arma::mat const &Sigma, unsigned const n_beta){
  Rcpp::XPtr<lb_optim> optim(ptr);
  std::vector<lower_bound_term const *> funcs = optim->get_ele_funcs();
  
  arma::vec out = par;
  unsigned const n_rng = Sigma.n_cols,
              n_groups = funcs.size(), 
                n_vcov = (n_rng * (1 + n_rng)) / 2;
  
  if(par.size() != static_cast<size_t>(
    n_beta + n_vcov + n_groups * (n_rng + n_vcov)) or 
       par.size() != optim->n_par)
    throw std::invalid_argument("get_start_vals: invalid par");
  
  arma::vec const beta(out.begin(), n_beta);
  double * va_start = out.begin() + n_beta + n_vcov;
  arma::mat const Sigma_inv = arma::inv_sympd(Sigma);
  arma::mat Lam_inv(n_rng, n_rng), 
  Lam    (n_rng, n_rng), 
  Lam_chol(n_rng, n_rng);
  
  for(auto &t : funcs){
    lower_bound_term const &term = *t;
    if(term.n_beta != n_beta or term.n_rng != n_rng)
      throw std::invalid_argument("get_start_vals: invalid data element");
    
    // make Taylor expansion around zero vector
    Lam_inv = Sigma_inv;
    for(unsigned i = 0; i < term.n_obs; ++i){
      double const eta = arma::dot(beta, term.X.col(i));
      double dd_eta(0.);
      if     (term.model == lower_bound_term::models::binomial_logit)
        dd_eta = partition::logit::dd_eta(eta);
      else if(term.model == lower_bound_term::models::Poisson_log )
        dd_eta = partition::poisson::dd_eta(eta);
      else 
        throw std::runtime_error("get_start_vals: model not implemented");
      
      for(unsigned j = 0; j < n_rng; ++j)
        for(unsigned k = 0; k < n_rng; ++k)
          Lam_inv.at(k, j) += term.Z.at(j, i) * term.Z.at(k, i) * dd_eta;
    }
    
    if(!arma::inv_sympd(Lam, Lam_inv))
      throw std::runtime_error("get_start_vals: inv_sympd failed");
    if(!arma::chol(Lam_chol, Lam, "lower"))
      throw std::runtime_error("get_start_vals: Lam_chol failed");
    
    double * theta = va_start + n_rng;
    for(unsigned j = 0; j < n_rng; ++j){
      *theta++ = std::log(Lam_chol.at(j, j));
      for(unsigned i = j + 1; i < n_rng; ++i)
        *theta++ = Lam_chol.at(i, j);
    }
    
    va_start += n_vcov + n_rng;
  }
  
  return out;
}

/// function to test the expected log partition function in R
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector logit_partition
  (double const mu, double const sigma, unsigned const order, 
   bool const adaptive = true){
  if       (order == 0L){
    Rcpp::NumericVector out(1);
    double const val{partition::logit::B(mu, sigma, adaptive)};
    return Rcpp::NumericVector{val};
  } else if(order == 1L){
    auto res = partition::logit::Bp(mu, sigma, adaptive);
    Rcpp::NumericVector out{res[0], res[1]};
    out.attr("value") = res[2];
    return out;
  }
  
  return Rcpp::NumericVector();
}

// functions to check get_pd_mat and d_get_pd_mat
// [[Rcpp::export(rng = false)]]
Rcpp::List get_pd_mat(Rcpp::NumericVector theta, unsigned const dim){
  arma::mat L(dim, dim), 
          res(dim, dim);
  get_pd_mat(&theta[0], L, res);
  return Rcpp::List::create(Rcpp::Named("X") = res, 
                            Rcpp::Named("L") = L);
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector d_get_pd_mat(arma::vec const &gr, arma::mat const &L){
  unsigned const n = L.n_cols;
  Rcpp::NumericVector out((n * (n + 1)) / 2, 0);
  std::unique_ptr<double[]> wk_mem(new double[dpd_mat::n_wmem(n)]);
  dpd_mat::get(L, &out[0], gr.begin(), wk_mem.get());
  return out;
}

/***R
# setup
n <- 5
set.seed(1L)
X <- drop(rWishart(1, n, diag(n)))
L <- t(chol(X))
diag(L) <- log(diag(L))
theta <- L[lower.tri(L, TRUE)]

# checks
all.equal(X, get_pd_mat(theta, n)[[1L]])

library(numDeriv)
gr_truth <- grad(
  function(x) sum(sin(get_pd_mat(x, n)[[1L]])), theta)
gr <- cos(c(X))
L <- get_pd_mat(theta, n)[[2L]]
all.equal(gr_truth, d_get_pd_mat(gr, L))

# should not be a bottleneck?
bench::mark(
    get_pd_mat = get_pd_mat(theta, n),
  d_get_pd_mat = d_get_pd_mat(cos(c(X)), L),
  check = FALSE, min_time = .5, max_iterations = 1e6)
*/

// test the expected log partition functions
/*** R
# switch the default
formals(logit_partition)$adaptive <- TRUE

# integrand in the logit partition function
f <- function(x, mu, sigma)
  dnorm(x) * ifelse(x > 30, x, log(1 + exp(sigma * x + mu)))

# check the relative error
mus <- seq(-4, 4, length.out = 100)
sigmas <- seq(.1, 3, length.out = 100)
grid <- expand.grid(mu = mus, sigma = sigmas)

rel_err <- mapply(function(mu, sigma){
  truth <- integrate(f, -Inf, Inf, mu = mu, sigma = sigma, rel.tol = 1e-13)
  est <- logit_partition(mu = mu, sigma = sigma, order = 0)
  (truth$value - est) / truth$value 
}, mu = grid$mu, sigma = grid$sigma)

# plot the errors
range(rel_err) # range of the relative errors
log10(max(abs(rel_err))) # digits of precision
contour(mus, sigmas, matrix(rel_err, length(mus)), 
        xlab = expression(mu), ylab = expression(sigma), 
        main = "Relative error of E(logit partition)")

# check the relative error of the derivative. First numerically 
mu <- 1
sigma <- 2
library(numDeriv)
grad(function(x) 
  integrate(f, -Inf, Inf, mu = x[1], sigma = x[2], rel.tol = 1e-12)$value, 
  c(mu, sigma))
logit_partition(mu = mu, sigma = sigma, order = 1)

# w/ derivative of the integrand
rel_err <- mapply(function(mu, sigma){
  t1 <- integrate(
    function(x) dnorm(x) / (1 + exp(-(sigma * x + mu))), 
    -Inf, Inf, rel.tol = 1e-12)$value
  t2 <- integrate(
    function(x) x * dnorm(x) / (1 + exp(-(sigma * x + mu))), 
    -Inf, Inf, rel.tol = 1e-12)$value
  
  truth <- c(t1, t2)
  est <- logit_partition(mu = mu, sigma = sigma, order = 1)
  
  (truth - est) / truth
}, mu = grid$mu, sigma = grid$sigma)

# plot the errors
range(rel_err[1, ]) # range of the relative errors (dmu)
log10(max(abs(rel_err[1, ]))) # digits of precision
contour(mus, sigmas, matrix(rel_err[1,], length(mus)), 
        xlab = expression(mu), ylab = expression(sigma), 
        main = "Relative error of dE(logit partition) / dmu")

range(rel_err[2, ]) # range of the relative errors (dsigma)
log10(max(abs(rel_err[2, ])))
contour(mus, sigmas, matrix(rel_err[2,], length(mus)), 
        xlab = expression(mu), ylab = expression(sigma), 
        main = "Relative error of dE(logit partition) / dsigma")

# check the computation time
bench::mark( f = logit_partition(mu = 1, sigma = 1.33, order = 0),
            df = logit_partition(mu = 1, sigma = 1.33, order = 1),
            min_time = .5, max_iterations = 1e6, check = FALSE)

# also work with extreme inputs
for(mu in c(-40, -20, -10, 10, 20, 40))
  for(sigma in c(100, 400, 800)){
    f  <- logit_partition(mu = mu, sigma = sigma, order = 0)
    dp <- logit_partition(mu = mu, sigma = sigma, order = 1)
    cat(sprintf("mu = %4d, sigma = %4d: %7.2f %7.2f %7.2f\n", 
                mu, sigma, f, dp[1], dp[2]))
  }

# plot partial derivatives for extreme sigmas
sds <- seq(1, 1000, length.out = 1000)
matplot(sds, t(sapply(sds, logit_partition, mu = 20, order = 1)), 
        type = "l", lty = 1, xlab = expression(sigma), 
        ylab = "Partial derivatives")
*/

/***R
# simple function to simulate from a mixed probit model with a random 
# intercept and a random slope. 
# 
# Args: 
#   sig: scale parameter for the correlation matrix
#   inter: the intercept
#   n_cluster: number of clusters
#   slope: slope of the covariate
sim_dat <- function(sig, inter, n_cluster = 100L, 
                    slope = 0){
  cor_mat <- matrix(c(1, -.25, -.25, 1), 2L) # the correlation matrix
  vcov_mat <- sig * cor_mat   # the covariance matrix
  beta <- c(inter, slope) # the fixed effects
  n_obs <- 10L # number of members in each cluster
  
  # simulate the clusters
  group <- 0L
  out <- replicate(n_cluster, {
    # the random effect 
    u <- drop(rnorm(NCOL(vcov_mat)) %*% chol(vcov_mat))
    
    # design matrix
    x <- runif(n_obs, -sqrt(12) / 2, sqrt(12) / 2)
    
    # linear predcitor
    eta <- drop(cbind(1, x) %*% c(u + beta))
    
    # the outcome 
    prob <- 1/(1 + exp(-eta))
    nis <- sample.int(5L, n_obs, replace = TRUE)
    y <- rbinom(n_obs, nis, prob)
    nis <- as.numeric(nis)
    y <- as.numeric(y)
    
    # return 
    Z <- rbind(1, x)
    group <<- group + 1L
    list(x = x, y = y, group = rep(group, n_obs), 
         nis = nis, X = Z, Z = Z, model = "binomial_logit")
  }, simplify = FALSE)
  
  # create a data.frame with the data set and return 
  . <- function(what)
    c(sapply(out, `[[`, what))
  list(
    sim_dat = data.frame(y     = .("y"), 
                         x     = .("x"), 
                         group = .("group"), 
                         nis   = .("nis")), 
    vcov_mat = vcov_mat, beta = beta, 
    list_dat = out)   
}

# simulate a small data set
n_clust <- 2L
n_rng <- 2L
n_fix <- 2L
small_dat <- sim_dat(sig = .5, inter = 0, n_cluster = n_clust,
                     slope = 0)

func <- get_lb_optimizer(small_dat$list_dat, 1L)
fn <- function(x)
  eval_lb   (x, ptr = func, n_threads = 1L)
gr <- function(x)
  eval_lb_gr(x, ptr = func, n_threads = 1L)

# check the gradient
set.seed(1)
point <- runif(
  n_fix + n_clust * n_rng + (n_clust + 1) * n_rng * (n_rng + 1L) / 2,
  -1, 1)

fn(point)
library(numDeriv)
num_aprx <- grad(fn, point, method.args = list(r = 10))
gr_cpp <- gr(point)
cbind(num_aprx, gr_cpp, diff = gr_cpp - num_aprx)
stopifnot(isTRUE(
  all.equal(num_aprx, gr_cpp, check.attributes = FALSE)))

# compare w/ Laplace approximation. First assign functions to estimate the 
# model
library(lme4)
est_Laplace <- function(dat){
  fit <- glmer(cbind(y, nis - y) ~ x + (1 + x | group), dat$sim_dat,
               family = binomial())
  vc <- VarCorr(fit)
  list(ll = c(logLik(fit)), fixef = fixef(fit), 
       stdev = attr(vc$group, "stddev"), 
       cormat = attr(vc$group, "correlation"))
}

est_va <- function(dat, rel_eps = 1e-8){
  func <- get_lb_optimizer(dat$list_dat, 1L)
  fn <- function(x)
    eval_lb   (x, ptr = func, n_threads = 1L)
  gr <- function(x)
    eval_lb_gr(x, ptr = func, n_threads = 1L)
  
  # setup stating values
  n_clust <- length(dat$list_dat)
  par <- numeric(n_fix + n_clust * n_rng +
                   (n_clust + 1) * n_rng * (n_rng + 1L) / 2)
  
  # estimate the fixed effects w/ a GLM
  if(n_fix > 0)
    par[1:n_fix] <- with(dat$sim_dat, glm.fit(
      x = cbind(1, x), y = y / nis, weights = nis, family = binomial()))[[
        "coefficients"]]

  par <- drop(get_start_vals(func, par = par, Sigma = diag(n_rng), 
                             n_beta = n_fix))
  
  res <- opt_lb(val = par, ptr = func, rel_eps = rel_eps, max_it = 1000L, 
                n_threads = 1L, c1 = 1e-4, c2 = .9, cg_tol = .2, 
                max_cg = max(2L, as.integer(log(n_clust) * 10)))
  
  mod_par <- head(res$par, n_fix + n_rng * (n_rng + 1) / 2)
  Sig_hat <- get_pd_mat(tail(mod_par, -n_fix), n_rng)[[1L]]
  
  list(lb = -res$value, fixef = head(mod_par, n_fix), 
       stdev = sqrt(diag(Sig_hat)), cormat = cov2cor(Sig_hat))
}

# then simulate and use the functions
set.seed(1)
n_clust <- 1000L
dat <- sim_dat(sig = .6^2, inter = 1, n_cluster = n_clust, slope = -1)
system.time(print(est_va     (dat)))
system.time(print(est_Laplace(dat)))

# truth is
list(fixef  = dat$beta, stdev = diag(sqrt(dat$vcov_mat)), 
     cormat = cov2cor(dat$vcov_mat))
*/
