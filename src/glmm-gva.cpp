// [[Rcpp::plugins(openmp, cpp11)]]
// [[Rcpp::depends(RcppArmadillo, psqn)]]
#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include "psqn.h"
#include "psqn-reporter.h"
#include <cmath>
#include <array>
#include <limits>
#include <memory.h>
#include <algorithm>

#ifdef DO_PROF
#include <gperftools/profiler.h>
#endif

/*
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
 *  Copyright (C) 2003-2004  The R Foundation
 *  Copyright (C) 1998--2014  The R Core Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 */

/* fmin.f -- translated by f2c (version 19990503). */

/*  R's  optimize() :   function	fmin(ax,bx,f,tol)
    =    ==========		            ~~~~~~~~~~~~~~~~~
    an approximation  x  to the point where  f  attains a minimum  on
    the interval  (ax,bx)  is determined.
    INPUT..
    ax    left endpoint of initial interval
    bx    right endpoint of initial interval
    f     function which evaluates  f(x, info)  for any  x
    in the interval  (ax,bx)
    tol   desired length of the interval of uncertainty of the final
    result ( >= 0.)
    OUTPUT..
    fmin  abcissa approximating the point where  f  attains a minimum
    The method used is a combination of  golden  section  search  and
    successive parabolic interpolation.  convergence is never much slower
    than  that  for  a  Fibonacci search.  If  f  has a continuous second
    derivative which is positive at the minimum (which is not  at  ax  or
    bx),  then  convergence  is  superlinear, and usually of the order of
    about  1.324....
    The function  f  is never evaluated at two points closer together
    than  eps*abs(fmin)+(tol/3), where eps is  approximately  the  square
    root  of  the  relative  machine  precision.   if   f   is a unimodal
    function and the computed values of   f   are  always  unimodal  when
    separated  by  at least  eps*abs(x)+(tol/3), then  fmin  approximates
    the abcissa of the global minimum of  f  on the interval  ax,bx  with
    an error less than  3*eps*abs(fmin)+tol.  if   f   is  not  unimodal,
    then fmin may approximate a local, but perhaps non-global, minimum to
    the same accuracy.
    This function subprogram is a slightly modified  version  of  the
    Algol  60 procedure  localmin  given in Richard Brent, Algorithms for
    Minimization without Derivatives, Prentice-Hall, Inc. (1973).
*/

template<class OptBoj>
double 
Brent_fmin(double ax, double bx, OptBoj &obj, double tol) noexcept
{
  /*  c is the squared inverse of the golden ratio */
  const double c = (3. - sqrt(5.)) * .5;

  /* Local variables */
  double a, b, d, e, p, q, r, u, v, w, x;
  double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

  /*  eps is approximately the square root of the relative machine precision. */
  eps = std::numeric_limits<double>::epsilon();
  tol1 = eps + 1.;/* the smallest 1.000... > 1 */
  eps = sqrt(eps);

  a = ax;
  b = bx;
  v = a + c * (b - a);
  w = v;
  x = v;

  d = 0.;/* -Wall */
  e = 0.;
  fx = obj.optimfunc(x);
  fv = fx;
  fw = fx;
  tol3 = tol / 3.;

  /*  main loop starts here ----------------------------------- */

  for(;;) {
    xm = (a + b) * .5;
    tol1 = eps * fabs(x) + tol3;
    t2 = tol1 * 2.;

    /* check stopping criterion */

    if (fabs(x - xm) <= t2 - (b - a) * .5) break;
    p = 0.;
    q = 0.;
    r = 0.;
    if (fabs(e) > tol1) { /* fit parabola */

    r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.;
      if (q > 0.) p = -p; else q = -q;
      r = e;
      e = d;
    }

    if (fabs(p) >= fabs(q * .5 * r) ||
        p <= q * (a - x) || p >= q * (b - x)) { /* a golden-section step */

    if (x < xm) e = b - x; else e = a - x;
    d = c * e;
    }
    else { /* a parabolic-interpolation step */

    d = p / q;
      u = x + d;

      /* f must not be evaluated too close to ax or bx */

      if (u - a < t2 || b - u < t2) {
        d = tol1;
        if (x >= xm) d = -d;
      }
    }

    /* f must not be evaluated too close to x */

    if (fabs(d) >= tol1)
      u = x + d;
    else if (d > 0.)
      u = x + tol1;
    else
      u = x - tol1;

    fu = obj.optimfunc(u);

    /*  update  a, b, v, w, and x */

    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w;    w = x;   x = u;
      fv = fw; fw = fx; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  /* end of main loop */

  return x;
} // Brent_fmin()

namespace ghq {
/// Gauss–Hermite quadrature weights
constexpr int const n_ghq_many = 40L, 
                    n_ghq_few  = 20L;
/// the GHQ weights 
constexpr double const ws_many[n_ghq_many] = { 2.59104371384724e-29, 8.54405696377501e-25, 2.56759336541157e-21, 1.98918101211647e-18, 6.00835878949072e-16, 8.8057076452162e-14, 7.15652805269044e-12, 3.52562079136532e-10, 1.12123608322759e-08, 2.41114416367052e-07, 3.63157615069303e-06, 3.9369339810925e-05, 0.00031385359454133, 0.00187149682959795, 0.00846088800825812, 0.0293125655361725, 0.0784746058654042, 0.163378732713271, 0.265728251877377, 0.338643277425589, 0.33864327742559, 0.265728251877377, 0.163378732713271, 0.0784746058654045, 0.0293125655361724, 0.0084608880082581, 0.00187149682959798, 0.00031385359454133, 3.93693398109249e-05, 3.631576150693e-06, 2.41114416367055e-07, 1.12123608322757e-08, 3.52562079136553e-10, 7.15652805269027e-12, 8.80570764521616e-14, 6.00835878949095e-16, 1.98918101211646e-18, 2.56759336541157e-21, 8.54405696377538e-25, 2.59104371384712e-29 }, 
                       ws_few [n_ghq_few ] = { 2.22939364553414e-13, 4.39934099227314e-10, 1.08606937076927e-07, 7.80255647853208e-06, 0.000228338636016353, 0.00324377334223785, 0.0248105208874637, 0.109017206020023, 0.286675505362834, 0.46224366960061, 0.46224366960061, 0.286675505362835, 0.109017206020023, 0.0248105208874636, 0.00324377334223785, 0.000228338636016355, 7.80255647853212e-06, 1.08606937076928e-07, 4.39934099227318e-10, 2.22939364553414e-13 }, 
                   ws_many_log[n_ghq_many] = { -65.8229069239704, -55.419391375547, -47.4113179263878, -40.7588086715673, -35.0482098566024, -30.0607911946327, -25.6629961614044, -21.7657943981563, -18.3062490214757, -15.2379942594073, -12.5258438026328, -10.1425232219853, -8.06658394021126, -6.28101672466453, -4.77230114535736, -3.52973899700253, -2.54498019868911, -1.81168425978064, -1.32528110188264, -1.08280800461648, -1.08280800461647, -1.32528110188264, -1.81168425978064, -2.5449801986891, -3.52973899700253, -4.77230114535736, -6.28101672466452, -8.06658394021126, -10.1425232219853, -12.5258438026328, -15.2379942594073, -18.3062490214757, -21.7657943981563, -25.6629961614044, -30.0607911946327, -35.0482098566023, -40.7588086715673, -47.4113179263878, -55.419391375547, -65.8229069239705 },
                   ws_few_log [n_ghq_few ] = { -29.1318765682564, -21.5443961747169, -16.0355305541669, -11.7610591243103, -8.38468078521053, -5.73101801501035, -3.69648748643781, -2.21624955581582, -1.24940434621696, -0.771663103561278, -0.771663103561278, -1.24940434621696, -2.21624955581582, -3.69648748643781, -5.73101801501035, -8.38468078521053, -11.7610591243103, -16.0355305541669, -21.5443961747169, -29.1318765682564 };
/// the GHQ nodes 
constexpr double const nodes_many[n_ghq_many] = { -8.09876113925084, -7.41158253148548, -6.84023730524935, -6.32825535122008, -5.8540950560304, -5.40665424797012, -4.97926097854525, -4.5675020728444, -4.1682570668325, -3.77920675343523, -3.39855826585963, -3.02487988390128, -2.6569959984429, -2.29391714187508, -1.9347914722823, -1.57886989493161, -1.22548010904629, -0.874006612357088, -0.523874713832277, -0.174537214597582, 0.174537214597582, 0.523874713832277, 0.874006612357089, 1.22548010904629, 1.57886989493162, 1.9347914722823, 2.29391714187508, 2.6569959984429, 3.02487988390129, 3.39855826585963, 3.77920675343522, 4.1682570668325, 4.56750207284439, 4.97926097854526, 5.40665424797013, 5.8540950560304, 6.32825535122008, 6.84023730524936, 7.41158253148547, 8.09876113925085 },
                       nodes_few [n_ghq_few ] = { -5.38748089001123, -4.60368244955075, -3.94476404011563, -3.34785456738322, -2.78880605842813, -2.25497400208928, -1.73853771211659, -1.23407621539532, -0.737473728545394, -0.245340708300901, 0.245340708300901, 0.737473728545395, 1.23407621539532, 1.73853771211659, 2.25497400208928, 2.78880605842813, 3.34785456738322, 3.94476404011563, 4.60368244955074, 5.38748089001123 };
} // namespace GHQ

namespace partition {
/** contains expected partitions functions */

constexpr double const sqrt_2pi_inv = 0.398942280401433, 
                             sqrt_2 = 1.4142135623731, 
                        sqrt_pi_inv = 0.564189583547756;

/** expected partition functions for the logit link with binomial data */
template<bool adaptive>
struct logit {
  /** finds the mode of the integrand. */
  struct mode_finder {
    double const mu; 
    double const sigma;
    
    mode_finder(double const mu, double const sigma): 
      mu(mu), sigma(sigma) { }
    
    inline double optimfunc(double const x) noexcept {
      double const eta = sigma * x + mu, 
                p_term = eta > 30 ? eta :  std::log(1 + std::exp(eta));
      return .5 * x * x - std::log(p_term);
    }
    
    double operator()() noexcept {
      if(sigma < 2 or std::abs(mu) < 4){
        // 4th order polynomial approximation 
        constexpr std::array<double, 15> const coef = { 0.00269191424089733, -0.0135120139816612, 0.000596313286406067, 0.00132194254552531, -9.65239787158926e-05, 0.693071200579536, -0.109014356271539, -0.0056401414162169, 0.000581436402165448, 0.00692133354547494, -0.0204524311765302, 0.00586824383813473, -0.100822202289977, 0.0160140669127429, 0.0166017681050071 };
        int ci(0L);
        double out(coef[ci++]);
        
        {
          double m = 1;
          for(int i = 0; i < 4; ++i){
            m *= mu;
            out += m * coef[ci++];
          }
        }
        double s = 1;
        for(int i = 0; i < 4; ++i){
          s *= sigma;
          double m = 1;
          for(int j = 0; j < 4 - i; ++j){
            out += m * s * coef[ci++];
            m *= mu;
          }
        }
        
        return std::max(1e-8, out);
      }
      
      // we work on the log scale
      constexpr double const eps = 1e-4;
      double ub = std::min(3., sigma), 
             lb = eps; // optimum has a lower bounded of zero 
      
      double out(std::numeric_limits<double>::quiet_NaN());
      for(int i = 0; i < 100; ++i){
        out = Brent_fmin(lb, ub, *this, eps);

        // check that we are not at a boundary
        if(std::abs(out - ub) < 1.25 * eps){
          double const diff = ub - lb;
          lb = ub - 2  * eps;
          ub += 5 * diff;
          continue;
        }

        break;
      }

      return out;
    }
  };
  
  /// computes the scale to use in adaptive Gauss–Hermite quadrature.
  static inline double get_scale(double const mode_eta, 
                                 double const sigma) noexcept {
    if(mode_eta > 50){
      double const h1 = mode_eta, 
                   h2 = -1, 
                 hess = -1 + (sigma / h1) * (sigma / h1) * h2;
      return std::sqrt(-1/hess);
      
    }
    
    double const exp_m = exp(mode_eta), 
                    h1 = log(1 + exp_m), 
                    h2 = (exp_m / (1 + exp_m)) * 
                      ((h1 - exp_m) / (1 + exp_m)), 
                  hess = -1 + (sigma / h1) * (sigma / h1) * h2;
    return std::sqrt(-1/hess);
  }
  
  /// evalutes the expected partition function 
  template<double const ns[], double const ws[], double const ws_log[], 
           int n_nodes>
  static inline double B_inner
    (double const mu, double const sigma) noexcept {
    double out(0.);
    
    if(!adaptive){
      for(int i = 0; i < n_nodes; ++i){
        double const x = sqrt_2 * sigma * ns[i] + mu, 
                  mult = x > 30 ? x : std::log(1 + exp(x));
        out += mult * ws[i];
      }
      
      return out * sqrt_pi_inv;
    }
    
    // get the mode and scale to use
    double const mode = mode_finder(mu, sigma)(),
                scale = get_scale(sigma * mode + mu, sigma);

    for(int i = 0; i < n_nodes; ++i){
      double const y = ns[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu, 
                mult = x > 30 ? x : std::log(1 + exp(x));
      out += mult * exp(y * y - z * z * .5 + ws_log[i]);
    }
    
    return out * sqrt_2pi_inv * scale;
  }
  
  static inline double B(double const mu, double const sigma) noexcept {
    bool const use_many = mu * mu * 0.1111111 + sigma * sigma > 1;
    return use_many ? 
      B_inner<ghq::nodes_many, ghq::ws_many, ghq::ws_many_log, 
              ghq::n_ghq_many>(mu, sigma) : 
      B_inner<ghq::nodes_few , ghq::ws_few , ghq::ws_few_log ,
              ghq::n_ghq_few >(mu, sigma);
  }
  
  /// evaluates the derivatives of the expected partition function 
  template<double const ns[], double const ws[], double const ws_log[],
           int n_nodes>
  static inline std::array<double, 2L> Bp_inner
  (double const mu, double const sigma) noexcept {
    std::array<double, 2L> out = { 0, 0 };
    if(!adaptive){
      for(int i = 0; i < n_nodes; ++i){
        double const mult = sqrt_2 * ns[i],
                        x = mult * sigma + mu, 
                    d_eta = ws[i] / (1 + exp(-x));
        out[0] +=        d_eta; // dmu
        out[1] += mult * d_eta; // dsigma
      }
      
      out[0] *= sqrt_pi_inv;
      out[1] *= sqrt_pi_inv;
      return out;
    }
    
    // get the mode and scale to use
    double const mode = mode_finder(mu, sigma)(), 
                scale = get_scale(sigma * mode + mu, sigma);
    
    for(int i = 0; i < n_nodes; ++i){
      double const y = ns[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu,
               d_eta = exp(y * y - z * z * .5 + ws_log[i]) / (1 + exp(-x));
      out[0] +=     d_eta; // dmu
      out[1] += z * d_eta; // dsigma
    }
    
    double const w = sqrt_2pi_inv * scale;
    out[0] *= w;
    out[1] *= w;
    return out;
  }
  
  static inline std::array<double, 2L> Bp
    (double const mu, double const sigma) noexcept {
    bool const use_many = mu * mu * 0.1111111 + sigma * sigma > 1;
    return use_many ? 
      Bp_inner<ghq::nodes_many, ghq::ws_many, ghq::ws_many_log, 
               ghq::n_ghq_many>(mu, sigma) : 
      Bp_inner<ghq::nodes_few , ghq::ws_few , ghq::ws_few_log ,
               ghq::n_ghq_few >(mu, sigma);
  }
};
} // namespace partition

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector logit_partition
  (double const mu, double const sigma, int const order, 
   bool const adaptive = true){
  if       (order == 0L){
    Rcpp::NumericVector out(1);
    out(0) = 
      adaptive ? partition::logit<true >::B(mu, sigma) : 
                 partition::logit<false>::B(mu, sigma);
    return out;
    
  } else if(order == 1L){
    Rcpp::NumericVector out(2);
    auto res = adaptive ? partition::logit<true >::Bp(mu, sigma) : 
                          partition::logit<false>::Bp(mu, sigma);
    out[0] = res[0];
    out[1] = res[1];
    return out;
    
  }
  
  return Rcpp::NumericVector();
}

// test the partition functions
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


// implement the lower bound 

/// simple function to avoid copying a vector. You can ignore this
inline arma::vec vec_no_cp(double const * x, size_t const n_ele){
  return arma::vec(const_cast<double *>(x), n_ele, false);
}

/** Computes LL^T where L is a lower triangular matrix. The argument is a
    a vector with the non-zero elements in column major order. The diagonal 
    entries are on the log scale. The method computes both L and LL^T.  */
void get_pd_mat(double const *theta, arma::mat &L, arma::mat &res){
  int const dim = L.n_rows;
  L.zeros();
  for(int j = 0; j < dim; ++j){
    L.at(j, j) = std::exp(*theta++);
    for(int i = j + 1; i < dim; ++i)
      L.at(i, j) = *theta++;
  }
  
  res = L * L.t();
}

arma::uvec get_commutation_unequal_vec
  (unsigned const n, unsigned const m, bool const transpose){
  unsigned const nm = n * m,
            nnm_p1 = n * nm + 1L,
             nm_pm = nm + m;
  arma::uvec out(nm);
  arma::uword * const o_begin = out.begin();
  size_t idx = 0L;
  for(unsigned i = 0; i < n; ++i, idx += nm_pm){
    size_t idx1 = idx;
    for(unsigned j = 0; j < m; ++j, idx1 += nnm_p1)
      if(transpose)
        *(o_begin + idx1 / nm) = (idx1 % nm);
      else
        *(o_begin + idx1 % nm) = (idx1 / nm);
  }

  return out;
}

// cached version of the above
arma::uvec const & get_commutation_unequal_vec_cached(unsigned const n){
  constexpr std::size_t n_cache = 1000L;
  if(n > n_cache or n == 0L)
    throw std::invalid_argument(
        "get_commutation_unequal_vec_cached: invalid n (too large or zero)");

  static std::array<arma::uvec, n_cache> cached_values;

  unsigned const idx = n - 1L;
  bool has_value = cached_values[idx].n_elem > 0;

  if(has_value)
    return cached_values[idx];

#ifdef _OPENMP
#pragma omp critical (coomuCached)
{
#endif
  has_value = cached_values[idx].n_elem > 0;
  if(!has_value)
    cached_values[idx] =
      get_commutation_unequal_vec(n, n, false);
#ifdef _OPENMP
}
#endif
  return cached_values[idx];
}

/**
 * computes x (X (x) I) where X is a k x p matrix, I is an l dimensional
 * diagonal matrix and x is an l x k vector. The result is stored in the
 * p x l dimensional output.
 */
inline void x_dot_X_kron_I
  (arma::vec const &x, arma::mat const &X, int const l,
   double * const __restrict__ out) {
  int const k = X.n_rows,
            p = X.n_cols,
           pl = p * l;
  std::fill(out, out + pl, 0.);

  for(int c = 0; c < p; ++c){
    for(int r = 0; r < k; ++r){
      double const mult = X.at(r, c);
      double const * const x_end = x.memptr() + r * l + l;
      double * o = out + c * l;
      for(double const * xp = x.memptr() + r * l;
          xp != x_end; ++xp, ++o)
        *o += *xp * mult;
    }
  }
}

/** computes the deratives of get_pd_mat. gr is a Jacobian vector 
    which is used as part of the chain rule. res holds the memory for the 
    result and has the same dimension as the pointer passed to 
    get_pd_mat. */
void d_get_pd_mat(arma::vec const &gr, arma::mat const &L, 
                  double * __restrict__ res, double * const work_mem){
  arma::vec gr_term(work_mem, gr.size(), false);
  std::copy(gr.begin(), gr.end(), gr_term.begin());
  gr_term += gr(get_commutation_unequal_vec_cached(L.n_cols));
  
  int const n = L.n_cols;
  arma::mat jac(gr_term.end(), n, n, false);
  x_dot_X_kron_I(gr_term, L, n, jac.begin());

  // copy the lower triangel and scale the diagonal entries  
  double * r = res;
  for(int j = 0; j < n; ++j){
    *r++ = L.at(j, j) * jac.at(j, j);
    for(int i = j + 1; i < n; ++i)
      *r++ = jac.at(i, j);
  }
}

// functions to check the above
// [[Rcpp::export(rng = false)]]
Rcpp::List get_pd_mat(Rcpp::NumericVector theta, int const dim){
  arma::mat L(dim, dim), 
          res(dim, dim);
  get_pd_mat(&theta[0], L, res);
  return Rcpp::List::create(Rcpp::Named("X") = res, 
                            Rcpp::Named("L") = L);
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector d_get_pd_mat(arma::vec const &gr, arma::mat const &L){
  int const n = L.n_cols;
  Rcpp::NumericVector out((n * (n + 1)) / 2);
  std::unique_ptr<double[]> wk_mem(new double[gr.n_elem + L.n_elem]);
  d_get_pd_mat(gr, L, &out[0], wk_mem.get());
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

inline double vec_dot(arma::vec const &x, double const *y) noexcept {
  double out(0.);
  for(arma::uword i = 0; i < x.n_elem; ++i)
    out += x.at(i) * *y++;
  return out;
}

inline double quad_form(arma::mat const &X, double const *x) noexcept {
  arma::uword const n = X.n_rows;
  double out(0.);
  for(arma::uword j = 0; j < n; ++j){
    out += X.at(j, j) * x[j] * x[j];
    for(arma::uword i = j + 1; i < n; ++i)
      out += 2 * X.at(i, j) * x[j] * x[i];
  }
  
  return out;
}

class lower_bound_term {
  arma::vec const y, nis;
  arma::mat const X, Z;
  
public:
  int const n_beta = X.n_rows, 
             n_rng = Z.n_rows,
             n_sig = (n_rng * (n_rng + 1L)) / 2L,
             n_obs = y.n_elem;
  
  double const norm_constant;
  
private:
  static int mem_per_thread;
  static std::unique_ptr<double[]> wk_mem;
  static inline double * get_thread_mem() noexcept {
#ifdef _OPENMP
    int const thread_num(omp_get_thread_num());
#else
    int const thread_num(0L));
#endif
    
    return wk_mem.get() + thread_num * mem_per_thread;
  }
  
public:
  lower_bound_term(arma::vec const &y, arma::vec const &nis, 
                   arma::mat const &X, arma::mat const &Z): 
  y(y), nis(nis), X(X), Z(Z), 
  norm_constant(([&]() -> double {
    if(static_cast<size_t>(n_obs) != nis.n_elem)
      // have to check this now
      throw std::invalid_argument("invalid nis");
    
    double out(n_rng / 2.);
    for(int i = 0; i < n_obs; ++i)
      out += std::lgamma(nis[i] + 1) - std::lgamma(y[i] + 1) - 
        std::lgamma(nis[i] - y[i] + 1);
    return -out;
  })()) {
    // checks
    if(X.n_cols != static_cast<size_t>(n_obs))
      throw std::invalid_argument("invalid X");
    if(Z.n_cols != static_cast<size_t>(n_obs))
      throw std::invalid_argument("invalid X");
  }
  
  lower_bound_term(Rcpp::List dat):
  lower_bound_term(Rcpp::as<arma::vec>(dat["y"]),
                   Rcpp::as<arma::vec>(dat["nis"]),
                   Rcpp::as<arma::mat>(dat["X"]),
                   Rcpp::as<arma::mat>(dat["Z"])) { }
  
  static void set_wk_mem(int const max_n_beta, int const max_n_rng, 
                         int const max_n_obs, int const max_threads){
    constexpr int const mult = cacheline_size() / sizeof(double),
                    min_size = 2L * mult;
    
    int n_ele = std::max(
      static_cast<int>(10L * max_n_rng * max_n_rng), min_size);
    n_ele = (n_ele + mult - 1L) / mult;
    n_ele *= mult;
    
    if(mem_per_thread < n_ele){
      mem_per_thread = n_ele;
      wk_mem.reset(new double[n_ele * max_threads]);
    }
  }
  
  size_t global_dim() const {
    return n_beta + n_sig;
  }
  size_t private_dim() const {
    return n_rng + n_sig;
  }
  
  template<bool comp_grad>
  double comp(double const *p, double *gr) const {
    // setup the objects we need
    int const beta_start = 0, 
               Sig_start = beta_start + n_beta, 
             va_mu_start = Sig_start + n_sig, 
               Lam_start = va_mu_start + n_rng;
    arma::vec const beta = vec_no_cp(p + beta_start , n_beta), 
                   va_mu = vec_no_cp(p + va_mu_start, n_rng); 
    
    double * w = get_thread_mem();
    auto get_wk_mem = [&](int const n_ele){
      double * out = w;
      w += n_ele;
      return out;
    };
    
    int const n_rng_sq = n_rng * n_rng;
    arma::mat Sig_L(get_wk_mem(n_rng_sq), n_rng, n_rng, false), 
              Sig  (get_wk_mem(n_rng_sq), n_rng, n_rng, false), 
              Lam_L(get_wk_mem(n_rng_sq), n_rng, n_rng, false), 
              Lam  (get_wk_mem(n_rng_sq), n_rng, n_rng, false);
    get_pd_mat(p + Sig_start, Sig_L, Sig);
    get_pd_mat(p + Lam_start, Lam_L, Lam);
    
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
    for(int i = 0; i < n_obs; ++i){
      double const eta = 
        vec_dot(beta, X.colptr(i)) + vec_dot(va_mu, Z.colptr(i)), 
              cond_sd  = std::sqrt(std::abs(
                quad_form(Lam, Z.colptr(i))));
      double const B = partition::logit<true>::B(eta, cond_sd);
      out += -y[i] * eta + nis[i] * B;
      
      if(comp_grad){
        auto const Bp = partition::logit<true>::Bp(eta, cond_sd);
        double const d_eta = -y[i] + nis[i] * Bp[0];
        dbeta  += d_eta * X.col(i);
        dva_mu += d_eta * Z.col(i);
        
        double const mult = nis[i] * Bp[1] / cond_sd * .5;
        for(int k = 0; k < n_rng; ++k){
          for(int j = 0; j < k; ++j){
            double const term = mult * Z.at(k, i) * Z.at(j, i); 
            dLam.at(j, k) += term;
            dLam.at(k, j) += term;
          }
          dLam.at(k, k) += mult * Z.at(k, i) * Z.at(k, i); 
        }
      }
    }
    
    // terms from the log of the ratio of the unconditional random effect 
    // density and the variational density
    double half_term(0.), 
               deter, 
              unused;
    
    // determinant terms
    arma::mat sig_inv(get_wk_mem(n_rng_sq), n_rng, n_rng, false);
    if(!arma::inv_sympd(sig_inv, Sig)){
      sig_inv.zeros();
      half_term = std::numeric_limits<double>::quiet_NaN();
    }
    
    arma::log_det(deter, unused, Sig);
    half_term -= deter;
    arma::log_det(deter, unused, Lam);
    half_term += deter;
    
    // TODO: not numerically stable
    arma::vec const sig_inv_va_mu = sig_inv * va_mu; // TODO: memory allocation
    half_term -= vec_dot(sig_inv_va_mu, va_mu.begin());
    for(int i = 0; i < n_rng; ++i){
      for(int j = 0; j < i; ++j){ 
        half_term -= 2 * sig_inv.at(j, i) * Lam.at(j, i);
        if(comp_grad){
          double const d_term = .5 * sig_inv.at(j, i);
          dLam.at(j, i) += d_term;
          dLam.at(i, j) += d_term;
          dSig.at(j, i) += d_term;
          dSig.at(i, j) += d_term;
        }
      }
      half_term -= sig_inv.at(i, i) * Lam.at(i, i);
      if(comp_grad){
        dLam.at(i, i) += .5 * sig_inv.at(i, i);
        dSig.at(i, i) += .5 * sig_inv.at(i, i);
      }
    }
    
    out -= .5 * half_term;
    if(comp_grad){
      dva_mu += sig_inv_va_mu;
      
      {
        arma::mat lam_inv(
            get_wk_mem(n_rng_sq), n_rng, n_rng, false); 
        if(!arma::inv_sympd(lam_inv, Lam))
          half_term = std::numeric_limits<double>::quiet_NaN();
        else
          dLam -= .5 * lam_inv; 
      }
      
      // modify Lam as we do not need it anymore
      double * const d_pd_mem = // last working memory we can use
        get_wk_mem(2 * n_rng_sq);
      for(int i = 0; i < n_rng; ++i)
        for(int j = 0; j < n_rng; ++j)
          Lam.at(j, i) += va_mu[i] * va_mu[j];
      arma::mat tmp(d_pd_mem, n_rng, n_rng, false);
      tmp = .5 * (sig_inv * Lam * sig_inv); // TODO: many temporaries ?
      dSig -= tmp; 
      
      // copy the result 
      {
        arma::vec dum(dSig.begin(), dSig.n_elem, false);
        d_get_pd_mat(dum, Sig_L, gr + Sig_start, d_pd_mem);
      }
      {
        arma::vec dum(dLam.begin(), dLam.n_elem, false);
        d_get_pd_mat(dum, Lam_L, gr + Lam_start, d_pd_mem);
      }
    }
    
    return out;
  }
  
  double func(double const *point) const {
    return comp<false>(point, nullptr);
  }
  
  double grad
  (double const * point, double * gr) const {
    return comp<true>(point, gr);
  }
  
  bool thread_safe() const {
    return true;
  }
};

int lower_bound_term::mem_per_thread = 0L;
std::unique_ptr<double[]> lower_bound_term::wk_mem = 
  std::unique_ptr<double[]>();

// psqn interface 
using lb_optim = PSQN::optimizer<lower_bound_term, PSQN::R_reporter,
                                 PSQN::R_interrupter>;

// [[Rcpp::export(rng = false)]]
SEXP get_lb_optimizer(Rcpp::List data, unsigned const max_threads){
  size_t const n_elem_funcs = data.size();
  std::vector<lower_bound_term> funcs;
  funcs.reserve(n_elem_funcs);
  
  int max_n_beta(0L), 
      max_n_rng (0L), 
      max_n_obs(0L);
  for(auto dat : data){
    funcs.emplace_back(Rcpp::List(dat));
    lower_bound_term const &obj = funcs.back();
    max_n_beta = std::max(max_n_beta, obj.n_beta);
    max_n_rng  = std::max(max_n_rng , obj.n_rng);
    max_n_obs = std::max(max_n_obs, obj.n_obs);
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
   double const c2, bool const use_bfgs = true, int const trace = 0L,
   double const cg_tol = .5, bool const strong_wolfe = true,
   size_t const max_cg = 0L, int const pre_method = 1L){
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
         nis = nis, X = Z, Z = Z)
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

set.seed(1)
point <- runif(
  n_fix + n_clust * n_rng + (n_clust + 1) * n_rng * (n_rng + 1L) / 2,
  -1, 1)

fn(point)
library(numDeriv)
num_aprx <- grad(fn, point, method.args = list(r = 10))
gr_cpp <- gr(point)
cbind(num_aprx, gr_cpp, diff = gr_cpp - num_aprx)
all.equal(num_aprx, gr_cpp, 
          check.attributes = FALSE)

# compare w/ Laplace approximation. First assign function to estimate the 
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

  # par <- opt_priv(val = par, ptr = func, rel_eps = rel_eps^(2/3),
  #                 max_it = 100, n_threads = 1L, c1 = 1e-4, c2 = .9)
  res <- opt_lb(val = par, ptr = func, rel_eps = rel_eps, max_it = 1000L, 
                n_threads = 1L, c1 = 1e-4, c2 = .9, cg_tol = .2, 
                max_cg = max(2L, as.integer(log(n_clust) * 10)))
  
  mod_par <- head(res$par, n_fix + n_rng * (n_rng + 1) / 2)
  Sig_hat <- get_pd_mat(tail(mod_par, -n_fix), n_rng)[[1L]]
  
  list(lb = -res$value, fixef = head(mod_par, n_fix), 
       stdev = sqrt(diag(Sig_hat)), cormat = cov2cor(Sig_hat))
}

# # then simulate and use the functions
set.seed(1)
n_clust <- 1000L
dat <- sim_dat(sig = .6^2, inter = 1, n_cluster = n_clust, slope = -1)
system.time(print(est_va     (dat)))
system.time(print(est_Laplace(dat)))

# truth is
list(fixef  = dat$beta, stdev = diag(sqrt(dat$vcov_mat)), 
     cormat = cov2cor(dat$vcov_mat))

# double const eta = sigma * x + mu, 
# p_term = eta > 30 ? eta :  std::log(1 + std::exp(eta));
# return .5 * x * x - std::log(p_term);
*/
