// [[Rcpp::plugins(cpp11)]]
#include <cmath>
#include <Rcpp.h>
#include <array>
#include <limits>

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
constexpr int const n_ghq = 28L;
/// the GHQ weights 
constexpr double const ws[n_ghq] = { 1.14013934790367e-19, 8.3159379512068e-16, 6.63943671490957e-13, 1.4758531682777e-10, 1.32568250154172e-08, 5.857719720993e-07, 1.43455042297144e-05, 0.000210618100024033, 0.00195733129440897, 0.0119684232143548, 0.0495148892898983, 0.141394609786955, 0.282561391259389, 0.398604717826451, 0.398604717826451, 0.282561391259388, 0.141394609786955, 0.0495148892898982, 0.0119684232143548, 0.00195733129440898, 0.000210618100024033, 1.43455042297145e-05, 5.85771972099299e-07, 1.32568250154172e-08, 1.47585316827766e-10, 6.63943671490963e-13, 8.31593795120661e-16, 1.14013934790364e-19 };
/// the GHQ nodes 
constexpr double const nodes[n_ghq] = { -6.59160544236775, -5.85701464138285, -5.24328537320294, -4.69075652394311, -4.17663674212927, -3.68913423846168, -3.22111207656146, -2.76779535291359, -2.32574984265644, -1.89236049683768, -1.46553726345741, -1.04353527375421, -0.624836719505209, -0.208067382690736, 0.208067382690737, 0.624836719505209, 1.04353527375421, 1.46553726345741, 1.89236049683769, 2.32574984265644, 2.76779535291359, 3.22111207656145, 3.68913423846168, 4.17663674212927, 4.69075652394312, 5.24328537320293, 5.85701464138285, 6.59160544236774 };
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
    
    inline double optimfunc(double x) noexcept {
      double const eta = sigma * x + mu, 
                p_term = eta > 30 ? eta :  std::log(1 + std::exp(eta));
      return .5 * x * x - std::log(p_term);
    }
    
    double operator()() noexcept {
      double lb = 0, 
             ub = sigma, 
            out(std::numeric_limits<double>::quiet_NaN());
      constexpr double const eps = 1e-4;
      for(int i = 0; i < 100; ++i){
        out = Brent_fmin(lb, ub, *this, eps);
        
        // check that we not at a boundary
        double const diff = ub - lb;
        if(std::abs(out - lb) < 1.25 * eps){
          ub = lb;
          lb -= 5 * diff;
          continue;
        }
        if(std::abs(out - ub) < 1.25 * eps){
          lb = ub;
          ub += 5 * diff;
          continue;
        }
        
        break;
      }
      
      return out;
    }
  };
  
  /// evalutes the expected partition function 
  static inline double B(double const mu, double const sigma) noexcept {
    double out(0.);
    
    if(!adaptive){
      for(int i = 0; i < ghq::n_ghq; ++i){
        double const x = sqrt_2 * sigma * ghq::nodes[i] + mu, 
                  mult = x > 30 ? x : std::log(1 + exp(x));
        out += mult * ghq::ws[i];
      }
      
      return out * sqrt_pi_inv;
    }
    
    // get the mode and scale to use
    double const mode = mode_finder(mu, sigma)(),
             mode_eta = sigma * mode + mu,
                exp_m = exp(mode_eta),
                   h1 = mode_eta > 30 ? mode_eta : log(1 + exp_m), 
                 hess = 
                   -1 + sigma * sigma * exp_m / (1 + exp_m) / (1 + exp_m) * 
                   (h1 - exp_m) / h1 / h1,
                scale = std::sqrt(-1/hess);
    
    for(int i = 0; i < ghq::n_ghq; ++i){
      double const y = ghq::nodes[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu, 
                mult = x > 30 ? x : std::log(1 + exp(x));
      out += mult * exp(y * y - z * z * .5) * ghq::ws[i];
    }
    
    return out * sqrt_2pi_inv * scale;
  }
  
  /// evaluates the derivatives of the expected partition function 
  static inline std::array<double, 2L> Bp
  (double const mu, double const sigma) noexcept {
    std::array<double, 2L> out = { 0, 0 };
    if(!adaptive){
      for(int i = 0; i < ghq::n_ghq; ++i){
        double const mult = sqrt_2 * ghq::nodes[i],
                        x = mult * sigma + mu, 
                    d_eta = ghq::ws[i] / (1 + exp(-x));
        out[0] +=        d_eta; // dmu
        out[1] += mult * d_eta; // dsigma
      }
      
      out[0] *= sqrt_pi_inv;
      out[1] *= sqrt_pi_inv;
      return out;
    }
    
    // get the mode and scale to use
    double const mode = mode_finder(mu, sigma)(),
             mode_eta = sigma * mode + mu,
                exp_m = exp(mode_eta),
                   h1 = mode_eta > 30 ? mode_eta : log(1 + exp_m), 
                 hess = 
                   -1 + sigma * sigma * exp_m / (1 + exp_m) / (1 + exp_m) * 
                   (h1 - exp_m) / h1 / h1,
                scale = std::sqrt(-1/hess);
    
    for(int i = 0; i < ghq::n_ghq; ++i){
      double const y = ghq::nodes[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu,
               d_eta = exp(y * y - z * z * .5) * ghq::ws[i] / (1 + exp(-x));
      out[0] +=     d_eta; // dmu
      out[1] += z * d_eta; // dsigma
    }
    
    double const w = sqrt_2pi_inv * scale;
    out[0] *= w;
    out[1] *= w;
    return out;
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
mus <- seq(-4, 4, length.out = 25)
sigmas <- seq(.1, 4, length.out = 25)
grid <- expand.grid(mu = mus, sigma = sigmas)

rel_err <- mapply(function(mu, sigma){
  truth <- integrate(f, -Inf, Inf, mu = mu, sigma = sigma, rel.tol = 1e-12)
  est <- logit_partition(mu = mu, sigma = sigma, order = 0)
  (truth$value - est) / truth$value 
}, mu = grid$mu, sigma = grid$sigma)

# plot the errors
range(rel_err) # range of the relative errors
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
contour(mus, sigmas, matrix(rel_err[1,], length(mus)), 
        xlab = expression(mu), ylab = expression(sigma), 
        main = "Relative error of dE(logit partition) / dmu")

range(rel_err[2, ]) # range of the relative errors (dsigma)
contour(mus, sigmas, matrix(rel_err[2,], length(mus)), 
        xlab = expression(mu), ylab = expression(sigma), 
        main = "Relative error of dE(logit partition) / dsigma")

# # check the computation time
bench::mark( f = logit_partition(mu = 1, sigma = 2, order = 0),
            df = logit_partition(mu = 1, sigma = 2, order = 1),
            min_time = 1, max_iterations = 1e6, check = FALSE)
*/
