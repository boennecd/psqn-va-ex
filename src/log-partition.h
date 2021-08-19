#pragma once

#include "brent.h"
#include "ghq.h"

namespace partition {
/** contains expected log partition functions */

constexpr double sqrt_2pi_inv{0.398942280401433}, 
                       sqrt_2{1.4142135623731}, 
                  sqrt_pi_inv{0.564189583547756};

/** expected log partition functions for the logit link with binomial data */
struct logit {  
  /// computes the scale to use in adaptive Gauss–Hermite quadrature.
  static double get_scale(double const mode_eta, double const sigma) noexcept {
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
  
  /// evaluates the expected log partition function 
  template<unsigned n_nodes>
  static double B_inner
    (double const mu, double const sigma, bool const adaptive) noexcept {
    double out(0.);
    using nw = ghq::nw_pairs<n_nodes>;
    
    if(!adaptive){
      for(unsigned i = 0; i < n_nodes; ++i){
        double const x = sqrt_2 * sigma * nw::ns[i] + mu, 
                  mult = x > 30 ? x : std::log(1 + exp(x));
        out += mult * nw::ws[i];
      }
      
      return out * sqrt_pi_inv;
    }
    
    // get the mode and scale to use
    double const mode{mode_finder(mu, sigma)()},
                scale{get_scale(sigma * mode + mu, sigma)};

    for(unsigned i = 0; i < n_nodes; ++i){
      double const y = nw::ns[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu, 
                mult = x > 30 ? x : std::log(1 + exp(x));
      out += mult * exp(y * y - z * z * .5 + nw::ws_log[i]);
    }
    
    return out * sqrt_2pi_inv * scale;
  }
  
  static double B
    (double const mu, double const sigma, bool const adaptive) noexcept {
    bool const use_many = mu * mu * 0.1111111 + sigma * sigma > 1;
    return use_many 
      ? B_inner<40>(mu, sigma, adaptive) 
      : B_inner<20>(mu, sigma, adaptive);
  }
  
  /**
   * evaluates the derivatives of the expected log partition function. The last 
   * element is the expected log partition function
   */
  template<unsigned n_nodes>
  static std::array<double, 3L> Bp_inner
  (double const mu, double const sigma, bool const adaptive) noexcept {
    std::array<double, 3L> out { 0, 0, 0 };
    using nw = ghq::nw_pairs<n_nodes>;
    
    if(!adaptive){
      for(unsigned i = 0; i < n_nodes; ++i){
        double const mult = sqrt_2 * nw::ns[i],
                        x = mult * sigma + mu, 
                    d_eta = nw::ws[i] / (1 + exp(-x)), 
                     func = x > 30 ? x : std::log(1 + exp(x));
        out[0] +=        d_eta; // dmu
        out[1] += mult * d_eta; // dsigma
        out[2] += nw::ws[i] * func;
      }
      
      out[0] *= sqrt_pi_inv;
      out[1] *= sqrt_pi_inv;
      out[2] *= sqrt_pi_inv;
      
      return out;
    }
    
    // get the mode and scale to use
    double const mode{mode_finder(mu, sigma)()}, 
                scale{get_scale(sigma * mode + mu, sigma)};
    
    for(unsigned i = 0; i < n_nodes; ++i){
      double const y = nw::ns[i], 
                   z = scale * y + mode,
                   x = sigma * z + mu,
                func = exp(y * y - z * z * .5 + nw::ws_log[i]),
               d_eta = func / (1 + exp(-x)), 
                mult = x > 30 ? x : std::log(1 + exp(x));
      out[0] +=     d_eta; // dmu
      out[1] += z * d_eta; // dsigma
      out[2] += mult * func;
    }
    
    double const w = sqrt_2pi_inv * scale;
    out[0] *= w;
    out[1] *= w;
    out[2] *= w;
    
    return out;
  }
  
  static std::array<double, 3L> Bp
    (double const mu, double const sigma, bool const adaptive) noexcept {
    bool const use_many = mu * mu * 0.1111111 + sigma * sigma > 1;
    return use_many 
      ? Bp_inner<40>(mu, sigma, adaptive)
      : Bp_inner<20>(mu, sigma, adaptive);
  }
  
  /// twice differential of the log partition function
  static double dd_eta(double const x) noexcept {
    double const exp_x = exp(x);
    return exp_x / (1 + exp_x) / (1 + exp_x);
  }
  
  /** finds the mode of the integrand. */
  struct mode_finder {
    double const mu; 
    double const sigma;
    
    mode_finder(double const mu, double const sigma): 
      mu(mu), sigma(sigma) { }
    
    double optimfunc(double const x) noexcept {
      double const eta = sigma * x + mu, 
                p_term = eta > 30 ? eta :  std::log(1 + std::exp(eta));
      return .5 * x * x - std::log(p_term);
    }
    
    double operator()() noexcept {
      if(sigma < 2 or std::abs(mu) < 4){
        // 4th order polynomial approximation 
        constexpr std::array<double, 15> const coef { 0.00269191424089733, -0.0135120139816612, 0.000596313286406067, 0.00132194254552531, -9.65239787158926e-05, 0.693071200579536, -0.109014356271539, -0.0056401414162169, 0.000581436402165448, 0.00692133354547494, -0.0204524311765302, 0.00586824383813473, -0.100822202289977, 0.0160140669127429, 0.0166017681050071 };
        unsigned ci{};
        double out(coef[ci++]);
        
        {
          double m = 1;
          for(unsigned i = 0; i < 4; ++i){
            m *= mu;
            out += m * coef[ci++];
          }
        }
        double s = 1;
        for(unsigned i = 0; i < 4; ++i){
          s *= sigma;
          double m = 1;
          for(unsigned j = 0; j < 4 - i; ++j){
            out += m * s * coef[ci++];
            m *= mu;
          }
        }
        
        return std::max(1e-8, out);
      }
      
      constexpr double eps{1e-4};
      double ub = std::min(3., sigma), 
             lb = eps; // optimum has a lower bounded of zero 
      
      double out(std::numeric_limits<double>::quiet_NaN());
      for(unsigned i = 0; i < 100; ++i){
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
};

/** expected log partition functions for the log link with Poisson data. */
struct poisson {
  static double B(double const mu, double const sigma) noexcept {
    return std::exp(mu + sigma * sigma * .5);
  }
  
  static std::array<double, 3L> Bp
  (double const mu, double const sigma) noexcept {
    double const d_mu = std::exp(mu + sigma * sigma * .5);
    return { d_mu, sigma * d_mu, d_mu };
  }
  
  static double dd_eta(double const x) noexcept {
    return exp(x);
  }
};
} // namespace partition
