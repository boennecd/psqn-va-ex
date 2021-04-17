
# Example of Using the psqn Package for GVAs for GLMMs

We use the [psqn](https://github.com/boennecd/psqn) package to estimate
Gaussian variational approximations (GVAs) for generalized linear mixed
models (GLMMs) for clustered data in this repository. All the formulas
for the lower bound are shown by Ormerod and Wand (2012). The C++
implementation is at [src/glmm-gva.cpp](src/glmm-gva.cpp). While this is
a small example, it should be stressed that:

  - The psqn package is particularly useful for variational
    approximations for clustered data. This is not just limited to GVAs.
  - The code for the GLMMs shown here is easy to extend to other types
    of outcomes and link functions.

First, we source the C++ file:

``` r
# File to source the C++ file with the GVA method.
source_cpp_files <- function(){
  library(Rcpp)
  sourceCpp(file.path("src", "glmm-gva.cpp"), embeddedR = FALSE)
}
source_cpp_files()
```

Then, we can start to run the examples.

## Univariate Random Effects

We start with univariate random effects. Here we can compare with the
adaptive Gauss-Hermite quadrature (AGHQ) implementation and the Laplace
implementation in the `lme4` package. First, we assign the seeds we will
use.

``` r
# the seeds we will use
seeds <- c(45093216L, 6708209L, 22871506L, 48729709L, 13815212L, 2445671L, 
  99644356L, 27804863L, 58720645L, 48698339L, 72800773L, 69728613L, 
  19695521L, 13426623L, 64035243L, 56568280L, 89030658L, 49793712L, 
  2894709L, 48970384L, 24040262L, 61005125L, 23185610L, 68597651L, 
  78549111L, 89299120L, 46807969L, 22263937L, 34217798L, 85485436L, 
  7908766L, 44965105L, 23503235L, 83798744L, 8458798L, 38496678L, 
  88280479L, 82780490L, 27180392L, 23062594L, 79088487L, 17579394L, 
  46336937L, 56713632L, 85726923L, 54222363L, 10841571L, 12920866L, 
  92517560L, 60918703L, 76145043L, 71814379L, 22910960L, 50168947L, 
  81727916L, 41688664L, 88884353L, 46266692L, 81146226L, 15193026L, 
  47061382L, 31518192L, 18851631L, 92243664L, 79531365L, 90443602L, 
  67905541L, 79594099L, 94216590L, 69815273L, 23773183L, 6138263L, 
  1026577L, 10150999L, 30310841L, 32496506L, 65580268L, 33782152L, 
  61816347L, 49119238L, 92337927L, 71432485L, 48391558L, 13332096L, 
  31349064L, 5859715L, 76292493L, 15131078L, 79902880L, 30472712L, 
  7443312L, 20150770L, 29484917L, 93109461L, 3083530L, 62986851L, 
  69007023L, 23760608L, 24451838L, 91022614L)
```

Next we assign the functions we will need to perform the simulation
study (feel free to skip this).

``` r
# simulates from a GLMM.
# 
# Args: 
#   n_cluster: number of clusters
#   cor_mat: the correlation/scale matrix for the random effects.
#   beta: the fixed effect coefficient vector.
#   sig: scale parameter for the covariance matrix.
#   n_obs: number of members in each cluster.
#   get_x: function to get the fixed effect covariate matrix.
#   get_z: function to get the random effect covariate matrix.
#   model: characther with model and link function.
sim_dat <- function(n_cluster = 100L, cor_mat, beta, sig, n_obs = 10L, 
                    get_x, get_z, model = "binomial_logit"){
  vcov_mat <- sig * cor_mat # the covariance matrix
  
  # simulate the clusters
  group <- 0L
  out <- replicate(n_cluster, {
    # the random effects
    u <- drop(rnorm(NCOL(vcov_mat)) %*% chol(vcov_mat))
    
    # design matrices
    X <- t(get_x(1:n_obs))
    Z <- t(get_z(1:n_obs))
    
    # linear predictors
    eta <- drop(beta %*% X +  u %*% Z)
    
    # the outcome
    group <<- group + 1L
    if(model == "binomial_logit"){
      prob <- 1/(1 + exp(-eta))
      nis <- sample.int(5L, n_obs, replace = TRUE)
      y <- rbinom(n_obs, nis, prob)
      nis <- as.numeric(nis)
      y <- as.numeric(y)
      
      
    } else if(model == "Poisson_log"){
      y <- rpois(length(eta), exp(eta))
      nis <- rep(1, length(y))
      
    } else
      stop("model not implemented")
    
    # return 
    list(y = y, group = rep(group, n_obs), 
         nis = nis, X = X, Z = Z, model = model)
  }, simplify = FALSE)
  
  # create a data.frame with the data set and return 
  df <- lapply(out, function(x){
    x$X <- t(x$X)
    x$Z <- t(x$Z)
    x$Z <- x$Z[, setdiff(colnames(x$Z), colnames(x$X))]
    data.frame(x)
  })
  df <- do.call(rbind, df)
  
  list(df_dat = df, list_dat = out, 
       beta = beta, cor_mat = cor_mat, stdev = sqrt(sig))   
}

# estimates the model using a Laplace approximation or adaptive 
# Gauss-Hermite quadrature. 
# 
# Args: 
#   dat: data.frame with the data.
#   formula: formula to pass to glmer. 
#   nAGQ: nAGQ argument to pass to glmer.
#   family: family to use.
est_lme4 <- function(dat, formula, nAGQ = 1L, family){
  library(lme4)
  fit <- glmer(
    formula = formula, dat, family = family, nAGQ = nAGQ, weights = nis, 
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun=2e5)))
  vc <- VarCorr(fit)
  list(ll = c(logLik(fit)), fixef = fixef(fit), 
       stdev = attr(vc$group, "stddev"), 
       cor_mat = attr(vc$group, "correlation"), 
       conv = fit@optinfo$conv$opt)
}

# estimates the model using a GVA. 
# 
# Args: 
#   dat: list with a list for each cluster. 
#   rel_eps: relative convergence threshold.
#   est_psqn: logical for whether to use psqn.
#   model: characther with model and link function.
#   family: family to pass to glm.
est_va <- function(dat, rel_eps = 1e-8, est_psqn, n_threads = 1L, 
                   model, formula, family){
  func <- get_lb_optimizer(dat$list_dat, n_threads)
  
  # setup stating values
  n_clust <- length(dat$list_dat)
  n_rng <- NROW(dat$list_dat[[1L]]$Z)
  n_fix <- NROW(dat$list_dat[[1L]]$X)
  
  par <- numeric(n_fix + n_clust * n_rng +
                   (n_clust + 1) * n_rng * (n_rng + 1L) / 2)
  
  # estimate starting values for the fixed effects w/ a GLM
  if(n_fix > 0)
    par[1:n_fix] <- with(dat$df_dat, glm.fit(
      x = dat$df_dat[, grepl("^X", colnames(dat$df_dat))], 
      y = y / nis, weights = nis, family = family))[[
        "coefficients"]]
  
  par <- drop(get_start_vals(func, par = par, Sigma = diag(n_rng), 
                             n_beta = n_fix))
  
  if(est_psqn){
    # use the psqn package
    res <- opt_lb(val = par, ptr = func, rel_eps = rel_eps, max_it = 1000L, 
                  n_threads = n_threads, c1 = 1e-4, c2 = .9, cg_tol = .2, 
                  max_cg = max(2L, as.integer(log(n_clust) * 10)))
    conv <- !res$convergence
    
  } else {
    # use a limited-memory BFGS implementation 
    library(lbfgs)
    fn <- function(x)
      eval_lb   (x, ptr = func, n_threads = n_threads)
    gr <- function(x)
      eval_lb_gr(x, ptr = func, n_threads = n_threads)
    res <- lbfgs(fn, gr, par, invisible = 1)
    conv <- res$convergence
  }
  
  mod_par <- head(res$par, n_fix + n_rng * (n_rng + 1) / 2)
  Sig_hat <- get_pd_mat(tail(mod_par, -n_fix), n_rng)[[1L]]
  
  list(lb = -res$value, fixef = head(mod_par, n_fix), 
       stdev = sqrt(diag(Sig_hat)), cor_mat = cov2cor(Sig_hat), 
       conv = conv)
}
```

Then we perform the simulation study.

``` r
# assign covariate functions
get_x <- function(x)
  cbind(`(Intercept)` = 1, `cont` = scale(x), `dummy` = x %% 2)
get_z <- function(x)
  cbind(`(Intercept)` = rep(1, length(x)))

# performs the simulation study. 
# 
# Args: 
#   n_cluster: number of clusters.
#   seeds_use: seeds to use in the simulations.
#   n_obs: number of members in each cluster.
#   sig: scale parameter for the covariance matrix.
#   cor_mat: the correlation matrix.
#   beta: the fixed effect coefficient vector.
#   prefix: prefix for saved files.
#   formula: formula for lme4. 
#   model: characther with model and link function.
run_study <- function(n_cluster, seeds_use, n_obs = 3L, sig = 1.5, 
                      cor_mat = diag(1), beta = c(-2, -1, 1), 
                      prefix = "univariate", 
                      formula = y / nis ~ X.V2 + X.dummy + (1 | group), 
                      model = "binomial_logit")
  lapply(seeds_use, function(s){
    f <- file.path("cache", sprintf("%s-%d-%d.RDS", prefix, n_cluster, s))
    if(!file.exists(f)){
      # simulate the data set
      set.seed(s)
      dat <- sim_dat(
        n_cluster = n_cluster, cor_mat = cor_mat, beta = beta, 
        sig = sig, n_obs = n_obs, get_x = get_x, get_z = get_z, 
        model = model)
      
      # fit the model
      fam <- switch (model,
        binomial_logit = binomial(), 
        Poisson_log = poisson(), 
        stop("model not implemented"))
      lap_time <- system.time(lap_fit <- est_lme4(
        formula = formula, family = fam,
        dat = dat$df_dat, nAGQ = 1L))
      if(NCOL(cor_mat) > 1){
        agh_time <- NULL
        agh_fit <- NULL
      } else 
        agh_time <- system.time(agh_fit <- est_lme4(
          formula = formula, family = fam,
          dat = dat$df_dat, nAGQ = 25L))
      gva_time <- system.time(
        gva_fit <- est_va(dat, est_psqn = TRUE, n_threads = 1L, 
                          model = model, family = fam))
      gva_time_four <- system.time(
        gva_fit_four <- est_va(dat, est_psqn = TRUE, n_threads = 4L, 
                               model = model, family = fam))
      gva_lbfgs_time <- system.time(
        gva_lbfgs_fit <- est_va(dat, est_psqn = FALSE, n_threads = 1L, 
                                model = model, family = fam))
      
      # extract the bias, the computation time, and the lower bound
      # or log likelihood and return
      get_bias <- function(x){
        if(is.null(x))
          list(fixed = NULL, 
               stdev = NULL, 
               cor_mat = NULL)
        else
          list(fixed = x$fixef - dat$beta, 
               stdev = x$stdev - dat$stdev,
               cor_mat = (x$cor_mat - dat$cor_mat)[lower.tri(dat$cor_mat)])
      }
      
      bias <- mapply(
        rbind, 
        Laplace = get_bias(lap_fit), 
        AGHQ = get_bias(agh_fit), GVA = get_bias(gva_fit), 
        `GVA (4 threads)` = get_bias(gva_fit_four),
        `GVA LBFGS` = get_bias(gva_lbfgs_fit), 
        SIMPLIFY = FALSE)
      
      tis <- rbind(Laplace = lap_time, AGHQ = agh_time, GVA = gva_time, 
                   `GVA (4 threads)` = gva_time_four,
                   `GVA LBFGS` = gva_lbfgs_time)[, 1:3]
      
      conv <- unlist(sapply(
        list(Laplace = lap_fit, AGHQ = agh_fit, GVA = gva_fit, 
             `GVA (4 threads)` = gva_fit_four, `GVA LBFGS` = gva_lbfgs_fit), 
        `[[`, "conv"))
      
      . <- function(x)
        if(is.null(x)) NULL else x$ll
      out <- list(bias = bias, time = tis, 
                  lls = c(Laplace = .(lap_fit), AGHQ = .(agh_fit), 
                          GVA = gva_fit$lb, 
                          `GVA (4 threads)` = gva_fit_four$lb, 
                          `GVA LBFGS` = gva_lbfgs_fit$lb), conv = conv,
                  seed = s)
      
      message(paste0(capture.output(out), collapse = "\n"))
      message("")
      saveRDS(out, f)
    }
    
    readRDS(f)
  })

# use the function to perform the simulation study
sim_res_uni <- run_study(1000L, seeds)
```

The bias estimates are given below. There are three GVA rows: the `GVA`
row is using the psqn package with one thread, the `GVA (4 threads)` row
is using the psqn package with four threads, and the `GVA LBFGS` row is
using a limited-memory BFGS implementation.

``` r
# function to compute the bias estimates and the standard errors.
comp_bias <- function(results, what){
  errs <- sapply(results, function(x) x$bias[[what]], 
                 simplify = "array")
  any_non_finite <- apply(!is.finite(errs), 3, any)
  if(any(any_non_finite)){
    cat(sprintf("Removing %d non-finite estimates cases\n", 
                sum(any_non_finite)))
    errs <- errs[, , !any_non_finite]
  }
  
  ests <- apply(errs, 1:2, mean)
  SE <- apply(errs, 1:2, sd) / sqrt(dim(errs)[[3]])
  list(bias = ests, `Standard error` = SE)
}

# bias estimates for the fixed effect 
comp_bias(sim_res_uni, "fixed")
```

    ## $bias
    ##                 (Intercept)     X.V2   X.dummy
    ## Laplace            0.005726 0.004543 -0.005410
    ## AGHQ               0.000219 0.003648 -0.004550
    ## GVA                0.005817 0.004497 -0.005416
    ## GVA (4 threads)    0.005817 0.004497 -0.005416
    ## GVA LBFGS          0.005705 0.004474 -0.005367
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.006462 0.003665 0.006755
    ## AGHQ               0.006512 0.003675 0.006788
    ## GVA                0.006478 0.003672 0.006788
    ## GVA (4 threads)    0.006478 0.003672 0.006788
    ## GVA LBFGS          0.006478 0.003672 0.006787

``` r
# bias estimates for the random effect standard deviation
comp_bias(sim_res_uni, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.039097
    ## AGHQ              -0.007815
    ## GVA               -0.022260
    ## GVA (4 threads)   -0.022260
    ## GVA LBFGS         -0.022144
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace            0.005343
    ## AGHQ               0.005502
    ## GVA                0.005347
    ## GVA (4 threads)    0.005347
    ## GVA LBFGS          0.005346

Summary stats for the computation time are given below:

``` r
# computes summary statistics for the computation time.
time_stats <- function(results){
  tis <- sapply(results, `[[`, "time", simplify = "array")
  tis <- tis[, "elapsed", ]
  
  # remove non-finite cases
  errs <- sapply(results, function(x) x$bias[["fixed"]], 
                 simplify = "array")
  any_non_finite <- apply(!is.finite(errs), 3, any)
  if(any(any_non_finite)){
    cat(sprintf("Removing %d non-finite estimates cases\n", 
                sum(any_non_finite)))
    tis <- tis[, !any_non_finite]
  }
  
  cbind(mean    = apply(tis, 1, mean), 
        meadian = apply(tis, 1, median))
}

# use the function
time_stats(sim_res_uni)
```

    ##                    mean meadian
    ## Laplace         0.39918  0.3965
    ## AGHQ            1.01240  0.9950
    ## GVA             0.23631  0.2395
    ## GVA (4 threads) 0.06832  0.0680
    ## GVA LBFGS       2.44273  2.4175

### Poisson Model

We re-run the simulation study below with a Poisson model instead.

``` r
sim_res_uni_poisson <- run_study(
  1000L, seeds, model = "Poisson_log", prefix = "Poisson", 
  formula = y ~ X.V2 + X.dummy + (1 | group))
```

Here are the results:

``` r
# bias of the fixed effect 
comp_bias(sim_res_uni_poisson, "fixed")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace          -0.0006674 0.002588 0.001696
    ## AGHQ             -0.0013045 0.002588 0.001696
    ## GVA               0.0183506 0.002588 0.001662
    ## GVA (4 threads)   0.0183506 0.002588 0.001662
    ## GVA LBFGS         0.0182736 0.002588 0.001696
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.007655 0.002972 0.006768
    ## AGHQ               0.007624 0.002972 0.006768
    ## GVA                0.007520 0.002972 0.006768
    ## GVA (4 threads)    0.007520 0.002972 0.006768
    ## GVA LBFGS          0.007520 0.002972 0.006768

``` r
# bias of the random effect standard deviation
comp_bias(sim_res_uni_poisson, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.020828
    ## AGHQ              -0.001322
    ## GVA               -0.041989
    ## GVA (4 threads)   -0.041989
    ## GVA LBFGS         -0.041973
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace            0.004558
    ## AGHQ               0.004661
    ## GVA                0.004430
    ## GVA (4 threads)    0.004430
    ## GVA LBFGS          0.004430

``` r
# computation time summary statistics 
time_stats(sim_res_uni_poisson)
```

    ##                    mean meadian
    ## Laplace         0.32536   0.324
    ## AGHQ            0.72737   0.725
    ## GVA             0.03169   0.032
    ## GVA (4 threads) 0.01515   0.015
    ## GVA LBFGS       0.29941   0.302

### Larger Sample

We re-run the simulation study below with more clusters.

``` r
sim_res_uni_large <- run_study(5000L, seeds)
```

Here are the results:

``` r
# bias of the fixed effect 
comp_bias(sim_res_uni_large, "fixed")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.002266 0.001883 0.001770
    ## AGHQ              -0.003234 0.001022 0.002620
    ## GVA                0.002373 0.001882 0.001811
    ## GVA (4 threads)    0.002373 0.001882 0.001811
    ## GVA LBFGS          0.002246 0.001852 0.001808
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.003335 0.001520 0.003239
    ## AGHQ               0.003358 0.001526 0.003254
    ## GVA                0.003346 0.001524 0.003252
    ## GVA (4 threads)    0.003346 0.001524 0.003252
    ## GVA LBFGS          0.003344 0.001524 0.003253

``` r
# bias of the random effect standard deviation
comp_bias(sim_res_uni_large, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.033339
    ## AGHQ              -0.002072
    ## GVA               -0.016519
    ## GVA (4 threads)   -0.016519
    ## GVA LBFGS         -0.016391
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace            0.002298
    ## AGHQ               0.002361
    ## GVA                0.002298
    ## GVA (4 threads)    0.002298
    ## GVA LBFGS          0.002299

``` r
# computation time summary statistics 
time_stats(sim_res_uni_large)
```

    ##                    mean meadian
    ## Laplace          1.8577   1.845
    ## AGHQ             4.9224   4.920
    ## GVA              1.1580   1.182
    ## GVA (4 threads)  0.3296   0.336
    ## GVA LBFGS       20.9588  20.809

## 3D Random Effects

We run a simulation study in this section with three random effects per
cluster. We use the same function as before to perform the simulation
study.

``` r
get_z <- get_x # random effect covariates are the same as the fixed

cor_mat <- matrix(c(1, -.25, .25, -.25, 1, 0, .25, 0, 1), 3L)
sim_res_mult <- run_study(
  n_cluster = 1000L, seeds_use = seeds, sig = .8, n_obs = 10L, 
  cor_mat = cor_mat, prefix = "multivariate", 
  formula = y / nis ~ X.V2 + X.dummy + (1 + X.V2 + X.dummy | group))
```

We show the bias estimates and summary statistics for the computation
time below.

``` r
# bias of the fixed effect 
comp_bias(sim_res_mult, "fixed")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace           -0.004472 0.002628 0.004506
    ## GVA               -0.001676 0.004653 0.004981
    ## GVA (4 threads)   -0.001678 0.004655 0.004975
    ## GVA LBFGS         -0.001672 0.004651 0.004986
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.003522 0.003919 0.004450
    ## GVA                0.003497 0.003901 0.004427
    ## GVA (4 threads)    0.003497 0.003902 0.004427
    ## GVA LBFGS          0.003495 0.003902 0.004427

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_mult, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace           -0.012482 -0.01631 -0.03593
    ## GVA               -0.004353 -0.01393 -0.01912
    ## GVA (4 threads)   -0.004356 -0.01393 -0.01911
    ## GVA LBFGS         -0.004439 -0.01397 -0.01964
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            0.003908 0.003236 0.005514
    ## GVA                0.003938 0.003203 0.005418
    ## GVA (4 threads)    0.003938 0.003203 0.005419
    ## GVA LBFGS          0.003940 0.003204 0.005419

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_mult, "cor_mat")
```

    ## $bias
    ##                      [,1]     [,2]     [,3]
    ## Laplace         -0.001307 0.022336 0.003455
    ## GVA             -0.006249 0.001460 0.008775
    ## GVA (4 threads) -0.006246 0.001483 0.008782
    ## GVA LBFGS       -0.006094 0.002230 0.008736
    ## 
    ## $`Standard error`
    ##                     [,1]     [,2]     [,3]
    ## Laplace         0.005582 0.008862 0.007487
    ## GVA             0.005520 0.008642 0.007223
    ## GVA (4 threads) 0.005523 0.008649 0.007222
    ## GVA LBFGS       0.005535 0.008641 0.007237

``` r
# computation time summary statistics 
time_stats(sim_res_mult)
```

    ##                    mean meadian
    ## Laplace          5.1964  5.2125
    ## GVA              2.1419  2.0905
    ## GVA (4 threads)  0.6011  0.5835
    ## GVA LBFGS       25.1598 24.8245

## 6D Random Effects

We run a simulation study in this section with six random effects per
cluster.

``` r
# setup for the simulation study
cor_mat <- diag(6)

get_z <- get_x <- function(x){
  n <- length(x)
  out <- cbind(1, matrix(rnorm(n * 5), n))
  colnames(out) <- c("(Intercept)", paste0("X", 1:5))
  out
}

# run the study
sim_res_6D <- run_study(
  n_cluster = 1000L, seeds_use = seeds, sig = 1/6, 
  n_obs = 10L, cor_mat = cor_mat, prefix = "6D", 
  beta = c(-2, rep(1/sqrt(5), 5)),
  formula = y / nis ~ X.X1 + X.X2 + X.X3 + X.X4 + X.X5 + 
    (1 + X.X1 + X.X2 + X.X3 + X.X4 + X.X5 | group))
```

We show the bias estimates and summary statistics for the computation
time below.

``` r
# bias of the fixed effect 
comp_bias(sim_res_6D, "fixed")
```

    ## $bias
    ##                 (Intercept)       X.X1       X.X2        X.X3       X.X4
    ## Laplace            0.147065 -0.0246307 -0.0256593 -0.02484397 -0.0257816
    ## GVA                0.008193 -0.0004003  0.0002840  0.00009311  0.0009278
    ## GVA (4 threads)    0.008193 -0.0003991  0.0002838  0.00009379  0.0009278
    ## GVA LBFGS          0.008194 -0.0003997  0.0002823  0.00009396  0.0009255
    ##                       X.X5
    ## Laplace         -0.0232167
    ## GVA              0.0006218
    ## GVA (4 threads)  0.0006224
    ## GVA LBFGS        0.0006208
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.002631 0.001966 0.002272 0.002147 0.002014 0.001944
    ## GVA                0.002691 0.001867 0.002188 0.002144 0.001891 0.001786
    ## GVA (4 threads)    0.002690 0.001867 0.002187 0.002144 0.001891 0.001786
    ## GVA LBFGS          0.002690 0.001867 0.002188 0.002144 0.001891 0.001786

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D, "stdev")
```

    ## $bias
    ##                 (Intercept)      X.X1      X.X2      X.X3      X.X4     X.X5
    ## Laplace             0.05582 -0.193100 -0.190815 -0.192642 -0.188739 -0.18643
    ## GVA                -0.01108 -0.009973 -0.009962 -0.009398 -0.002402 -0.01113
    ## GVA (4 threads)    -0.01109 -0.009975 -0.009960 -0.009403 -0.002402 -0.01112
    ## GVA LBFGS          -0.01101 -0.009944 -0.009996 -0.009431 -0.002418 -0.01118
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.002791 0.007034 0.004893 0.005488 0.004325 0.004041
    ## GVA                0.002945 0.002560 0.003003 0.003152 0.003311 0.003174
    ## GVA (4 threads)    0.002945 0.002560 0.003004 0.003153 0.003310 0.003174
    ## GVA LBFGS          0.002954 0.002556 0.003007 0.003157 0.003314 0.003180

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D, "cor_mat")
```

    ## $bias
    ##                      [,1]     [,2]     [,3]      [,4]     [,5]      [,6]
    ## Laplace         -0.101933 -0.09854 -0.07963 -0.099039 -0.12207 -0.039944
    ## GVA             -0.002219 -0.01346  0.01078 -0.007057 -0.02337  0.001600
    ## GVA (4 threads) -0.002215 -0.01345  0.01077 -0.007062 -0.02337  0.001595
    ## GVA LBFGS       -0.002195 -0.01345  0.01076 -0.007110 -0.02333  0.001559
    ##                      [,7]    [,8]    [,9]    [,10]   [,11]    [,12]     [,13]
    ## Laplace          0.040542 0.04409 0.02199 0.023684 0.03180 0.032495  0.068268
    ## GVA             -0.005542 0.02713 0.01576 0.004724 0.01165 0.005980 -0.007329
    ## GVA (4 threads) -0.005556 0.02711 0.01577 0.004730 0.01167 0.005989 -0.007334
    ## GVA LBFGS       -0.005599 0.02714 0.01577 0.004735 0.01167 0.005939 -0.007314
    ##                    [,14]    [,15]
    ## Laplace         0.024040  0.03989
    ## GVA             0.007395 -0.01399
    ## GVA (4 threads) 0.007406 -0.01399
    ## GVA LBFGS       0.007410 -0.01402
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.03443 0.01808 0.01712 0.01676 0.01584 0.04076 0.03194 0.02615
    ## GVA             0.01211 0.01103 0.01106 0.01047 0.01190 0.01265 0.01299 0.01050
    ## GVA (4 threads) 0.01211 0.01103 0.01106 0.01047 0.01189 0.01265 0.01299 0.01050
    ## GVA LBFGS       0.01211 0.01103 0.01106 0.01047 0.01190 0.01266 0.01301 0.01050
    ##                    [,9]   [,10]   [,11]   [,12]   [,13]   [,14]   [,15]
    ## Laplace         0.02161 0.03730 0.02923 0.02442 0.02862 0.02235 0.02556
    ## GVA             0.01192 0.01229 0.01082 0.01006 0.01076 0.01189 0.01001
    ## GVA (4 threads) 0.01192 0.01230 0.01082 0.01006 0.01076 0.01188 0.01001
    ## GVA LBFGS       0.01192 0.01230 0.01083 0.01008 0.01076 0.01190 0.01001

``` r
# computation time summary statistics 
time_stats(sim_res_6D)
```

    ##                   mean meadian
    ## Laplace         88.664  75.874
    ## GVA              4.016   3.958
    ## GVA (4 threads)  1.280   1.214
    ## GVA LBFGS       49.652  47.704

## 6D Random Effects Poisson

We run a simulation study in this section with six random effects per
cluster using a Poisson model.

``` r
# run the study
sim_res_6D_pois <- run_study(
  n_cluster = 1000L, seeds_use = seeds, sig = 1/6, 
  n_obs = 10L, cor_mat = cor_mat, prefix = "Poisson_6D", 
  beta = c(-2, rep(1/sqrt(5), 5)), model = "Poisson_log",
  formula = y / nis ~ X.X1 + X.X2 + X.X3 + X.X4 + X.X5 + 
    (1 + X.X1 + X.X2 + X.X3 + X.X4 + X.X5 | group))
```

We show the bias estimates and summary statistics for the computation
time below.

``` r
# bias of the fixed effect 
comp_bias(sim_res_6D_pois, "fixed")
```

    ## Removing 1 non-finite estimates cases

    ## $bias
    ##                 (Intercept)      X.X1         X.X2      X.X3      X.X4
    ## Laplace             0.09932 -0.002963 -0.003861342 -0.009347 0.0002625
    ## GVA                 0.02912 -0.002551 -0.000016549 -0.006778 0.0006123
    ## GVA (4 threads)     0.02911 -0.002551 -0.000017059 -0.006776 0.0006147
    ## GVA LBFGS           0.02912 -0.002552 -0.000009826 -0.006777 0.0006124
    ##                      X.X5
    ## Laplace         -0.009825
    ## GVA             -0.005376
    ## GVA (4 threads) -0.005376
    ## GVA LBFGS       -0.005377
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.004837 0.002826 0.002899 0.002659 0.002301 0.002630
    ## GVA                0.003764 0.002180 0.002088 0.002125 0.001900 0.002036
    ## GVA (4 threads)    0.003763 0.002180 0.002087 0.002124 0.001899 0.002036
    ## GVA LBFGS          0.003764 0.002180 0.002088 0.002125 0.001900 0.002036

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D_pois, "stdev")
```

    ## Removing 1 non-finite estimates cases

    ## $bias
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace             0.18170 -0.05789 -0.05284 -0.05405 -0.05230 -0.04906
    ## GVA                -0.02153 -0.02016 -0.01675 -0.02056 -0.01679 -0.01804
    ## GVA (4 threads)    -0.02152 -0.02017 -0.01675 -0.02056 -0.01679 -0.01804
    ## GVA LBFGS          -0.02162 -0.02018 -0.01674 -0.02051 -0.01679 -0.01801
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.003818 0.005115 0.004505 0.004894 0.004082 0.004290
    ## GVA                0.003678 0.003092 0.002907 0.002918 0.003008 0.002801
    ## GVA (4 threads)    0.003678 0.003091 0.002907 0.002918 0.003008 0.002801
    ## GVA LBFGS          0.003693 0.003087 0.002907 0.002919 0.003011 0.002800

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D_pois, "cor_mat")
```

    ## Removing 1 non-finite estimates cases

    ## $bias
    ##                     [,1]      [,2]     [,3]     [,4]      [,5]          [,6]
    ## Laplace         -0.12005 -0.120031 -0.12613 -0.14214 -0.110762  0.0519871131
    ## GVA              0.01289 -0.008884 -0.01209 -0.01004 -0.001190 -0.0000008717
    ## GVA (4 threads)  0.01293 -0.008899 -0.01213 -0.01004 -0.001229 -0.0000126585
    ## GVA LBFGS        0.01292 -0.008867 -0.01210 -0.01011 -0.001247 -0.0000426178
    ##                      [,7]    [,8]    [,9]   [,10]     [,11]    [,12]    [,13]
    ## Laplace          0.035034 0.03443 0.04841 0.02950  0.032171  0.02655  0.01833
    ## GVA             -0.006746 0.02073 0.01329 0.02612 -0.001874 -0.01423 -0.02220
    ## GVA (4 threads) -0.006744 0.02073 0.01328 0.02615 -0.001879 -0.01422 -0.02223
    ## GVA LBFGS       -0.006788 0.02074 0.01328 0.02608 -0.001903 -0.01418 -0.02221
    ##                     [,14]    [,15]
    ## Laplace          0.045087 0.043516
    ## GVA             -0.009723 0.008918
    ## GVA (4 threads) -0.009718 0.008907
    ## GVA LBFGS       -0.009774 0.008918
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.01639 0.01202 0.01223 0.01195 0.01199 0.01645 0.01809 0.01564
    ## GVA             0.01190 0.01142 0.01139 0.01238 0.01116 0.01035 0.01135 0.01093
    ## GVA (4 threads) 0.01190 0.01142 0.01139 0.01238 0.01115 0.01036 0.01135 0.01093
    ## GVA LBFGS       0.01191 0.01144 0.01139 0.01240 0.01115 0.01035 0.01134 0.01093
    ##                    [,9]   [,10]   [,11]   [,12]    [,13]   [,14]   [,15]
    ## Laplace         0.01421 0.01463 0.01669 0.01484 0.014499 0.01359 0.01642
    ## GVA             0.01035 0.01098 0.01149 0.01068 0.009794 0.01152 0.01160
    ## GVA (4 threads) 0.01035 0.01098 0.01149 0.01067 0.009790 0.01152 0.01160
    ## GVA LBFGS       0.01035 0.01098 0.01150 0.01067 0.009787 0.01152 0.01161

``` r
# computation time summary statistics 
time_stats(sim_res_6D_pois)
```

    ## Removing 1 non-finite estimates cases

    ##                   mean meadian
    ## Laplace         56.209  53.272
    ## GVA              4.182   4.066
    ## GVA (4 threads)  1.443   1.363
    ## GVA LBFGS       11.398  10.972

## References

<div id="refs" class="references">

<div id="ref-Ormerod11">

Ormerod, J. T., and M. P. Wand. 2012. “Gaussian Variational Approximate
Inference for Generalized Linear Mixed Models.” *Journal of
Computational and Graphical Statistics* 21 (1): 2–17.
<https://doi.org/10.1198/jcgs.2011.09118>.

</div>

</div>
