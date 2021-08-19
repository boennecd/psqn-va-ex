
# Example of Using the psqn Package for GVAs for GLMMs

We use the [psqn](https://github.com/boennecd/psqn) package to estimate
Gaussian variational approximations (GVAs) for generalized linear mixed
models (GLMMs) for clustered data in this repository. All the formulas
for the lower bound are shown by Ormerod and Wand (2012). The C++
implementation is at [src/glmm-gva.cpp](src/glmm-gva.cpp). While this is
a small example, it should be stressed that:

-   The psqn package is particularly useful for variational
    approximations for clustered data. This is not just limited to GVAs.
-   The code for the GLMMs shown here is easy to extend to other types
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

    ## Registered S3 methods overwritten by 'RcppEigen':
    ##   method               from         
    ##   predict.fastLm       RcppArmadillo
    ##   print.fastLm         RcppArmadillo
    ##   summary.fastLm       RcppArmadillo
    ##   print.summary.fastLm RcppArmadillo

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
seeds <- head(seeds, 5)
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
#   model: character with model and link function.
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
    ##                 (Intercept)      X.V2   X.dummy
    ## Laplace             0.02203 -0.003205 0.0009942
    ## AGHQ                0.01686 -0.003957 0.0015530
    ## GVA                 0.02231 -0.003129 0.0007811
    ## GVA (4 threads)     0.02231 -0.003129 0.0007811
    ## GVA LBFGS           0.02213 -0.003166 0.0008145
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2  X.dummy
    ## Laplace             0.01306 0.01729 0.008903
    ## AGHQ                0.01325 0.01732 0.008878
    ## GVA                 0.01299 0.01727 0.008881
    ## GVA (4 threads)     0.01299 0.01727 0.008881
    ## GVA LBFGS           0.01300 0.01729 0.008871

``` r
# bias estimates for the random effect standard deviation
comp_bias(sim_res_uni, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.035015
    ## AGHQ              -0.003727
    ## GVA               -0.018051
    ## GVA (4 threads)   -0.018051
    ## GVA LBFGS         -0.017897
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace             0.02402
    ## AGHQ                0.02478
    ## GVA                 0.02405
    ## GVA (4 threads)     0.02405
    ## GVA LBFGS           0.02405

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

    ##                   mean meadian
    ## Laplace         0.3390   0.322
    ## AGHQ            0.6844   0.673
    ## GVA             0.1154   0.112
    ## GVA (4 threads) 0.0354   0.035
    ## GVA LBFGS       1.0378   1.051

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
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace           -0.002972 0.01136 0.03283
    ## AGHQ              -0.003656 0.01136 0.03283
    ## GVA                0.015696 0.01135 0.03278
    ## GVA (4 threads)    0.015696 0.01135 0.03278
    ## GVA LBFGS          0.015666 0.01136 0.03283
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace             0.05881 0.01698 0.04813
    ## AGHQ                0.05856 0.01698 0.04813
    ## GVA                 0.05760 0.01698 0.04809
    ## GVA (4 threads)     0.05760 0.01698 0.04809
    ## GVA LBFGS           0.05760 0.01698 0.04813

``` r
# bias of the random effect standard deviation
comp_bias(sim_res_uni_poisson, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.014581
    ## AGHQ               0.005097
    ## GVA               -0.035202
    ## GVA (4 threads)   -0.035202
    ## GVA LBFGS         -0.035210
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace             0.02205
    ## AGHQ                0.02248
    ## GVA                 0.02086
    ## GVA (4 threads)     0.02086
    ## GVA LBFGS           0.02083

``` r
# computation time summary statistics 
time_stats(sim_res_uni_poisson)
```

    ##                   mean meadian
    ## Laplace         0.2432   0.243
    ## AGHQ            0.4850   0.483
    ## GVA             0.0258   0.025
    ## GVA (4 threads) 0.0132   0.013
    ## GVA LBFGS       0.2258   0.219

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
    ## Laplace           -0.010009 0.002076 0.003333
    ## AGHQ              -0.015596 0.001187 0.004284
    ## GVA               -0.009918 0.002036 0.003460
    ## GVA (4 threads)   -0.009918 0.002036 0.003460
    ## GVA LBFGS         -0.009986 0.002003 0.003449
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.V2 X.dummy
    ## Laplace             0.01975 0.007790 0.01997
    ## AGHQ                0.01993 0.007853 0.02014
    ## GVA                 0.02000 0.007845 0.02008
    ## GVA (4 threads)     0.02000 0.007845 0.02008
    ## GVA LBFGS           0.02000 0.007834 0.02011

``` r
# bias of the random effect standard deviation
comp_bias(sim_res_uni_large, "stdev")
```

    ## $bias
    ##                 (Intercept)
    ## Laplace           -0.022636
    ## AGHQ               0.008949
    ## GVA               -0.005867
    ## GVA (4 threads)   -0.005867
    ## GVA LBFGS         -0.005735
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace             0.01211
    ## AGHQ                0.01236
    ## GVA                 0.01207
    ## GVA (4 threads)     0.01207
    ## GVA LBFGS           0.01207

``` r
# computation time summary statistics 
time_stats(sim_res_uni_large)
```

    ##                   mean meadian
    ## Laplace         1.4128   1.322
    ## AGHQ            3.3768   3.557
    ## GVA             0.5682   0.571
    ## GVA (4 threads) 0.1710   0.174
    ## GVA LBFGS       9.4546   9.366

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
    ##                 (Intercept)       X.V2    X.dummy
    ## Laplace             0.01317 -0.0026339 -0.0002608
    ## GVA                 0.01572 -0.0007145  0.0001545
    ## GVA (4 threads)     0.01571 -0.0007245  0.0001613
    ## GVA LBFGS           0.01567 -0.0006770  0.0001898
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace            0.009606 0.01929 0.01587
    ## GVA                0.009564 0.01922 0.01583
    ## GVA (4 threads)    0.009557 0.01921 0.01581
    ## GVA LBFGS          0.009542 0.01922 0.01581

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_mult, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            -0.01990 -0.02612 -0.06963
    ## GVA                -0.01186 -0.02336 -0.05232
    ## GVA (4 threads)    -0.01183 -0.02336 -0.05227
    ## GVA LBFGS          -0.01194 -0.02348 -0.05284
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace             0.01614 0.01998 0.01278
    ## GVA                 0.01605 0.01989 0.01258
    ## GVA (4 threads)     0.01605 0.01989 0.01256
    ## GVA LBFGS           0.01604 0.01989 0.01261

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_mult, "cor_mat")
```

    ## $bias
    ##                      [,1]     [,2]      [,3]
    ## Laplace         -0.004912 0.031773 -0.008101
    ## GVA             -0.009890 0.009857 -0.002328
    ## GVA (4 threads) -0.009850 0.009723 -0.002347
    ## GVA LBFGS       -0.009667 0.010592 -0.002310
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]
    ## Laplace         0.02863 0.04578 0.03811
    ## GVA             0.02799 0.04472 0.03677
    ## GVA (4 threads) 0.02799 0.04472 0.03677
    ## GVA LBFGS       0.02796 0.04449 0.03680

``` r
# computation time summary statistics 
time_stats(sim_res_mult)
```

    ##                    mean meadian
    ## Laplace          3.9834   3.956
    ## GVA              1.2226   1.183
    ## GVA (4 threads)  0.3658   0.335
    ## GVA LBFGS       14.1466  14.239

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
    ##                 (Intercept)      X.X1      X.X2      X.X3      X.X4      X.X5
    ## Laplace             0.15035 -0.017374 -0.020926 -0.025174 -0.036874 -0.025923
    ## GVA                 0.01739  0.002951  0.007571  0.003525 -0.004839 -0.003274
    ## GVA (4 threads)     0.01738  0.002950  0.007571  0.003529 -0.004838 -0.003273
    ## GVA LBFGS           0.01740  0.002954  0.007572  0.003519 -0.004845 -0.003269
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.010237 0.008602 0.010742 0.006530 0.006154 0.007654
    ## GVA                0.008014 0.006648 0.008623 0.006736 0.004540 0.007828
    ## GVA (4 threads)    0.008013 0.006649 0.008622 0.006735 0.004540 0.007829
    ## GVA LBFGS          0.008015 0.006649 0.008625 0.006735 0.004543 0.007827

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.X1       X.X2     X.X3     X.X4      X.X5
    ## Laplace             0.04979 -0.16762 -0.1694292 -0.17262 -0.17556 -0.200872
    ## GVA                -0.02377 -0.01900 -0.0001321 -0.02261  0.01244 -0.003921
    ## GVA (4 threads)    -0.02377 -0.01898 -0.0001281 -0.02260  0.01245 -0.003933
    ## GVA LBFGS          -0.02360 -0.01889 -0.0001810 -0.02282  0.01242 -0.003978
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.X1     X.X2    X.X3    X.X4    X.X5
    ## Laplace             0.01754 0.01323 0.017497 0.02380 0.01651 0.02329
    ## GVA                 0.01822 0.01614 0.007088 0.01498 0.01086 0.01268
    ## GVA (4 threads)     0.01822 0.01614 0.007091 0.01497 0.01086 0.01268
    ## GVA LBFGS           0.01824 0.01614 0.007098 0.01506 0.01089 0.01275

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D, "cor_mat")
```

    ## $bias
    ##                     [,1]     [,2]     [,3]      [,4]     [,5]    [,6]      [,7]
    ## Laplace         -0.16939 -0.05712 -0.10296 -0.171690 -0.17276 0.02738  0.003903
    ## GVA             -0.02332  0.02221  0.02594 -0.007064  0.03357 0.05105 -0.045031
    ## GVA (4 threads) -0.02331  0.02217  0.02598 -0.007078  0.03357 0.05096 -0.045027
    ## GVA LBFGS       -0.02321  0.02209  0.02583 -0.007115  0.03395 0.05103 -0.045218
    ##                     [,8]    [,9]  [,10]    [,11]   [,12]    [,13]    [,14]
    ## Laplace         -0.12332 0.07779 0.1600 -0.16071 0.10527  0.00651  0.03554
    ## GVA              0.03226 0.12654 0.1252  0.06092 0.06327 -0.03093 -0.03651
    ## GVA (4 threads)  0.03222 0.12652 0.1252  0.06096 0.06330 -0.03093 -0.03650
    ## GVA LBFGS        0.03198 0.12647 0.1249  0.06110 0.06329 -0.03072 -0.03660
    ##                   [,15]
    ## Laplace         0.10685
    ## GVA             0.02608
    ## GVA (4 threads) 0.02609
    ## GVA LBFGS       0.02597
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.03875 0.02627 0.02559 0.06617 0.11824 0.05298 0.06157 0.13055
    ## GVA             0.05697 0.05544 0.06345 0.05000 0.07428 0.04982 0.03838 0.03405
    ## GVA (4 threads) 0.05695 0.05547 0.06347 0.05001 0.07431 0.04984 0.03838 0.03404
    ## GVA LBFGS       0.05682 0.05540 0.06343 0.04998 0.07431 0.04994 0.03845 0.03388
    ##                    [,9]   [,10]   [,11]   [,12]   [,13]   [,14]    [,15]
    ## Laplace         0.08853 0.20925 0.14214 0.05001 0.12317 0.08787 0.169925
    ## GVA             0.05589 0.05502 0.04217 0.03111 0.03058 0.04342 0.009050
    ## GVA (4 threads) 0.05586 0.05503 0.04216 0.03111 0.03055 0.04344 0.009049
    ## GVA LBFGS       0.05591 0.05489 0.04215 0.03121 0.03042 0.04358 0.009115

``` r
# computation time summary statistics 
time_stats(sim_res_6D)
```

    ##                    mean meadian
    ## Laplace         108.566  72.948
    ## GVA               2.696   2.528
    ## GVA (4 threads)   0.887   0.827
    ## GVA LBFGS        26.157  25.412

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

    ## $bias
    ##                 (Intercept)      X.X1      X.X2      X.X3      X.X4      X.X5
    ## Laplace             0.12736 -0.003914 -0.007713 -0.008166 -0.005027 -0.007599
    ## GVA                 0.04730 -0.010034 -0.002465 -0.012160 -0.002973 -0.005343
    ## GVA (4 threads)     0.04729 -0.010054 -0.002459 -0.012155 -0.002969 -0.005329
    ## GVA LBFGS           0.04732 -0.010028 -0.002463 -0.012150 -0.002982 -0.005322
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1    X.X2    X.X3     X.X4     X.X5
    ## Laplace             0.02175 0.007305 0.01359 0.01550 0.007720 0.010690
    ## GVA                 0.02403 0.005141 0.01217 0.01236 0.003961 0.006188
    ## GVA (4 threads)     0.02403 0.005136 0.01217 0.01236 0.003957 0.006181
    ## GVA LBFGS           0.02403 0.005134 0.01217 0.01235 0.003959 0.006174

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D_pois, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace             0.17804 -0.05702 -0.06687 -0.07791 -0.05547 -0.05932
    ## GVA                -0.04054 -0.02408 -0.02072 -0.01997 -0.01170 -0.01547
    ## GVA (4 threads)    -0.04046 -0.02400 -0.02078 -0.01996 -0.01174 -0.01551
    ## GVA LBFGS          -0.04098 -0.02390 -0.02056 -0.01998 -0.01170 -0.01541
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.X1     X.X2    X.X3    X.X4    X.X5
    ## Laplace             0.02287 0.01246 0.009722 0.03472 0.01526 0.01075
    ## GVA                 0.01481 0.02045 0.011258 0.01461 0.01032 0.01000
    ## GVA (4 threads)     0.01482 0.02046 0.011286 0.01465 0.01031 0.01000
    ## GVA LBFGS           0.01492 0.02034 0.011329 0.01462 0.01033 0.01002

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D_pois, "cor_mat")
```

    ## $bias
    ##                    [,1]     [,2]     [,3]     [,4]      [,5]    [,6]     [,7]
    ## Laplace         -0.1832 -0.12546 -0.16884 -0.16487 -0.134819 0.08803  0.03834
    ## GVA              0.1151 -0.01891 -0.03667  0.03124 -0.005848 0.08674 -0.03878
    ## GVA (4 threads)  0.1151 -0.01886 -0.03651  0.03125 -0.005728 0.08672 -0.03862
    ## GVA LBFGS        0.1150 -0.01923 -0.03704  0.03118 -0.005937 0.08624 -0.03877
    ##                    [,8]    [,9]   [,10]    [,11]   [,12]    [,13]    [,14]
    ## Laplace         0.16287 0.16003 0.01555  0.10090 0.03247 -0.02557  0.09248
    ## GVA             0.03898 0.01527 0.01500 -0.05351 0.04639 -0.06470 -0.03965
    ## GVA (4 threads) 0.03895 0.01539 0.01504 -0.05354 0.04645 -0.06472 -0.03963
    ## GVA LBFGS       0.03873 0.01547 0.01485 -0.05348 0.04618 -0.06446 -0.03980
    ##                    [,15]
    ## Laplace         -0.01128
    ## GVA              0.01851
    ## GVA (4 threads)  0.01862
    ## GVA LBFGS        0.01848
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.04916 0.05360 0.05012 0.04923 0.02335 0.05982 0.08840 0.07918
    ## GVA             0.02854 0.02866 0.07551 0.04224 0.03732 0.07836 0.02861 0.04792
    ## GVA (4 threads) 0.02847 0.02867 0.07544 0.04223 0.03735 0.07832 0.02860 0.04791
    ## GVA LBFGS       0.02821 0.02865 0.07557 0.04232 0.03738 0.07813 0.02872 0.04794
    ##                    [,9]   [,10]   [,11]   [,12]   [,13]    [,14]   [,15]
    ## Laplace         0.05550 0.03102 0.09468 0.08898 0.07336 0.030804 0.06422
    ## GVA             0.03133 0.05378 0.06659 0.02832 0.03810 0.006569 0.06608
    ## GVA (4 threads) 0.03126 0.05381 0.06671 0.02836 0.03813 0.006552 0.06606
    ## GVA LBFGS       0.03124 0.05386 0.06679 0.02832 0.03805 0.006596 0.06612

``` r
# computation time summary statistics 
time_stats(sim_res_6D_pois)
```

    ##                   mean meadian
    ## Laplace         50.075  48.797
    ## GVA              4.105   4.098
    ## GVA (4 threads)  1.423   1.521
    ## GVA LBFGS       10.191   9.656

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Ormerod11" class="csl-entry">

Ormerod, J. T., and M. P. Wand. 2012. “Gaussian Variational Approximate
Inference for Generalized Linear Mixed Models.” *Journal of
Computational and Graphical Statistics* 21 (1): 2–17.
<https://doi.org/10.1198/jcgs.2011.09118>.

</div>

</div>
