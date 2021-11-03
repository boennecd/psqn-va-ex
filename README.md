
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
    ## GVA LBFGS           0.02213 -0.003166 0.0008144
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
    ## Laplace         0.3292   0.313
    ## AGHQ            0.6674   0.656
    ## GVA             0.1222   0.120
    ## GVA (4 threads) 0.0352   0.035
    ## GVA LBFGS       1.1136   1.145

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
    ## GVA LBFGS          0.015665 0.01136 0.03283
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
    ## GVA LBFGS         -0.035209
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
    ## Laplace         0.2354   0.235
    ## AGHQ            0.4646   0.462
    ## GVA             0.0334   0.033
    ## GVA (4 threads) 0.0148   0.015
    ## GVA LBFGS       0.3502   0.361

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
    ## GVA LBFGS         -0.009985 0.002004 0.003448
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
    ## GVA LBFGS         -0.005738
    ## 
    ## $`Standard error`
    ##                 (Intercept)
    ## Laplace             0.01211
    ## AGHQ                0.01236
    ## GVA                 0.01207
    ## GVA (4 threads)     0.01207
    ## GVA LBFGS           0.01206

``` r
# computation time summary statistics 
time_stats(sim_res_uni_large)
```

    ##                   mean meadian
    ## Laplace         1.3476   1.275
    ## AGHQ            3.2326   3.312
    ## GVA             0.5960   0.609
    ## GVA (4 threads) 0.1696   0.174
    ## GVA LBFGS       9.5336   9.796

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
    ## GVA                 0.01571 -0.0007233  0.0001874
    ## GVA (4 threads)     0.01572 -0.0007045  0.0001573
    ## GVA LBFGS           0.01567 -0.0006771  0.0001920
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace            0.009606 0.01929 0.01587
    ## GVA                0.009564 0.01922 0.01582
    ## GVA (4 threads)    0.009550 0.01922 0.01582
    ## GVA LBFGS          0.009541 0.01922 0.01581

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_mult, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.V2  X.dummy
    ## Laplace            -0.01990 -0.02612 -0.06963
    ## GVA                -0.01183 -0.02337 -0.05238
    ## GVA (4 threads)    -0.01188 -0.02337 -0.05230
    ## GVA LBFGS          -0.01193 -0.02348 -0.05284
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.V2 X.dummy
    ## Laplace             0.01614 0.01998 0.01278
    ## GVA                 0.01605 0.01989 0.01258
    ## GVA (4 threads)     0.01604 0.01989 0.01259
    ## GVA LBFGS           0.01604 0.01990 0.01261

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_mult, "cor_mat")
```

    ## $bias
    ##                      [,1]     [,2]      [,3]
    ## Laplace         -0.004912 0.031773 -0.008101
    ## GVA             -0.009843 0.009873 -0.002351
    ## GVA (4 threads) -0.009895 0.009844 -0.002319
    ## GVA LBFGS       -0.009665 0.010581 -0.002314
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]
    ## Laplace         0.02863 0.04578 0.03811
    ## GVA             0.02798 0.04467 0.03676
    ## GVA (4 threads) 0.02800 0.04473 0.03676
    ## GVA LBFGS       0.02796 0.04450 0.03680

``` r
# computation time summary statistics 
time_stats(sim_res_mult)
```

    ##                    mean meadian
    ## Laplace          3.8034   3.792
    ## GVA              1.1514   1.074
    ## GVA (4 threads)  0.3168   0.284
    ## GVA LBFGS       12.2184  11.713

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
    ## GVA                 0.01738  0.002953  0.007574  0.003525 -0.004838 -0.003272
    ## GVA (4 threads)     0.01738  0.002954  0.007573  0.003528 -0.004836 -0.003270
    ## GVA LBFGS           0.01740  0.002954  0.007572  0.003519 -0.004845 -0.003269
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace            0.010237 0.008602 0.010742 0.006530 0.006154 0.007654
    ## GVA                0.008012 0.006648 0.008623 0.006737 0.004540 0.007829
    ## GVA (4 threads)    0.008011 0.006648 0.008622 0.006736 0.004540 0.007829
    ## GVA LBFGS          0.008014 0.006649 0.008625 0.006735 0.004542 0.007827

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.X1       X.X2     X.X3     X.X4      X.X5
    ## Laplace             0.04979 -0.16762 -0.1694292 -0.17262 -0.17556 -0.200872
    ## GVA                -0.02377 -0.01898 -0.0001075 -0.02263  0.01243 -0.003889
    ## GVA (4 threads)    -0.02376 -0.01899 -0.0001143 -0.02262  0.01243 -0.003896
    ## GVA LBFGS          -0.02360 -0.01889 -0.0001793 -0.02282  0.01243 -0.003983
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.X1     X.X2    X.X3    X.X4    X.X5
    ## Laplace             0.01754 0.01323 0.017497 0.02380 0.01651 0.02329
    ## GVA                 0.01821 0.01614 0.007078 0.01498 0.01085 0.01267
    ## GVA (4 threads)     0.01821 0.01614 0.007077 0.01498 0.01085 0.01267
    ## GVA LBFGS           0.01824 0.01614 0.007097 0.01506 0.01089 0.01275

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D, "cor_mat")
```

    ## $bias
    ##                     [,1]     [,2]     [,3]      [,4]     [,5]    [,6]      [,7]
    ## Laplace         -0.16939 -0.05712 -0.10296 -0.171690 -0.17276 0.02738  0.003903
    ## GVA             -0.02334  0.02218  0.02599 -0.007114  0.03355 0.05098 -0.045032
    ## GVA (4 threads) -0.02334  0.02217  0.02599 -0.007133  0.03355 0.05098 -0.045040
    ## GVA LBFGS       -0.02321  0.02209  0.02584 -0.007104  0.03394 0.05102 -0.045210
    ##                     [,8]    [,9]  [,10]    [,11]   [,12]    [,13]    [,14]
    ## Laplace         -0.12332 0.07779 0.1600 -0.16071 0.10527  0.00651  0.03554
    ## GVA              0.03228 0.12656 0.1252  0.06095 0.06329 -0.03097 -0.03649
    ## GVA (4 threads)  0.03228 0.12656 0.1251  0.06093 0.06331 -0.03099 -0.03648
    ## GVA LBFGS        0.03199 0.12648 0.1249  0.06110 0.06329 -0.03072 -0.03660
    ##                   [,15]
    ## Laplace         0.10685
    ## GVA             0.02608
    ## GVA (4 threads) 0.02613
    ## GVA LBFGS       0.02597
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.03875 0.02627 0.02559 0.06617 0.11824 0.05298 0.06157 0.13055
    ## GVA             0.05700 0.05546 0.06354 0.05003 0.07427 0.04985 0.03838 0.03403
    ## GVA (4 threads) 0.05698 0.05546 0.06352 0.05003 0.07425 0.04985 0.03838 0.03404
    ## GVA LBFGS       0.05682 0.05540 0.06343 0.04998 0.07431 0.04994 0.03845 0.03388
    ##                    [,9]   [,10]   [,11]   [,12]   [,13]   [,14]    [,15]
    ## Laplace         0.08853 0.20925 0.14214 0.05001 0.12317 0.08787 0.169925
    ## GVA             0.05588 0.05502 0.04216 0.03111 0.03057 0.04345 0.009054
    ## GVA (4 threads) 0.05588 0.05503 0.04215 0.03110 0.03058 0.04345 0.009042
    ## GVA LBFGS       0.05591 0.05489 0.04215 0.03121 0.03042 0.04358 0.009116

``` r
# computation time summary statistics 
time_stats(sim_res_6D)
```

    ##                     mean meadian
    ## Laplace         100.5400  70.325
    ## GVA               2.5774   2.418
    ## GVA (4 threads)   0.7644   0.724
    ## GVA LBFGS        23.7926  22.445

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
    ## GVA                 0.04731 -0.010054 -0.002455 -0.012158 -0.002973 -0.005322
    ## GVA (4 threads)     0.04730 -0.010045 -0.002476 -0.012159 -0.002975 -0.005329
    ## GVA LBFGS           0.04731 -0.010028 -0.002463 -0.012150 -0.002982 -0.005322
    ## 
    ## $`Standard error`
    ##                 (Intercept)     X.X1    X.X2    X.X3     X.X4     X.X5
    ## Laplace             0.02175 0.007305 0.01359 0.01550 0.007720 0.010690
    ## GVA                 0.02403 0.005139 0.01217 0.01235 0.003961 0.006186
    ## GVA (4 threads)     0.02403 0.005130 0.01218 0.01234 0.003957 0.006181
    ## GVA LBFGS           0.02403 0.005134 0.01217 0.01235 0.003959 0.006174

``` r
# bias of the random effect standard deviations
comp_bias(sim_res_6D_pois, "stdev")
```

    ## $bias
    ##                 (Intercept)     X.X1     X.X2     X.X3     X.X4     X.X5
    ## Laplace             0.17804 -0.05702 -0.06687 -0.07791 -0.05547 -0.05932
    ## GVA                -0.04062 -0.02403 -0.02070 -0.01996 -0.01171 -0.01548
    ## GVA (4 threads)    -0.04043 -0.02404 -0.02074 -0.02000 -0.01174 -0.01554
    ## GVA LBFGS          -0.04098 -0.02390 -0.02056 -0.01997 -0.01170 -0.01541
    ## 
    ## $`Standard error`
    ##                 (Intercept)    X.X1     X.X2    X.X3    X.X4    X.X5
    ## Laplace             0.02287 0.01246 0.009722 0.03472 0.01526 0.01075
    ## GVA                 0.01484 0.02043 0.011269 0.01461 0.01032 0.01000
    ## GVA (4 threads)     0.01485 0.02046 0.011268 0.01464 0.01030 0.01000
    ## GVA LBFGS           0.01492 0.02034 0.011329 0.01462 0.01033 0.01002

``` r
# bias of the correlation coefficients for the random effects
comp_bias(sim_res_6D_pois, "cor_mat")
```

    ## $bias
    ##                    [,1]     [,2]     [,3]     [,4]      [,5]    [,6]     [,7]
    ## Laplace         -0.1832 -0.12546 -0.16884 -0.16487 -0.134819 0.08803  0.03834
    ## GVA              0.1150 -0.01895 -0.03666  0.03122 -0.005803 0.08666 -0.03874
    ## GVA (4 threads)  0.1149 -0.01890 -0.03660  0.03120 -0.005734 0.08660 -0.03869
    ## GVA LBFGS        0.1150 -0.01923 -0.03704  0.03118 -0.005934 0.08623 -0.03877
    ##                    [,8]    [,9]   [,10]    [,11]   [,12]    [,13]    [,14]
    ## Laplace         0.16287 0.16003 0.01555  0.10090 0.03247 -0.02557  0.09248
    ## GVA             0.03898 0.01536 0.01497 -0.05346 0.04636 -0.06471 -0.03965
    ## GVA (4 threads) 0.03887 0.01537 0.01516 -0.05339 0.04635 -0.06450 -0.03962
    ## GVA LBFGS       0.03873 0.01547 0.01485 -0.05348 0.04618 -0.06446 -0.03980
    ##                    [,15]
    ## Laplace         -0.01128
    ## GVA              0.01856
    ## GVA (4 threads)  0.01869
    ## GVA LBFGS        0.01848
    ## 
    ## $`Standard error`
    ##                    [,1]    [,2]    [,3]    [,4]    [,5]    [,6]    [,7]    [,8]
    ## Laplace         0.04916 0.05360 0.05012 0.04923 0.02335 0.05982 0.08840 0.07918
    ## GVA             0.02846 0.02865 0.07551 0.04226 0.03738 0.07838 0.02863 0.04791
    ## GVA (4 threads) 0.02839 0.02864 0.07535 0.04226 0.03731 0.07841 0.02856 0.04786
    ## GVA LBFGS       0.02821 0.02865 0.07557 0.04232 0.03738 0.07813 0.02872 0.04794
    ##                    [,9]   [,10]   [,11]   [,12]   [,13]    [,14]   [,15]
    ## Laplace         0.05550 0.03102 0.09468 0.08898 0.07336 0.030804 0.06422
    ## GVA             0.03131 0.05379 0.06670 0.02832 0.03810 0.006603 0.06607
    ## GVA (4 threads) 0.03130 0.05380 0.06672 0.02837 0.03800 0.006515 0.06607
    ## GVA LBFGS       0.03124 0.05386 0.06679 0.02832 0.03805 0.006596 0.06612

``` r
# computation time summary statistics 
time_stats(sim_res_6D_pois)
```

    ##                   mean meadian
    ## Laplace         46.460  44.106
    ## GVA              4.183   4.062
    ## GVA (4 threads)  1.284   1.227
    ## GVA LBFGS       11.251  11.114

## References

<div id="refs" class="references">

<div id="ref-Ormerod11">

Ormerod, J. T., and M. P. Wand. 2012. “Gaussian Variational Approximate
Inference for Generalized Linear Mixed Models.” *Journal of
Computational and Graphical Statistics* 21 (1): 2–17.
<https://doi.org/10.1198/jcgs.2011.09118>.

</div>

</div>
