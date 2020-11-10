
First, we source the files:

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
adaptive Gauss-Hermite quadrature implementation and the Laplace
implementation in `lme4` package. First we assign the seeds we will use.

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

Next we set up the functions we will need to perform the simulation
study.

``` r
# simulates from a mixed probit model. 
# 
# Args: 
#   n_cluster: number of clusters
#   cor_mat: the correlation matrix.
#   beta: the fixed effect coefficient vector.
#   sig: scale parameter for the covariance matrix.
#   n_obs: number of members in each cluster.
#   get_x: function to get the fixed effect covariate matrix.
#   get_z: function to get the random effect covariate matrix.
sim_dat <- function(n_cluster = 100L, cor_mat, beta, sig, n_obs = 10L, 
                    get_x, get_z){
  vcov_mat <- sig * cor_mat   # the covariance matrix
  
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
    prob <- 1/(1 + exp(-eta))
    nis <- sample.int(5L, n_obs, replace = TRUE)
    y <- rbinom(n_obs, nis, prob)
    nis <- as.numeric(nis)
    y <- as.numeric(y)
    
    # return 
    group <<- group + 1L
    list(y = y, group = rep(group, n_obs), 
         nis = nis, X = X, Z = Z)
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
est_lme4 <- function(dat, formula, nAGQ = 1L){
  library(lme4)
  fit <- glmer(formula = formula, dat, family = binomial(), nAGQ = nAGQ, 
               weights = nis)
  vc <- VarCorr(fit)
  list(ll = c(logLik(fit)), fixef = fixef(fit), 
       stdev = attr(vc$group, "stddev"), 
       cor_mat = attr(vc$group, "correlation"))
}

# estimates the model using a GVA. 
# 
# Args: 
#   dat: list with a list for each cluster. 
#   rel_eps: relative convergence threshold.
#   est_psqn: logical for whether to use psqn.
est_va <- function(dat, rel_eps = 1e-8, est_psqn){
  func <- get_lb_optimizer(dat$list_dat, 1L)
  
  # setup stating values
  n_clust <- length(dat$list_dat)
  n_rng <- NROW(dat$list_dat[[1L]]$Z)
  n_fix <- NROW(dat$list_dat[[1L]]$X)
  
  par <- numeric(n_fix + n_clust * n_rng +
                   (n_clust + 1) * n_rng * (n_rng + 1L) / 2)
  
  # estimate the fixed effects w/ a GLM
  if(n_fix > 0)
    par[1:n_fix] <- with(dat$df_dat, glm.fit(
      x = dat$df_dat[, grepl("^X", colnames(dat$df_dat))], 
      y = y / nis, weights = nis, family = binomial()))[[
        "coefficients"]]
  
  if(est_psqn){
    # par <- opt_priv(val = par, ptr = func, rel_eps = rel_eps^(2/3),
    #                 max_it = 100, n_threads = 1L, c1 = 1e-4, c2 = .9)
    res <- opt_lb(val = par, ptr = func, rel_eps = rel_eps, max_it = 1000L, 
                  n_threads = 1L, c1 = 1e-4, c2 = .9, cg_tol = .2, 
                  max_cg = max(2L, as.integer(log(n_clust) * 10)))
  } else {
    library(lbfgs)
    fn <- function(x)
      eval_lb   (x, ptr = func, n_threads = 1L)
    gr <- function(x)
      eval_lb_gr(x, ptr = func, n_threads = 1L)
    res <- lbfgs(fn, gr, par, invisible = 1)
  }
  
  mod_par <- head(res$par, n_fix + n_rng * (n_rng + 1) / 2)
  Sig_hat <- get_pd_mat(tail(mod_par, -n_fix), n_rng)[[1L]]
  
  list(lb = -res$value, fixef = head(mod_par, n_fix), 
       stdev = sqrt(diag(Sig_hat)), cor_mat = cov2cor(Sig_hat))
}
```

Then we perform the simulation study.

``` r
# assign covariate functions
get_x <- function(x)
  cbind(`(Intercept)` = 1, `cont` = scale(x), `dummy` = x %% 2)
get_z <- function(x)
  cbind(`(Intercept)` = rep(1, length(x)))

# perform the simulation
sim_res_uni <- lapply(seeds, function(s){
  f <- file.path("cache", sprintf("univariate-%d.RDS", s))
  if(!file.exists(f)){
    set.seed(s)
    dat <- sim_dat(n_cluster = 1000L, cor_mat = diag(1), beta = c(-2, -1, 1), 
                   sig = 1.5, n_obs = 3L, get_x = get_x, get_z = get_z)
    
    lap_time <- system.time(lap_fit <- est_lme4(
      formula = y / nis ~ X.V2 + X.dummy + (1 | group), 
      dat = dat$df_dat, nAGQ = 1L))
    agh_time <- system.time(agh_fit <- est_lme4(
      formula = y / nis ~ X.V2 + X.dummy + (1 | group), 
      dat = dat$df_dat, nAGQ = 25L))
    gva_time <- system.time(gva_fit <- est_va(dat, est_psqn = TRUE))
    gva_lbfgs_time <- system.time(
      gva_lbfgs_fit <- est_va(dat, est_psqn = FALSE))
    
    get_bias <- function(x)
      list(fixed = x$fixef - dat$beta, 
           stdev = x$stdev - dat$stdev,
           cor_mat = (x$cor_mat - dat$cor_mat)[lower.tri(dat$cor_mat)])
    
    bias <- mapply(rbind, Laplace = get_bias(lap_fit), 
                   AGHQ = get_bias(agh_fit), GVA = get_bias(gva_fit), 
                   `GVA LBFGS` = get_bias(gva_lbfgs_fit))
    tis <- rbind(Laplace = lap_time, AGHQ = agh_time, GVA = gva_time, 
                 `GVA LBFGS` = gva_lbfgs_time)[, 1:3]
    
    out <- list(bias = bias, time = tis, 
                lls = c(Laplace = lap_fit$ll, AGHQ = agh_fit$ll, 
                        GVA = gva_fit$lb, `GVA LBFGS` = gva_lbfgs_fit$lb), 
                seed = s)
    
    message(paste0(capture.output(out), collapse = "\n"))
    message("")
    saveRDS(out, f)
  }
  
  readRDS(f)
})
```

The bias estimates are given below:

``` r
# function to compute the bias and the standard errors.
comp_bias <- function(results, what){
  errs <- sapply(results, function(x) x$bias[[what]], 
                 simplify = "array")
  ests <- apply(errs, 1:2, mean)
  SE <- apply(errs, 1:2, sd) / sqrt(dim(errs)[[3]])
  list(bias = ests, `Standard error` = SE)
}

# the fixed effect 
comp_bias(sim_res_uni, "fixed")
```

    ## $bias
    ##           (Intercept)     X.V2   X.dummy
    ## Laplace     0.0057265 0.004543 -0.005410
    ## AGHQ        0.0002187 0.003648 -0.004550
    ## GVA         0.0057922 0.004485 -0.005392
    ## GVA LBFGS   0.0057042 0.004474 -0.005367
    ## 
    ## $`Standard error`
    ##           (Intercept)     X.V2  X.dummy
    ## Laplace      0.006462 0.003665 0.006755
    ## AGHQ         0.006512 0.003675 0.006788
    ## GVA          0.006479 0.003672 0.006789
    ## GVA LBFGS    0.006478 0.003672 0.006787

``` r
# the random effect standard deviation
comp_bias(sim_res_uni, "stdev")
```

    ## $bias
    ##           (Intercept)
    ## Laplace     -0.039098
    ## AGHQ        -0.007815
    ## GVA         -0.022147
    ## GVA LBFGS   -0.022143
    ## 
    ## $`Standard error`
    ##           (Intercept)
    ## Laplace      0.005343
    ## AGHQ         0.005502
    ## GVA          0.005343
    ## GVA LBFGS    0.005347

Summary stats for the computation are given below:

``` r
time_stats <- function(results){
  tis <- sapply(results, `[[`, "time", simplify = "array")
  tis <- tis[, "elapsed", ]
  cbind(mean    = apply(tis, 1, mean), 
        meadian = apply(tis, 1, median))
}

time_stats(sim_res_uni)
```

    ##             mean meadian
    ## Laplace   0.7119  0.6950
    ## AGHQ      1.8258  1.8220
    ## GVA       0.2416  0.2445
    ## GVA LBFGS 2.6478  2.6075
