library(Rcpp)
library(numDeriv)
library(fastGHQuad)
sourceCpp(file.path("src", "glmm-gva.cpp"), embeddedR = FALSE, 
          verbose = TRUE)


# switch the default
formals(logit_partition)$adaptive <- TRUE

# integrand in the logit partition function
mus <- seq(-4, 4, length.out = 20)
sigmas <- seq(.1, 4, length.out = 20)
gr <- expand.grid(mu = mus, sigma = sigmas)

res <- mapply(function(mu, sigma) {
  f <- function(x, mu, sigma)
    dnorm(x) * ifelse(x > 30, x, log(1 + exp(sigma * x + mu)))
  log_f <- function(x, mu, sigma)
    -(dnorm(x, log = TRUE) + 
        ifelse(x > 30, log(x), log(log(1 + exp(sigma * x + mu)))))
  get_scale <- function(mu, sigma, mode)
    sqrt(1/hessian(function(x) log_f(x, mu = mu, sigma = sigma), mode))
  
  r_use <- gaussHermiteData(40L)
  
  (truth <- integrate(function(x) f(x, mu, sigma), -Inf, Inf, 
                      rel.tol = 1e-12)$value)
    
  opt <- optim(0, function(x) log_f(x, mu = mu, sigma = sigma))
  (mode <- opt$par)
  
  (scal <- drop(get_scale(mu = mu, sigma = sigma, mode = mode)))
  
  err1 <- aghQuad(function(x) f(x, mu = mu, sigma = sigma), mode, scal          , rule = r_use) - truth
  err2 <- aghQuad(function(x) f(x, mu = mu, sigma = sigma), mode, scal / sqrt(2), rule = r_use) - truth
  c(scaled = err1, nonscaled = err2, scal, truth)
}, mu = gr$mu, sigma = gr$sigma)

range(res[1, ])
contour(mus, sigmas, matrix(res[1, ], length(mus)))
range(res[2, ])
contour(mus, sigmas, matrix(res[2, ], length(mus)))

range(res[4, ])
contour(mus, sigmas, matrix(res[4, ], length(mus)))
persp(mus, sigmas, matrix(res[4, ], length(mus)))

# check the relative error

grid <- expand.grid(mu = mus, sigma = sigmas)

rel_err <- mapply(function(mu, sigma){
  truth <- integrate(f, -Inf, Inf, mu = mu, sigma = sigma, rel.tol = 1e-13)
  est <- logit_partition(mu = mu, sigma = sigma, order = 0)
  (truth$value - est) / truth$value 
}, mu = grid$mu, sigma = grid$sigma)




ddf(-0.210526, 3, 0.78775)
