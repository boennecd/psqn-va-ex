f <- function(x, mu, sigma){
  eta <- sigma * x + mu
  -dnorm(x) * ifelse(eta > 30, x, log(1 + exp(eta)))
}
  
mus <- seq(-4, 4, length.out = 100)
sigmas <- seq(.0001, 2, length.out = 100)
gr <- expand.grid(mu = mus, sigma = sigmas)
int_vals <- mapply(function(x, mu, sigma){
  opt <- optim(0, f, mu = mu, sigma = sigma, 
               method = "BFGS", control = list(reltol = 1e-10, 
                                               maxit = 10000L))
  if(!opt$convergence == 0L)
    browser()
  stopifnot(opt$convergence == 0L)
  opt$par
}, mu = gr$mu, sigma = gr$sigma)
range(int_vals)

contour(mus, sigmas, matrix(int_vals, length(mus)))

df <- gr
df$y <- int_vals
fit <- lm(y ~ poly(mu, sigma, raw = TRUE, degree = 4), df)
summary(fit)
dput(coef(fit), control = c())
length(coef(fit))

# the range of error
range(fit$residuals)
range((df$y - pmax(0, predict(fit))) / df$y)
contour(mus, sigmas, matrix(pmax(0, predict(fit)), length(mus)))
contour(mus, sigmas, matrix(fit$residuals, length(mus)))

head(with(df, poly(mu, sigma, raw = TRUE, degree = 4)))
