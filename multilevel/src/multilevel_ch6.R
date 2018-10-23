## This is a new package that's useful to extend the rstan package
install.packages('rstanarm')
library(rstanarm)


## This is an old package for Gelman's multi-level book
# install.packages('arm') 
# library(arm)

# install.packages(c("foreign", "MASS", "reshape2", "Hmisc"))
library(foreign)
library(MASS)
library(reshape2)
library(Hmisc)

# 6.5 Multinomial Regression ----------------------------------------------

# Example online: https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
## Read the data in
dat <- read.dta("https://stats.idre.ucla.edu/stat/data/ologit.dta")
head(dat)

## one at a time, make descriptive stats on these columns
lapply(dat[, c("apply", "pared", "public")], table)

## three way cross tabs (xtabs) and flatten the table
ftable(xtabs(~ public + apply + pared, data = dat))

summary(dat$gpa)
sd(dat$gpa)

## ORDERED LOGISTIC REGRESSION
# Below we use the polr command from the MASS package to estimate an ordered logistic regression model. The command name comes from proportional odds logistic regression, highlighting the proportional odds assumption in our model. polr uses the standard formula interface in R for specifying a regression model with outcome followed by predictors. We also specify Hess=TRUE to have the model return the observed information matrix from optimization (called the Hessian) which is used to get standard errors.

## fit ordered logit model and store results 'm'
m <- polr(apply ~ pared + public + gpa, data=dat, Hess=TRUE)
summary(m)

## we could approximate a p-value with normality assumption, which would
## hold if you have enough datapoints
## store table
(ctable <- coef(summary(m)))

## calculate and store p values
pvals <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
pvals
## combined table
(ctable <- cbind(ctable, "p value" = pvals))

## we can also get confidence intervals
(ci <- confint(m)) # default method gives profiled CIs



sf <- function(y) {
  c('Y>=1' = qlogis(mean(y >= 1)),
    'Y>=2' = qlogis(mean(y >= 2)),
    'Y>=3' = qlogis(mean(y >= 3)))
}

(s <- with(dat, summary(as.numeric(apply) ~ pared + public + gpa, fun=sf)))


glm(I(as.numeric(apply) >= 2) ~ pared, family="binomial", data = dat)
glm(I(as.numeric(apply) >= 3) ~ pared, family="binomial", data = dat)

s[, 4] <- s[, 4] - s[, 3]
s[, 3] <- s[, 3] - s[, 3]
s # print

plot(s, which=1:3, pch=1:3, xlab='logit', main=' ', xlim=range(s[,3:4]))


newdat <- data.frame(
  pared = rep(0:1, 200),
  public = rep(0:1, each = 200),
  gpa = rep(seq(from = 1.9, to = 4, length.out = 100), 4))

newdat <- cbind(newdat, predict(m, newdat, type = "probs"))

##show first few rows
head(newdat)

lnewdat <- melt(newdat, id.vars = c("pared", "public", "gpa"),
                variable.name = "Level", value.name="Probability")
## view first few rows
head(lnewdat)

ggplot(lnewdat, aes(x = gpa, y = Probability, colour = Level)) +
  geom_line() + facet_grid(pared ~ public, labeller="label_both")


