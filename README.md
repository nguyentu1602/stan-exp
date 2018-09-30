# stan-exp
Experiments with Stan

# Installation
## Stan
	http://pystan.readthedocs.io/en/latest/getting_started.html
	
## Conda comands
conda install -c anaconda pystan
conda install -c conda-forge SOMEPACKAGE


## Conda R installation
Activate the env, and then:
$ conda install -c r r
$ conda install -c r rstan


For some reason, the env is called rstan anyway, so you could activate `rstan`
instead. 
### Rstudio
Inside the env run:
$ conda install -c r rstudio
Launching rstudio from inside the env makes sure you get the
configuration right.

### Rstanarm
For `rstanarm` you'll need to install it inside the R terminal, then the
compilation of rstan takes a while, but it's worth it.

Testing:
https://cran.r-project.org/web/packages/rstanarm/vignettes/rstanarm.html


### Jupyternotebook with R
TODO


# Making new notebooks
- Activate the env
- jupyter notebook --port=8889 --ip=0.0.0.0 &

