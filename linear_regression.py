import pymc
import numpy as np
import matplotlib.pyplot as plt

#here we import the data

name_file = 'Tremaine_2002.txt'

mbh = np.loadtxt(name_file, usecols=(3,))
sigma = np.loadtxt(name_file, usecols=(7,))

MBH = np.log10(mbh)
SIGMA = np.log10(sigma/200)

plt.scatter(SIGMA, MBH)
plt.show()

#now we define the priors (alpha, beta). In this case we choose uniform priors

alpha =  pymc.Uniform('alpha', lower=-5, upper=5)
beta = pymc.Uniform('beta', lower=-10, upper=10)

#Finally, we define our observations

x = pymc.Normal('x', mu=0, tau=1, value=SIGMA, observed=True)

@pymc.deterministic(plot=False)
def linear_regress(x=x, alpha=alpha, beta=beta):
    return x*alpha + beta

y = pymc.Normal('output',mu=linear_regress, value=MBH, observed=True)


model = pymc.Model([x, y, alpha, beta])
mcmc = pymc.MCMC(model)
mcmc.sample(iter=10000,burn=1000, thin=10)
