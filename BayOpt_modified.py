import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import math
 
# objective function
def objective(x):
	# QoC function (i.e., RSSI)
	r_max = 50 # RSSI when main lobe is aligned with Tx
	Delta_r = 20 # RSSI excursion when main lobe is opposite to Tx
	a_c = -Delta_r/np.pi**2 # absolute value proportional to the main lobe of the Rx
	sigma_max = 4.0
	sigma_min = 1.0
	lamda = (sigma_max - sigma_min)/np.pi
	sigma = sigma_min + lamda * abs(x)  # noise proportional to the Tx-Rx alignment
	X_sigma = np.random.normal(loc=0, scale=sigma) 
	r = r_max + a_c*x**2 + X_sigma
	
	# QoS function (i.e., PoD)
	alpha = np.pi/3 # half-angle of view
	pod_max = 0.95
	a_s = - pod_max/alpha**2
	omega = np.random.normal(loc=0, scale=5)*0.25 # [mm]
	f = 50 # [mm]
	pod = 0 if abs(x) > alpha else  pod_max + a_s* (math.atan( math.tan( x ) + omega/f ))**2 

	return r*pod
 
# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = np.random.uniform(-np.pi,np.pi,100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = np.argmax(scores)
	return Xsamples[ix, 0]
 
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = np.asarray(np.arange(min(X), max(X), 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()
 
# sample the domain sparsely with noise
X = np.random.uniform(-np.pi,np.pi,100)
y = np.asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)
# perform the optimization process
for i in range(400):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)
	# summarize the finding
	est, _ = surrogate(model, [[x]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# add the data to the dataset
	X = np.vstack((X, [[x]]))
	y = np.vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)
 
# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = np.argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix]*180/np.pi, y[ix]))
