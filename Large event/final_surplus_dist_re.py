from scipy import stats
from scipy.stats import norm
# Approximate probability of ruin:
trials = 1000
U_hist = np.zeros(trials)
U_cov_hist = np.zeros(trials)

for n in range(trials):
    U_array, U_array_cov = U_wo(t,kappa,P,P_re,N,U_0,initial,t_M,t_mu,c_hat,p1=0.9,p2=0.05,p3=0.05)
    U_hist[n] = U_array[-1]
    U_cov_hist[n] = U_array_cov[-1]

mu = np.mean(U_hist)
mu1 = np.mean(U_cov_hist)
sigma = np.std(U_hist)
sigma1 = np.std(U_cov_hist)

y, bins, patches = plt.hist(U_hist, density = True, bins = 80, label = '0.9 cash without coverage', rwidth=0.9)
y1, bins1, patches1 = plt.hist(U_cov_hist, density = True, bins = 80, label = '0.9 cash with coverage', rwidth=0.9)

y = norm.pdf(bins, mu, sigma)
y1 = norm.pdf(bins1, mu1, sigma1)

plt.plot(bins, y, 'r-', linewidth=1)
plt.plot(bins1, y1, 'r-', linewidth=1)

#print("The median is {} ".format(median))

#print(stats.describe(U_hist))
#print(y.max())
#print(np.argmax(y))
#print(bins)
#print('interval is [{0}, {1}]'.format(bins[np.argmax(y) + 2], bins[np.argmax(y) + 3]))


plt.legend()
plt.show()



## Zoom to the right end:

y, bins, patches = plt.hist(U_hist, density = True, bins = 80, label = '0.9 cash without coverage', rwidth=0.9)
y1, bins1, patches1 = plt.hist(U_cov_hist, density = True, bins = 80, label = '0.9 cash with coverage', rwidth=0.9)

median = '{:.2e}'.format(np.median(U_hist))
median1 = '{:.2e}'.format(np.median(U_cov_hist))

y = norm.pdf(bins, mu, sigma)
y1 = norm.pdf(bins1, mu1, sigma1)

plt.plot(bins, y, 'r-', linewidth=1)
plt.plot(bins1, y1, 'r-', linewidth=1)

plt.xlim(left = 1e9)

plt.legend()
plt.show()


## Zoom to the left end:

y, bins, patches = plt.hist(U_hist, density = True, bins = 80, label = '0.9 cash without coverage', rwidth=0.9)
y1, bins1, patches1 = plt.hist(U_cov_hist, density = True, bins = 80, label = '0.9 cash with coverage', rwidth=0.9)

median = '{:.2e}'.format(np.median(U_hist))
median1 = '{:.2e}'.format(np.median(U_cov_hist))

y = norm.pdf(bins, mu, sigma)
y1 = norm.pdf(bins1, mu1, sigma1)

plt.plot(bins, y, 'r-', linewidth=1)
plt.plot(bins1, y1, 'r-', linewidth=1)

plt.xlim(left = -8e9, right = 0.0)
plt.ylim(top = 0.1e-9)

plt.legend()
plt.show()
