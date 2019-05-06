from scipy import stats
from scipy.stats import norm
# Approximate probability of ruin:
trials = 1000
U_hist_1 = np.zeros(trials)
U_hist_2 = np.zeros(trials)

for n in range(trials):
    U_array_1 = U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1=0.4,p2=0.6*0.75,p3=0.6*0.25)
    U_array_2 = U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1=0.9,p2=0.1*0.75,p3=0.1*0.25)
    U_hist_1[n] = U_array_1[-1]
    U_hist_2[n] = U_array_2[-1]

mu = np.mean(U_hist_1)
mu1 = np.mean(U_hist_2)
sigma = np.std(U_hist_1)
sigma1 = np.std(U_hist_2)

y, bins, patches = plt.hist(U_hist_1, density = True, bins = 50, label = '0.4 cash', rwidth=0.9)
y1, bins1, patches1 = plt.hist(U_hist_2, density = True, bins = 50, label = '0.9 cash', rwidth=0.9)

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
