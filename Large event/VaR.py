# Approximate probability of ruin:
p = 0
p_cov = 0
trials = 1000
U_hist = np.zeros(trials)
U_cov_hist = np.zeros(trials)

for n in range(trials):
    U_array, U_array_cov = U_wo(t,kappa,P,P_re,N,U_0,initial,t_M,t_mu,c_hat,p1=0.9,p2=0.05,p3=0.05)
    U_hist[n] = np.min(U_array)
    U_cov_hist[n] = np.min(U_array_cov)
    
    if any(u_t < 0 for u_t in U_array) == True:
        p += 1
    if any(u_t_cov < 0 for u_t_cov in U_array_cov) == True:
        p_cov += 1
    
p /= trials
p_cov /= trials

p = p*100
p_cov = p_cov*100
# portfolio structure: cash = p1 and investment = 1-p1 and stock:bonds = 1:1 and vary p1
#                    : cash = p1 and investment = 1-p1 and stock:bonds = 1:3 and vary p1
#                    :   ''                   ''                       = 3:1 and vary p1


def add_neg(arr):
    lst = sorted(arr[arr < 0])
    
    return lst[-1]

print('The probability of ruin approx without coverage will be {0} percent and with coverage will be {1} percent'.format(p, p_cov))
print('Without coverage {} percent of confidence to avoid loss not exceeding {:.2e}'.format(100-p, add_neg(U_hist)))
print('With coverage {} percent of confidence to avoid loss not exceeding {:.2e}'.format(100-p_cov, add_neg(U_cov_hist)))
print('The value at risk of {} percent for 28 months without reinsurance is {:.2e}',format(100-p,-add_neg(U_hist)))
print('The value at risk of {} percent for 28 months with reinsurance is {:.2e}',format(100-p,-add_neg(U_cov_hist)))
