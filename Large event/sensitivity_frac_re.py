import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import poisson

def S(t,S0=1292.98):
    """Black scholes model for pricing stocks
    input: time t is t'th month
           S0 is initial price of stock
    output: estimate of price at month t"""
    
    SS = 0
    for j in range(t):
        SS += 0.013072-((0.036825)**2)/2 + (0.036825)*np.random.normal(0,1)
    value = S0*np.exp(SS)
    
    return value


def R(t):
    """model of interest rate per month"""
    """output: (price at time t - price at time t-1)/(price at time t-1)
               using S(t) to estimate price"""
    RR = (S(t)-S(t-1))/S(t-1)
    
    # in case negative return, we will not realize the capital loss
    if RR <0:
        RR=0
    return RR


def C(initial,t):
    """input: t how many months ahead of initial month, and months take num value Jan = 0, Feb = 1, and 13 is Jan, 14 Feb, etc"""
    """output: estimated claim amount need to be payed at month (initial + t) mod 12"""
     
    # number of claims in 2018-2019, from Jan, Feb, etc, our estimate of number of claims/mon [https://www.fema.gov/number-losses-month]
    lamb = np.asarray([np.random.normal(292,10),np.random.normal(2933,100),np.random.normal(2064,100),np.random.normal(1157,100),np.random.normal(1215,100),np.random.normal(1995,100),np.random.normal(1336,100),np.random.normal(2092,100),np.random.normal(18975,1000),np.random.normal(7013,100),np.random.normal(445,10),np.random.normal(909,10)])
    # average paid losses (claim amounts in 2018-19) [https://www.fema.gov/average-claim-payments-date-loss]
    clam = np.asarray([14493,34900,27527,47056,28903,45289,20684,35027,45049,52767,20233,28355])
    
    # t modulo 12: this is used because if we forecast more than a year, we have to go back to Jan, etc    
    k = (initial + t) % 12
    N = poisson.rvs(lamb[k])
    X = 0
    for i in range(N):
        X += np.random.normal(clam[k],1000)
        
    return X


# O function:
def O(t,t_mu):
    if t>t_mu or t == t_mu:
        value = 0 
    else:
        value = 1
    return value

# Indicator fcn (for I(t=t_M)):
def I(t,t_M):
    if t==t_M:
        value = 1
    else:
        value = 0
    return value

# Indicator fcn (for I(t>=t_mu)):
def I_2(t,t_mu):
    if t>t_mu or t==t_mu:
        value = 1
    else:
        value = 0
    return value

# Indicator fcn (for I(C1<C2)):
def I_3(C1,C2):
    if C1 <= C2:
        value = 1
    else:
        value = 0
    return value

# Surplus modeling in time with and without reinsurance, with the risk of large, rare event occuring:
def U_wo(t,kappa,P,P_re,N,U_0,initial,t_M,t_mu,c_hat,frac,p1,p2,p3):
    """model of surplus after time t, after considering investment income, premium, and liabilities"""
    """input: p is the percentage of money left as emergency/faciliatory use"""
    """       P is the monthly premium cost"""
    """       P_re is the yearly reinsurance premium"""
    """       N is the number of insurance holders"""
    """       frac is the random(normal) fraction of payers each month"""
    """       U_0 is the initial amount invested by others as seed money for insurance program"""
    """       initial is the numerical value of the month you are starting from"""
    """       t_M is the time of maturity (return the principal to investors)"""
    """       kappa is coupon rate of bond insurance company buys for income"""
    """       c_hat is the coupon rate we pay out to our investors"""
    """       t_mu is the time of maturity of bonds invested by insurance company for income"""
    """       p1 is cash proportion
              p2 is stock proportion
              p3 is bond proportion"""
    
    
    # initial case:
    if t == 0:
        U_arr = np.zeros(1)
        U_arr[0] = 0.1*U_0
        U_t = U_arr[0]
        
    # for t > 0 forecast:    
    if t > 0:
        U_arr = np.zeros(t+1) # no coverage
        U_arr[0] = 0.1*U_0
        U_arr_cov = np.zeros(t+1) # coverage
        U_arr_cov[0] = 0.1*U_0
        
        # rare large event claim estimate:
        C_rare = np.random.normal(8e9,1e9) # based on reinsurer's coverage detail
        
        for i in range(1,t+1):
            # did rare large event occur?
            I_33 = I_3(np.random.uniform(0,1),0.0033) #prob estimated from return curve
            tt = i
            # to compare with and without reins coverage, we save the same simulation value
            Rt = R(tt)
            Ct = C(initial, tt)
            U_arr[i] = U_arr[i-1] + frac*p1*N*P + O(tt,t_mu+1)*p3*U_0*kappa/12 + Rt*p2*U_0 + I_2(tt,2)*Rt*p2*frac*N*P +\
            O(tt,t_mu+1)*I_2(tt,2)*p3*frac*N*P*kappa/12 - Ct - O(tt,t_M+1)*c_hat/12 * U_0 - I(tt,t_M)*U_0 + I(tt,t_mu)*(p3*U_0+(t_mu-1)*p3*frac*N*P)-\
            14.69e6 - I_33*(C_rare) # no coverage
            
            U_arr_cov[i] =U_arr_cov[i-1] + frac*p1*N*P + O(tt,t_mu+1)*p3*U_0*kappa/12 + Rt*p2*U_0 + I_2(tt,2)*Rt*p2*frac*N*P +\
            O(tt,t_mu+1)*I_2(tt,2)*p3*frac*N*P*kappa/12 - Ct - O(tt,t_M+1)*c_hat/12 * U_0 - I(tt,t_M)*U_0 + I(tt,t_mu)*(p3*U_0+(t_mu-1)*p3*frac*N*P)-\
            14.69e6 - I_33*(C_rare-0.56*C_rare) - I(t%12,0)*P_re # coverage

  #  U_t = U_arr[-1]
  #  if U_t < 0:
       # print("Going bankrupt!!")
       # print(U_t)
       # raise Exception("Company went bankrupt")

    return U_arr, U_arr_cov

initial = 0 # April,2009
t = 28 # number of months to forecast
t_M = 25 # maturity time to pay back investors
t_mu = 24 # mat time for our bond investment
P = 60 # avg 700 per year for insurance
N = 5000000
P_re = 235e6
U_0 = 800e+6 # initial accrued by investors
kappa = np.random.uniform(0.1,0.11) # coupon rate for bond investments
c_hat = np.random.uniform(0.1,0.11)


# Approximate probability of ruin, and see the sensitivity to uncertainty of fraction of payers:
# change st dev of fraction of premium payers while keeping the mean constant (mu=80%)
# to test how not investing at all is not good option because if there is more uncertainty in fraction of payers then we are in trouble

# array to hold all different probs of ruin wrt frac, first row without coverage and second row with coverage:
PR1 = np.zeros((2,len(np.linspace(0.1,0.4,15))))
i = 0
for stdev in np.linspace(0.1,0.4,15):
    
    # calculate prob of ruin:
    p = 0  # without coverage
    q = 0  # with coverage
    trials = 1000

    for n in range(trials):
        frac = np.random.normal(0.8,stdev)
        U_array, U_array_cov = U_wo(t,kappa,P,P_re,N,U_0,initial,t_M,t_mu,c_hat,frac,p1=0.9,p2=0.1*0.75,p3=0.25*0.1)
        if any(u_t < 0 for u_t in U_array)==True:
            p += 1
        if any(u_t < 0 for u_t in U_array_cov)==True:
            q += 1
    p /= trials
    p = p*100
    q /= trials
    q = q*100
    PR1[0,i] = p
    PR1[1,i] = q
    i += 1


fig1 = plt.figure()
plt.title('Uncertainty in fraction of payers, with p1=0.9, Pf3')
plt.errorbar(np.linspace(0.1,0.4,15),PR1[0,:],yerr=100/np.sqrt(1000),label='without coverage')
plt.errorbar(np.linspace(0.1,0.4,15),PR1[1,:],yerr=100/np.sqrt(1000),label='with coverage')
plt.xlabel('Uncertainty in fraction of payers')
plt.ylabel('Prob of Ruin')
plt.legend()
#plt.show()
fig1.savefig('PR_frac_re.png')


