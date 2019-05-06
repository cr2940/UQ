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

# Surplus modeling in time:

## U(t) = (1-p)U_0 + pU_0e^{tR(t)} + tNP - C(t,\omega) - \hat{c}(t,\omega)/12*U_0 - I_{t_M}U_0 ##

def U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1,p2,p3):
    """model of surplus after time t, after considering investment income, premium, and liabilities"""
    """input: p is the percentage of money left as emergency/faciliatory use"""
    """       P is the monthly premium cost"""
    """       N is the number of insurance holders"""
    """       U_0 is the initial amount invested by others as seed money for insurance program"""
    """       initial is the numerical value of the month you are starting from"""
    """       t_M is the time of maturity (return the principal to investors)"""
    """       kappa is coupon rate of bond insurance company buys for income"""
    """       t_mu is the time of maturity of bonds invested by insurance company for income"""
    """       p1 is cash proportion
              p2 is stock proportion
              p3 is bond proportion"""
    
    frac = np.random.normal(0.8,0.1) # fraction of people who pay premium
#     print("for t=",t)
#     print("cash=",p1*(U_0+t*N*P*frac_pay))
#     print("buy stock=",-I(t,0)*p2*U_0)
#     print("stock price=",p2*(U_0+t*N*P*frac_pay)*(1+R(t))**t)
#     print("bond inv coupon=",O(t,t_mu+1)*p3*kappa*t*U_0)
#     print("get back bond=",I(t,t_mu)*p3*U_0)
#     print("premium accrue=",0.6*t*N*P*frac_pay)  # [https://www.fema.gov/total-earned-premium-calendar-year]
#     print("claim liabilities=",-C(initial,t))
#     print("coupon payment liab to investors=",-O(t,t_M+1)*t*(c_hat(t)/12)*U_0)
#     print("principal payback to investor=",-I(t,t_M)*U_0)
    
    # value array for storage:
    
    
    # initial case:
    if t == 0:
        U_arr = np.zeros(1)
        U_arr[0] = 0.1*U_0
        U_t = U_arr[0]
        
    if t > 0:
        U_arr = np.zeros(t+1)
        U_arr[0] = 0.1*U_0
        for i in range(1,t+1):
            tt = i
            
            # the uncertain terms:
            Rt = R(tt)
            Ct = C(initial, tt)
           
            U_arr[i] = U_arr[i-1] + frac*p1*N*P + O(tt,t_mu+1)*p3*U_0*kappa/12 + Rt*p2*U_0 + I_2(tt,2)*Rt*p2*frac*N*P +\
            O(tt,t_mu+1)*I_2(tt,2)*p3*frac*N*P*kappa/12 - Ct - O(tt,t_M+1)*c_hat/12 * U_0 - I(tt,t_M)*U_0 + I(tt,t_mu)*(p3*U_0+(t_mu-1)*p3*frac*N*P) -\
            14.69e6        

    U_t = U_arr[-1]
#    if U_t < 0:
       # print("Going bankrupt!!")
 #       print(U_t)
       # raise Exception("Company went bankrupt")

    return U_arr

initial = 0 # April,2009
t = 28 # number of months to forecast
t_M = 24 # maturity time to pay back investors
t_mu = 23 # mat time for our bond investment
P = 80 # avg 700 per year for insurance
N = 5000000
U_0 = 800e+6 # initial accrued by investors
kappa = np.random.uniform(0.1,0.11) # coupon rate for bond investments
c_hat = np.random.uniform(0.08,0.1)

# Add uncertainty in people paying their fees


# Approximate probability of ruin, and optimize over portfolio:
# portfolio structure 1: cash = p1 and investment = 1-p1 and stock:bonds = 1:1 and vary p1
#                     2: cash = p1 and investment = 1-p1 and stock:bonds = 1:3 and vary p1
#                     3:   ''                   ''                       = 3:1 and vary p1


# array to hold all different probs of ruin wrt p1 and portfolio:
PR1 = np.zeros(10)
i = 0
for p1 in np.linspace(0.1,1,10):
    invest = 1-p1
    p2 = invest/2
    p3 = invest/2

    # calculate prob of ruin:
    p = 0
    trials = 1000

    for n in range(trials):
        U_array = U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1,p2,p3)
        if any(u_t < 0 for u_t in U_array)==True:
            p += 1
    p /= trials
    p = p*100
    PR1[i] = p
    i += 1
#    print("The probability of ruin approx will be ", p, "percent with portfolio type 1")


fig1 = plt.figure()
#plt.plot(np.linspace(0.1,1,10),PR1,label='Portfolio 1')
plt.errorbar(np.linspace(0.1,1,10),PR1,yerr=100/np.sqrt(1000),marker='*',label='Portfolio 1')
plt.xlabel('p1, cash holding')
plt.ylabel('Prob of ruin')


# array to hold all different probs of ruin wrt p2 and portfolio:
PR2 = np.zeros(10)
i = 0
for p1 in np.linspace(0.1,1,10):
    invest = 1-p1
    p2 = invest/4
    p3 = 3*invest/4

    # calculate prob of ruin:
    p = 0
    trials = 1000

    for n in range(trials):
        U_array = U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1,p2,p3)
        if any(u_t < 0 for u_t in U_array)==True:
            p += 1
    p /= trials
    p = p*100
    PR2[i] = p
    i+=1

#plt.plot(np.linspace(0.1,1,10),PR2,label='Portfolio 2')
plt.errorbar(np.linspace(0.1,1,10),PR2,yerr=100/np.sqrt(1000),marker='*',label='Portfolio 2')

# array to hold all different probs of ruin wrt p1 and portfolio:
PR3 = np.zeros(10)
i = 0
for p1 in np.linspace(0.1,1,10):
    invest = 1-p1
    p2 = 3*invest/4
    p3 = invest/4

    # calculate prob of ruin:
    p = 0
    trials = 1000

    for n in range(trials):
        U_array = U(t,kappa,P,N,U_0,initial,t_M,t_mu,c_hat,p1,p2,p3)
        if any(u_t < 0 for u_t in U_array)==True:
            p += 1
    p /= trials
    p = p*100
    PR3[i] = p
    i+=1

   # print("The probability of ruin approx will be ", p, "percent with portfolio type 3")
#plt.plot(np.linspace(0.1,1,10),PR3,label='Portfolio 3')
plt.errorbar(np.linspace(0.1,1,10),PR3,yerr=100/np.sqrt(1000),marker='*',label='Portfolio 3')
plt.legend()
fig1.savefig('PR123.png')


