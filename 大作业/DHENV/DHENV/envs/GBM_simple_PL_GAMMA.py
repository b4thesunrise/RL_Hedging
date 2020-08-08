import math
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import describe

import gym
from gym import spaces
from gym.utils import seeding

#发现两处错误：
# 1. maturity的单位有问题————__init__, reset, step
# 2. 计算公式里面忘了除单位—— bsput, bscall
# 3. reward里面多头 / 空头写反，修改step
# 4. reward写的有大问题，修改step
# 5. 原来的PL没有金融意义，改成账户价值，修改init，step
# 6. 打印的要修改，修改show
# 7. env.actions 没有reset，修改reset

class GBM_simple_PL_GAMMA(gym.Env):
    
    def __init__(self, std = 0.3, mean = 0.2, T = 100, S = 10, strike = 8, riskfree = 0.01, dividen = 0, deltat = 0.001, transac = 0.01):
        self.std = std
        self.mean = mean
        self.maturity = T * deltat
        self.maturity_const = T * deltat
        self.strike = strike
        self.riskfree = riskfree
        self.dividen = dividen
        self.deltat = deltat
        self.transac = transac
        self.count = 0

        self.S = S
        self.S_const = S
        self.prices = [S]
        
        self.reward = 0
        self.rewards = []

        self.stock_number = 0
        self.stock_numbers = [0]  # 我认为不应该先放一个零进去，应该让该表的长度与action、reward表相同，毕竟调仓(action)和选择持有的仓位其实是一回事
        # 如果后续需要打印本表或者作图，与action、reward表长度相同也是方便对应的，长度都为T。

        c = self.bscall()
        p = self.bsput()

        self.callprices = [c]
        self.putprices = [p]
        
        self.saving = c[0]
        self.savings = [c[0]]  # 我认为第一个元素应该是c，因为此时你卖了一份期权，收到了那么多钱。
        # 如果认为agent会考虑money account，它最初面临的就是这么多钱

        self.Account = - c[0] + self.stock_number * self.S + self.saving # 最好认为初始值为0，即-c+c，这个影响不大
        self.Accounts = [self.Account]

        self.seednumber = self.seed()

        self.actions = []

        self.action_space = spaces.Box(low = -np.inf, high = np.inf, shape = (1,), dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (4,),dtype=np.float32)#asset: stock, bank, stockprice, maturity
    
    def bscall(self):
        '''
        bsCall <- function(s, K, sigma, t, r=0, d=0){
        d1 <- (log(s/K) + (r - d)*t)/(sigma*sqrt(t)) + sigma*sqrt(t)/2
        d2 <- d1 - sigma*sqrt(t)
    
        c <- s*exp(-d*t)*pnorm(d1) - K*exp(-r*t)*pnorm(d2)
        delta <- exp(-d*t)*pnorm(d1)
        Gam <- dnorm(d1)/s/sigma/sqrt(t)
    
        data.frame(c, delta, Gam)
        }
        '''
        d1 = ( math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.maturity ) / self.std / math.sqrt(self.maturity) + self.std * math.sqrt(self.maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.maturity)
        c = self.S * math.exp(-self.dividen*self.maturity) * stats.norm.cdf(d1) - self.strike * math.exp(-self.riskfree*self.maturity) * stats.norm.cdf(d2)
        delta = math.exp(-self.dividen*self.maturity) * stats.norm.cdf(d1)
        Gam = stats.norm.pdf(d1)/self.S/self.std/math.sqrt(self.maturity)
        return c, delta, Gam, d1, d2


    def bsput(self):
        d1 = ( math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.maturity )  / self.std / math.sqrt(self.maturity) + self.std * math.sqrt(self.maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.maturity)
        p = self.strike * math.exp(-self.riskfree*self.maturity) * stats.norm.cdf(-d2) - self.S * math.exp(-self.dividen*self.maturity) * stats.norm.cdf(-d1) 
        delta = -math.exp(-self.dividen*self.maturity) * stats.norm.cdf(-d1)
        Gam = stats.norm.pdf(d1)/self.S/self.std/math.sqrt(self.maturity)
        return p, delta, Gam


    def GBMmove(self):
        dW = np.random.randn(1)[0] * math.sqrt(self.deltat)
        self.S = self.S + self.S * (dW * self.std + self.mean * self.deltat)
        # 此处并非几何布朗运动


    def step(self, action):
        #i时刻，决定买多少，付钱，得到股票
        if len(self.callprices) == 1:
            stock_add = self.callprices[-1][2]
        else:
            stock_add = self.callprices[-1][2] - self.callprices[-2][2]
        self.actions.append(stock_add)
        stock_money = stock_add * self.S
        self.saving -= stock_money
        self.saving -= abs(stock_money) * self.transac#减少transac
        self.stock_number += stock_add
        self.stock_numbers.append(self.stock_number)
        #i到i+1时刻，股票走动，利息滚动，时间减少
        self.GBMmove()
        self.maturity -= self.deltat
        self.maturity = max(self.maturity, 1e-15)
        saving_reward = self.saving * (math.exp(self.riskfree * self.deltat) - 1)
        self.saving += saving_reward  # 指数上不应该有负号！！！存银行里的钱怎么能变少
        self.savings.append(self.saving)
        self.prices.append(self.S)
        self.callprices.append(self.bscall())
        self.putprices.append(self.bsput())
        #i+1时刻，记录account
        self.Account = -self.callprices[-1][0] + self.savings[-1] + self.prices[-1] * self.stock_numbers[-1]
        self.Accounts.append(self.Account)
        #i+1时刻，得到0时刻行动的收益，资产价格变动 + 行动带来的收益变动 + transac + 银行存款收益
        self.reward = ( -self.callprices[-1][0] - (-self.callprices[-2][0]) ) + (self.prices[-1] - self.prices[-2]) * action - abs(stock_money) * self.transac + saving_reward
        # 如同讨论的那样，应该为stock_numnbers[-1]
        # 另外我认为应该加入money account的改变量，文献中没有这一项，因为它认为无风险利率为0，
        # 应该奖励股市不利的时候将钱存在银行的行为。

        done = False
        if self.maturity <= 1e-15:
            done = True
            self.count += 1
            if self.count % 10 == 1:
                #self.show()
                pass
        if done:
            self.reward += (-self.transac * self.stock_number * self.S)#在最后一步的时候卖掉所有股票
        self.rewards.append(self.reward)
        self.construct_state()
        
        # 文献里说，最后一步必须是出手所有股票，所以reward也需要特殊设置。

        return self.state, self.reward, done, {}
    
    def construct_state(self):
        self.state = self.state = np.array([self.stock_number, self.saving, self.S, self.maturity], dtype=np.float32)
        # 不太了解这里的语法qwq
        # 这里有个问题，返回的state应该是下一次action的依据，
        # 而这里的self.saving只是一次action之后的结果，和下次行动时面临的saving差利息
        # 觉得你saving更新的位置有点问题

    def show(self):
        plt.plot(self.actions, label = 'rewards')
        plt.show()
        plt.plot(self.Accounts, label = 'Accounts')
        plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # 这里没有维护Accounts

        self.maturity = self.maturity_const

        self.S = self.S_const
        self.prices = [self.S]
        
        self.reward = 0
        self.rewards = []

        self.stock_number = 0
        self.stock_numbers = [0]  # 我认为不应该先放一个零进去，应该让该表的长度与action、reward表相同，毕竟调仓(action)和选择持有的仓位其实是一回事
        # 如果后续需要打印本表或者作图，与action、reward表长度相同也是方便对应的，长度都为T。

        c = self.bscall()
        p = self.bsput()

        self.saving = c[0]
        self.savings = [self.saving]  # 我认为第一个元素应该是c，因为此时你卖了一份期权，收到了那么多钱。
        # 如果认为agent会考虑money account，它最初面临的就是这么多钱

        self.callprices = [c]
        self.putprices = [p]

        self.Account = - c[0] + self.stock_number * self.S + self.saving # 最好认为初始值为0，即-c+c，这个影响不大
        self.Accounts = [self.Account]
        
        self.actions = []

        self.construct_state()
        
        return self.state

