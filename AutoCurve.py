# -*- coding: utf-8 -*-
from inspect import Attribute
from itertools import combinations
from os import stat
import matplotlib.pyplot as plt
import random as rd
import numpy as np
from numba import njit
from numba.typed import List
#np.random.seed(11)
#rd.seed(11)
from numpy import random
from numpy.testing._private.utils import check_free_memory
np.seterr(all="ignore")
import string
import warnings
import sympy as sp
#warnings.simplefilter("error")
import numbers
import cmath as math

import math as math_regular

class NumberWithText:
    def __init__(self, num, text=None,opr = 0):
        self.opr_count = opr
        self.num = num
        if text is None:
            self.text = f'{num}'
        else:
            self.text = text
    
    @property
    def get_opr_plus(self):
        return self.opr_count+1

    def __repe__(self):
        #self.inc_opr()
        return f'{self.text}'

    def __str__(self):
        return f'{self.text}'

    def __add__(self, b):
        try:
            num = self.num + b.num
        except:
            num = np.nan
        return NumberWithText(num, f'({self.text}+{b.text})',opr = self.get_opr_plus)

    def __mul__(self, b):
        try:
            num = self.num * b.num
        except:
            num = np.nan
        return NumberWithText(num, f'({self.text}*{b.text})',opr = self.get_opr_plus)

    def __sub__(self, b):
        try:
            num = self.num - b.num
        except:
            num = np.nan
        return NumberWithText(num, f'({self.text}-{b.text})',opr = self.get_opr_plus)

    def __truediv__(self, b):
        try:
            num = self.num/b.num
        except ZeroDivisionError:
            num = np.nan
        return NumberWithText(num, f'({self.text}/{b.text})',opr = self.get_opr_plus)

    def __pow__(self, b):
        try:
            num = self.num**b.num
        except:
            #num = self.num**b.num
            num = np.nan

        return NumberWithText(num, f'({self.text}**{b.text})',opr = self.get_opr_plus)
    
    def abs(self):
        try:
            num = np.abs(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(abs({self.text}))',opr = self.get_opr_plus)
    
    def sin(self):
        try:
            num = np.sin(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(sin({self.text}))',opr = self.get_opr_plus)

    def cos(self):
        try:
            num = np.cos(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(cos({self.text}))',opr = self.get_opr_plus)
    
    def tan(self):
        try:
            num = np.tan(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(tan({self.text}))',opr = self.get_opr_plus)
    
    def exp(self):
        try:
            num = np.exp(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(exp({self.text}))',opr = self.get_opr_plus)
    
    def ln(self):
        try:
            num = np.log(self.num)
        except:
            num = np.nan
        return NumberWithText(num, f'(log({self.text}))',opr = self.get_opr_plus)

    def factorial(self):
        return NumberWithText(math_regular.factorial(self.num), f'({self.text}!)',opr = self.get_opr_plus)

    def sqrt(self):
        try:
            num = np.sqrt(self.num)
        except:
            print(self.num)
        
        return NumberWithText(num, f'(sqrt({self.text}))',opr = self.get_opr_plus)



def double_plus_combi(N, max_n=4):
    ans = []
    if N == 0:
        return 0
    return ans


def anything_is_closet(quest, ans, g):
    sortv = []
    for i in quest:
        if i.num == "x":
            continue
        #dist_phd = abs(i.num-ans.num)
        dist_phd = find_dist(ans.num,i.num)
        sortv.append(SortObj(i, i, dist_phd))
    p = sorted(sortv)
    #g = rd.randint(0, len(p)-1)
    g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    if len(p) == 0:
        return None, None
    if p[g].subset is None:
        return None, p[g].value
    return [p[g].subset], p[g].value


def factorial(N):
    if N > 10 or isinstance(N, numbers.Integral):
        return None
    if N.num == 1:
        return NumberWithText(1)
    return N*factorial(N-NumberWithText(1))


def anything_factorial_closet(quest, ans, g,prob_of_use_x):
    sortv = []
    for i in quest:
        if isinstance(i.num,complex) or (i.num > 7 or i.num < 0 or not isinstance(i.num, numbers.Integral) or i.num == 1 or i.num == 2):
            continue
        abc = i.factorial()
        if abc is None:
            continue
        #dist_phd = abs(abc.num-ans.num)
        dist_phd = find_dist(ans.num,abc.num)
        sortv.append(SortObj(abc, i, dist_phd))
    p = sorted(sortv)
    #print(p)
    if len(p) == 1:
        g = rd.randint(0, len(p)-1)
    else:
        g = 0
    #print(p)
    #g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    if len(p) == 0:
        return None, None
    #print(g)
    if p[g].subset is None:
        return None, None
    return [p[g].subset], p[g].value

def generate_random_choice(length,k = rd.randint(1,11)):
    if length == 0:
        return [1]
    remain_prob = 1
    prob = []
    for _ in range(length):
        i_prob = remain_prob/k
        remain_prob -= i_prob
        prob.append(i_prob)
    prob[0] += remain_prob
    #print(prob,length)
    #for i in range(0,length+1):
        #print(i)
    return prob


def anything_plus_closet(quest, ans, g,prob_of_use_x):
    sortv = []
    for i in range(2, 3):
        combi = list(combinations(quest, i))
        for subset in combi:
            subset = list(subset)
            #vas = reduce(lambda x, y: x+y, subset)
            #vas = subset[0] + subset[1]
            #case ans = x + n ;x = ans - n
            x_ind = locate_x(subset)
            if x_ind is not None:
                if rd.randint(0,100) <= prob_of_use_x:
                    x = math.nan
                else:
                    n_ind = x_ind-1

                    x = np.mean(ans.num - subset[n_ind].num)
                    subset[x_ind] = NumberWithText(x,text=str(x))
                #print(subset[x_ind].num)
            
            vas = subset[0] + subset[1]

            #dist_phd = abs(ans.num - vas.num)
            dist_phd = find_dist(ans.num,vas.num)
            sortv.append(SortObj(vas, subset, dist_phd))
    p = sorted(sortv)
    try:
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    except:
        g = 0
    if len(p) == 0:
        return None, None
    if p[g].subset is None:
        return None, p[g].value
    return [*(p[g].subset)], p[g].value


def anything_mul_closet(quest, ans, g,prob_of_use_x):
    sortv = []
    for i in range(2, 3):
        combi = list(combinations(quest, i))
        for subset in combi:
            subset = list(subset)
            #vas = reduce(lambda x, y: x*y, subset)
            #vas = subset[0] * subset[1]
            #case ans = x * n ; x = ans/n
            x_ind = locate_x(subset)
            if x_ind is not None:
                if rd.randint(0,100) <= prob_of_use_x:
                    x = math.nan
                else:
                    n_ind = x_ind-1

                    x = np.mean(ans.num/subset[n_ind].num)
                    subset[x_ind] = NumberWithText(x,text=str(x))
            
            vas = subset[0] * subset[1]
            #dist_phd = abs(ans.num - vas.num)
            dist_phd = find_dist(ans.num,vas.num)
            sortv.append(SortObj(vas, subset, dist_phd))
    p = sorted(sortv)
    try:
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    except:
        g = 0
    if len(p) == 0:
        return None, None
    if p[g].subset is None:
        return None, p[g].value
    return [*(p[g].subset)], p[g].value


def special_exp(subset,OC = False):
    ans = subset[0]
    for i in subset[1:]:
        #if not OC and (abs(i.num) == 0 or abs(ans.num) > 20 or abs(i.num) > 10 or (ans.num == 0 and i.num <= 0)):
            #return None
        ans = ans**i
        #if isinstance(ans.num, complex):
            #return None
    return ans


def anything_exp_closet(quest, ans, g,prob_of_use_x):
    sortv = []
    for i in range(2, 3):
        combi = list(combinations(quest, i))
        for subset in combi:
            subset = list(subset)
            #vas = special_exp(subset,OC = True)
            #vas = subset[0] ** subset[1]
            x_ind = locate_x(subset)
            if x_ind is not None:
                if rd.randint(0,100) <= prob_of_use_x:
                    x = math.nan
                else:
                    n_ind = x_ind-1
                    #case ans = x ** n ; x = ans**(1/n)
                    if x_ind == 0:
                        x = np.mean(ans.num**((1/subset[n_ind].num)))
                    #case nas = n ** x ; x = log n a 
                    else:
                        x = np.mean((np.log(ans.num)/np.log(subset[n_ind].num)))
                    subset[x_ind] = NumberWithText(x,text=str(x))
            vas = subset[0] ** subset[1]
            #print(vas)
            if vas is None or vas is math.nan:
                continue
            #dist_phd = abs(ans.num - vas.num)
            dist_phd = find_dist(ans.num,vas.num)
            sortv.append(SortObj(vas, subset, dist_phd))
    p = sorted(sortv)
    try:
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    except:
        g = 0
    if len(p) == 0:
        return None, None
    return [*(p[g].subset)], p[g].value

def anything_sqrt_closet(quest, ans, g,prob_of_use_x,OC = True):
    sortv = []
    for i in quest:
        if i.num == "x":
            continue
        #if not (i.num == 1 or i.num == 0) and (not OC and (i.num > 7 or i.num < 0 or not isinstance(i.num, numbers.Integral))):
            #continue
        abc = i.sqrt()
        if abc is None or abc is math.nan:
            continue
        dist_phd = find_dist(ans.num,abc.num)
        #dist_phd = abs(abc.num-ans.num)
        sortv.append(SortObj(abc, i, dist_phd))
    p = sorted(sortv)
    #g = rd.randint(0, len(p)-1)
    g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    if len(p) == 0:
        return None, None
    if p[g].subset is None:
        return None, None
    return [p[g].subset], p[g].value

def create_function_closet(function):
    def anything_function_closet(quest, ans, g,prob_of_use_x,OC = True):
        sortv = []
        for i in quest:
            if i.num == "x":
                continue
            #if not (i.num == 1 or i.num == 0) and (not OC and (i.num > 7 or i.num < 0 or not isinstance(i.num, numbers.Integral))):
                #continue
            abc = function(i)
            if abc is None or abc is math.nan:
                continue
            dist_phd = find_dist(ans.num,abc.num)
            #dist_phd = abs(abc.num-ans.num)
            sortv.append(SortObj(abc, i, dist_phd))
        p = sorted(sortv)
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
        if len(p) == 0:
            return None, None
        if p[g].subset is None:
            return None, None
        return [p[g].subset], p[g].value
    return anything_function_closet

def anything_minus_closet(quest, ans, g,prob_of_use_x, invoker = True):
    sortv = []
    if invoker and rd.randint(0,100) > 70:
        pass
    for i in range(2, 3):
        combi = list(combinations(quest, i))
        for subset in combi:
            subset = list(subset)
            x_ind = locate_x(subset)
            if x_ind is not None:
                if rd.randint(0,100) <= prob_of_use_x:
                    x = math.nan
                else:
                    n_ind = x_ind-1
                    #case ans = x - n ; x = ans + n
                    if x_ind == 0:
                        x = np.mean(ans.num+subset[n_ind].num)
                    #case ans = n - x ; x = n - ans
                    else:
                        x = np.mean(subset[n_ind].num - ans.num)
                    subset[x_ind] = NumberWithText(x,text=str(x))
            #vas = reduce(lambda x, y: x-y, subset)
            vas = subset[0] - subset[1]
            #dist_phd = abs(ans.num - vas.num)
            dist_phd = find_dist(ans.num,vas.num)
            sortv.append(SortObj(vas, subset, dist_phd))
    p = sorted(sortv)
    try:
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    except:
        g = 0
    if len(p) == 0:
        return None, None
    if p[g].subset is None:
        return None, p[g].value
    return [*(p[g].subset)], p[g].value


def special_div(subset):
    ans = subset[0]
    for i in subset[1:]:
        #if i.num == 0:
            #return None
        ans = ans/i
    return ans


class SortObj:
    def __init__(self, value, subset, dis):
        self.value = value
        self.dis = dis
        self.subset = subset

    def __lt__(self, other):
        return self.dis < other.dis

    def __eq__(self, other):
        return self.dis == other.dis

def locate_x(quest_list):
    for i in range(len(quest_list)):
        if quest_list[i].num == "x":
            return i

def anything_div_closet(quest, ans, g,prob_of_use_x):
    sortv = []
    for i in range(2, 3):
        combi = list(combinations(quest, i))
        for subset in combi:
            subset = list(subset)
            x_ind = locate_x(subset)
            if x_ind is not None:
                if rd.randint(0,100) <= prob_of_use_x:
                    x = math.nan
                else:
                    n_ind = x_ind-1
                    #case ans = x / n ; x = ans * n
                    if x_ind == 0:
                        x = np.mean(ans.num*subset[n_ind].num)
                    #case ans = n / x ; x = n - ans
                    else:
                        x = np.mean(subset[n_ind].num / ans.num)
                subset[x_ind] = NumberWithText(x,text=str(x))
            vas = subset[0] / subset[1]
            if vas is None:
                continue
            #dist_phd = abs(ans.num - vas.num)
            dist_phd = find_dist(ans.num,vas.num)
            sortv.append(SortObj(vas, subset, dist_phd))
    p = sorted(sortv)
    try:
        #g = rd.randint(0, len(p)-1)
        g = rd.choices(range(len(p)),generate_random_choice(len(p)))[0]
    except:
        g = 0
    if len(p) == 0:
        return None, None
    return [*(p[g].subset)], p[g].value


def list_of_np_to_list_of_list(elements):
    target = elements.copy()
    return target

class Function:
    def __init__(self,eq):
        self.eq_original = eq
        self.eq = self.replace_system(eq)
        self.variable_list = []
        for i in self.eq:
            if i in string.ascii_uppercase:
                self.variable_list.append(i)
        self.variable_list = set(self.variable_list)
        #print(f'{self.eq}',self.variable_list)
    @staticmethod
    def replace_system(eq : str):
        eq = Function.if_in_this_then_replace_by(eq,"sin","np.sin")
        eq = Function.if_in_this_then_replace_by(eq,"cos","np.cos")
        eq = Function.if_in_this_then_replace_by(eq,"tan","np.tan")
        eq = Function.if_in_this_then_replace_by(eq,"sqrt","np.sqrt")
        eq = Function.if_in_this_then_replace_by(eq,"abs","np.abs")
        eq = Function.if_in_this_then_replace_by(eq,"log","np.log")
        eq = Function.if_in_this_then_replace_by(eq,"exp","np.exp")
        return eq

    @staticmethod
    def if_in_this_then_replace_by(text,element,by):
        if element in text:
            return text.replace(element,by)
        return text

    def eval_eq(self,data:dict):

        if set(data.keys()) != self.variable_list:
            raise ValueError
        
        phd = self.eq
        for i in set(data.keys()):
            phd = phd.replace(i,f'{data[i]}')
            

        return eval(phd)
        
class IQ180_Solution:
    def __init__(self):
        self.running = False
        super().__init__()

    
    def clean_text(self,eq_cleaned):
        #print(self.init_quest)
        for i in range(len(self.init_quest)-1):
            element = str(self.init_quest[i])
            #print(element)
            eq_cleaned = eq_cleaned.replace(element,self.get_value_of_value(element))
        #print(eq_cleaned)
        #print(eq_cleaned)
        eq_cleaned = sp.parsing.sympy_parser.parse_expr(eq_cleaned)
        return eq_cleaned
    
    def get_value_of_value(self,replace):
        return self.element_get_key[replace]

    def set_start(self,quest : dict, ans,not_use_all = False,prob_of_use_x = 90,keep_best_variable_params = 4,keep_factor = 50, sol_nums=1,reset_params = 300, allow_plus=True, allow_minus=True, allow_mul=True, allow_div=True, allow_exp=True, allow_fac=False, allow_sqrt = True,allow_sin = True,allow_cos = True ,allow_tan = True,allow_expo = True ,allow_ln = True,allow_abs = True):
        self.quest = []
        self.prob_of_use_x = 100 - prob_of_use_x
        self.reset_param = reset_params
        self.key_by_order = quest.copy()
        self.element_get_key = {}
        self.keep_best_variable_params = keep_best_variable_params
        self.keep_factor = keep_factor

        for i in self.key_by_order.keys():
            self.element_get_key[str(self.key_by_order[i])] = i

        for i in quest.keys():
            self.quest.append(NumberWithText(quest[i]))
        #self.quest = [NumberWithText(i) for i in quest]
        self.quest.append(NumberWithText("x"))
        self.init_quest = self.quest.copy()
        self.ans = NumberWithText(ans)
        allow_arg = [allow_plus, allow_minus,
                 allow_mul, allow_div, allow_exp, allow_fac,allow_sqrt,allow_sin,allow_cos,allow_tan,allow_expo,allow_ln,allow_abs]
        self.found_solution = []
        self.sol_nums = sol_nums
        self.not_use_all = not_use_all
        self.packing = [self.quest]
        self.all_methods = [anything_plus_closet, anything_minus_closet, anything_mul_closet,
                   anything_div_closet, anything_exp_closet, anything_factorial_closet , anything_sqrt_closet,create_function_closet(NumberWithText.sin),create_function_closet(NumberWithText.cos),create_function_closet(NumberWithText.tan),create_function_closet(NumberWithText.exp),create_function_closet(NumberWithText.ln),create_function_closet(NumberWithText.abs)]
        self.methods = []
        for i in range(len(allow_arg)):
            if allow_arg[i] is True or allow_arg[i] == 1:
                self.methods.append(self.all_methods[i])
        self.allow_methods_id = range(len(self.methods))
        self.worked_text = []
        self.all_solution_equation = []
        self.worked = []
        self.wpe = 0
        self.running = True

    def check_if_in_packing(self,new_pack):
        pack = ([i.num for i in new_pack])
        pack = list_of_np_to_list_of_list(pack)
        #print(type())
        for packin in self.packing:
            packin = list_of_np_to_list_of_list(packin)
            count = 0
            #print(pack)
            for i in pack:
                if i in packin:
                    count += 1
            if count == len(packin):
                return True


        return False

    @njit
    def run_a_solution(self,accept_error=10e-5):
        shortest = math.inf
        rework = 0
        best_variable = []
        while True:
            #self.packing = [self.init_quest.copy()]
            go_out = False
            #rework = 0
            for quest in self.packing:
                rework += 1
                ms = [rd.sample(self.methods, len(self.methods))[0]]
                for m in ms:
                    used, equal_to = m(quest, self.ans, 0,self.prob_of_use_x)
                    self.wpe += 1
                    try:
                        if used is None or np.isnan(equal_to.num).any():
                            continue
                    except:
                        continue
                    MSE = find_dist(equal_to.num,self.ans.num)/self.ans.num.shape[0]
                    if MSE > shortest * self.keep_factor:
                        continue
                    rd.shuffle(self.packing)
                    #print(MSE,self.wpe,rework,len(self.packing))
                    if MSE < shortest and isinstance(equal_to.num,np.ndarray):
                        shortest = MSE
                        #print(shortest)
                        #plt.title(f'error : {MSE}\n{self.clean_text(equal_to.text)}')
                        #plt.plot(x,self.ans.num,"or")
                        #plt.plot(x,equal_to.num,"b")
                        #plt.show()
                        #plt.pause(0.5)
                        best_variable.append([equal_to,*self.init_quest])
                        #plt.clf()
                        #display.clear_output(wait=True)
                        print(MSE)
                        #print(shortest)
                    if (MSE<accept_error): #and (self.not_use_all or len(remain) == 0):
                        if equal_to.text not in self.worked_text:
                            eq_cleaned = self.clean_text(equal_to.text)
                            self.worked_text.append(eq_cleaned)
                            self.worked.append(equal_to)
                            self.solution = Function(str(eq_cleaned))
                            
                            print(str(eq_cleaned))
                            return self.solution

                    elif np.isnan(equal_to.num).any() is not True and isinstance(equal_to,NumberWithText):
                        new_pack = [equal_to,*self.init_quest]
                        if True:
                            self.packing.append(new_pack)
                            self.all_solution_equation.append(equal_to.num.tolist())
                            #print(MSE,len(self.packing),equal_to.opr_count)
                    rd.shuffle(self.packing)
                    #print(MSE,self.wpe,rework)
                    if len(self.packing) > self.reset_param:
                        best_variable = best_variable[-int(len(self.packing)/self.keep_best_variable_params):]
                        self.packing = [self.init_quest,*best_variable]
                        #go_out = True
                        #break
                    #print(len(lt_long))
                if go_out:
                    break

def find_dist(a,b):
    try:
        dis = np.sum(np.abs(a-b))
        #print(b)
    except:
        return np.inf
    return dis

#prob_of_use_x = 90
#prob_of_use_x = 100 - prob_of_use_x
v = np.linspace(0,10,100)
m = np.linspace(1,10,100)
ele = {"M":m,"V":v}
ans = m * (v ** 2) * 0.5


tool = IQ180_Solution()
tool.set_start(ele,ans,sol_nums=1,not_use_all=False,allow_fac=False,reset_params=300,prob_of_use_x= 90)
a = tool.run_a_solution(accept_error=0.00003)


print(a.eval_eq({"M":5,"V":2}))
#print(tool.worked_text)