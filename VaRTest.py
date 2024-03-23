# 必要库 libs
import numpy as np
import pandas as pd
from scipy.stats import chi2

class ExpectileReg:
    """
    简介Intro
    ----------------------------
    该ExpectileReg类主要用于对金融产品测算的VaR进行检验，包括Kupiec及Engle检验。\n
    The class is mainly used to verify the VaR estimation of financial instruments, including Kupiec Test and Engle Test.\n

    实例形参Param
    ----------------------------
    yields:np.array -> 金融工具原始收益率; The original rate of return on a certain financial instrument\n
    var:np.array -> 金融工具VaR; The VaR of a certain financial instrument\n
    method:str -> 模型选择（可选）'KupiecTUFF'、'KupiecPF'、'EngleIS'、'EngleOS'; Choosing Model(Select)'KupiecTUFF'、'KupiecPF'、'EngleIS'、'EngleOS'\n
    theta:float -> VaR覆盖率，缺省值0.05; The coverage of VaR, default 0.05\n
    lag:int -> 滞后阶数，缺省值1; lag, default 1\n

    调用Use
    ----------------------------
    ExpectileReg.process(self) -> None\n

    附言Other
    ----------------------------
    注意Att：输出打表; Print the certain output\n
    注意Att：该检验会自动按照VaR的长度识别样本内与样本外，不必人工区分（前VaR个长度的原始收益率为样本内，后VaR个长度的原始收益率为样本外）。 No need for artificial division \n
    """

    def __init__(self, yields:np.array, var:np.array, method:str = 'KupiecTUFF', theta:float = 0.05, lag:int = 1) -> any:
        self.yields:np.array = yields
        self.var_level:float = 1 - theta
        self.lag:int = lag
        self.var:np.array = var
        self.method:str = method

    # 样本内比较的Engle检验（DQ_IS：样本内DQ）
    def __engleis_test(self) -> any:
        yields = self.yields[0:len(self.var)]
        hit = np.ones(len(yields))
        var_level = 1 - self.var_level
        hit[yields < self.var] = 1 - var_level
        hit[yields > self.var] = 0 - var_level
        hit_matrix = np.zeros((len(yields) - self.lag, self.lag))
        yields_lag = hit[self.lag:len(yields)]
        ind = 1
        while ind <= self.lag:
            hit_matrix[:, ind - 1] = hit[ind - 1:(len(yields) - self.lag + ind - 1)]
            ind += 1
        x_ini = np.ones(len(hit_matrix))
        x = np.column_stack((x_ini, hit_matrix, self.var[self.lag:len(yields)]))
        if np.linalg.det(np.dot(x.T,x)) == 0:
            raise ValueError(f'xTx：{np.shape(np.dot(x.T,x))}是奇异的。')
        # 计算xTx的逆
        xtx_inv = np.linalg.inv(np.dot(x.T,x))
        # 计算ols闭集系数
        ols_par = xtx_inv @ x.T
        ols_par = ols_par @ yields_lag
        # 计算样本内DQ检验的统计量（DQ_IS）
        denominator = var_level * (1 - var_level)
        statistics = (ols_par.T @ x.T @ x @ ols_par) / denominator
        # 注意自由度为滞后阶数加2
        p_value = chi2.sf(statistics, df=(self.lag + 2))
        return statistics, p_value

    # 样本外比较的Engle检验（DQ_OS：样本外DQ）
    def __engleos_test(self) -> tuple:
        # 原收益率yields与计算出的VaR截取相同长度便于比较
        yields = self.yields[-len(self.var):]
        # 初始化hit：hit表示击中（即VaR覆盖失败）
        hit = np.zeros(len(yields))
        # 置信水平：var_level
        var_level = 1 - self.var_level
        hit[yields < self.var] = 1 - var_level
        hit[yields > self.var] = 0 - var_level
        # 创建hit与VaR当前项与滞后项
        hit_ahead = hit[self.lag:len(yields)]
        var_ahead = self.var[self.lag:len(yields)]
        hit_lag = np.zeros((len(yields) - self.lag, self.lag))
        yields_lag = yields[(self.lag - 1):(len(yields) - 1)]
        yields_lag = yields_lag ** 2
        ind = 1
        while ind <= self.lag:
            hit_lag[:, ind - 1] = hit[ind - 1:(len(yields) - (self.lag - ind + 1))]
            ind += 1
        # 注意hit滞后的长度要获取行数
        min_len = min(len(var_ahead), int(hit_lag.shape[0]), len(yields_lag))
        # 初始化x
        x_ini = np.ones(min_len)
        # 更新其它变量
        var_ahead = var_ahead[-min_len:]
        hit_lag = hit_lag[-min_len:]
        yields_lag = yields_lag[-min_len:]
        hit_ahead = hit_ahead[-min_len:]
        # 计算x
        x = np.column_stack((x_ini, var_ahead, hit_lag, yields_lag))
        if np.linalg.det(np.dot(x.T,x)) == 0:
            raise ValueError(f'xTx：{np.shape(np.dot(x.T,x))}是奇异的。')
        # 计算xTx的广义逆
        xtx_inv = np.linalg.pinv(np.dot(x.T, x))
        # 计算样本外DQ检验的统计量（DQ_OS）
        denominator = var_level * (1 - var_level)
        statistics = (hit_ahead.T @ x @ xtx_inv @ x.T @ hit_ahead) / denominator
        # 注意自由度是x的列数
        p_value = chi2.sf(statistics, df=(self.lag + 2))
        return statistics, p_value

    # Kupiec检验(仅适用样本外比较)
    def __kupiec_test(self) -> tuple:
        # 原收益率yields与计算出的VaR截取相同长度便于比较
        yields = self.yields[-len(self.var):]
        # 初始化hit：hit表示击中（即VaR覆盖失败）
        hit = np.zeros(len(yields))
        hit[yields < self.var] = 1
        # 找到第一次失败
        first_fail = np.where(hit == 1)[0]
        if len(first_fail) > 0:
            first_fail = first_fail[0]
        else:
            first_fail = None
        # 统计所有失败次数
        fails = sum(hit)
        # 成功概率
        p = 1 - self.var_level
        if self.method == 'KupiecPF':
            # 计算KupiecPF统计量
            statistics = -2 * np.log((((1 - p) ** (len(yields) - fails)) * (p ** fails)) / ((1 - fails / len(yields)) ** (len(yields) - fails) * (fails / len(yields)) ** fails))
            # 注意自由度为1
            p_value = chi2.sf(statistics, df=1)
            return statistics, p_value
        elif self.method == 'KupiecTUFF':
            if first_fail is None:
                raise UserWarning(f'{type(self.method)}不适用KupiecTUFF检验。')
            # 计算KupiecTUFF统计量
            statistics = -2 * np.log((p * (1 - p) ** (first_fail - 1)) / ((1 / first_fail) * (1 - 1 / first_fail) ** (first_fail - 1)))
            # 注意自由度为1
            p_value = chi2.sf(statistics, df=1)
            return statistics, p_value

    def process(self) -> None:
        if self.method == 'EngleIS':
            statistics_is, p_value_is = self.__engleis_test()
            print(f'--At {100 - int(self.var_level * 100)}% VaR coverage--\nDQ-Statistics(in sample): {np.round(statistics_is, decimals=4)}\nDQ-Pvalue(in sample): {np.round(p_value_is, decimals=8)}')
        elif self.method == 'EngleOS':
            statistics_os, p_value_os = self.__engleos_test()
            print(f'--At {100 - int(self.var_level*100)}% VaR coverage--\nDQ-Statistics(out of sample): {np.round(statistics_os, decimals=4)}\nDQ-Pvalue(out of sample): {np.round(p_value_os, decimals=8)}')
        elif self.method == 'KupiecPF':
            statistics_pf, p_value_pf = self.__kupiec_test()
            print(f'--At {100 - int(self.var_level * 100)}% VaR coverage--\nLR-Statistics(proportion of failures): {np.round(statistics_pf, decimals=4)}\nLR-Pvalue(proportion of failures): {np.round(p_value_pf, decimals=8)}')
        elif self.method == 'KupiecTUFF':
            statistics_tuff, p_value_tuff = self.__kupiec_test()
            print(f'--At {100 - int(self.var_level * 100)}% VaR coverage--\nLR-Statistics(time until first failure): {np.round(statistics_tuff, decimals=4)}\nLR-Pvalue(time until first failure): {np.round(p_value_tuff, decimals=8)}')
        else:
            raise AttributeError(f'找不到名为{self.method}的模型。')

# 调用示范
if __name__ == '__main__':
    # 生成原始收益率的测试数据 Yields for Test
    yields_input = np.random.normal(0, 4, 2000)
    # 表示5%分位数的VaR覆盖水平 VaR coverage for Test
    var_temp = np.quantile(yields_input, 0.05)
    # 生成VaR的测试数据(前半部分作为样本内；后半部分作为样本外) VaR for Test（first half in sample; last half out of sample）
    var_input = np.repeat(var_temp, 0.5 * len(yields_input))
    # 实例化 Initialization
    reg_one = ExpectileReg(yields=yields_input, var=var_input, method='KupiecTUFF')
    # 执行检验 Run
    reg_one.process()
