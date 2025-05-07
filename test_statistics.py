from scipy import stats
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests
import pandas as pd

def convert_p_d_value_to_asterisks(p=None, d=None):
    if p is not None:
        return f'{convert_pvalue_to_asterisks(p)}'
    # if d is not None:
    #     return f'{convert_cohen_d_to_asterisks(d)}'
    return ' '

# 效应量与标记的转化关系
def convert_cohen_d_to_asterisks(d):
    d = abs(d)
    return ['N', 'S', 'M', 'L'][min(3, int(d > 0.2) + int(d > 0.5) + int(d > 0.8))]

# P值与星号的转化关系
def convert_pvalue_to_asterisks(p):
    return ['n.s.', '*', '**', '***'][min(3, int(p <= 0.05) + int(p <= 0.01) + int(p <= 0.001))]




### 计算三组数据的单因素方差统计结果，使用Tukey进行事后校正
def statistical_analysis(pair_data1, pair_data2, pair_data3):
    f, p = stats.f_oneway(pair_data1, pair_data2, pair_data3)
    print(f"ANOVA F-value: {f}, p-value: {p}")
    print(stats.kruskal(pair_data1, pair_data2, pair_data3))
    # 如果p值小于显著性水平（例如0.05），则进行后续的多重比较
    if p < 0.05:
        # 进行Tukey的HSD多重比较
        tukey_result = stats.tukey_hsd(pair_data1, pair_data2, pair_data3)
        print(f"Tukey's HSD post-hoc test results:\n{tukey_result}")
        p = [tukey_result.pvalue[0][1], tukey_result.pvalue[0][2], tukey_result.pvalue[1][2]]
    else:
        p = [p] *3
    return p
    
def shapiro_test_and_ttest_u_test(array1, array2, alpha=0.05):
    # 统计检验，先判断是否符合正态分布，然后再做相应的统计检验
    # 对每个数组进行正态性检验
    normal_test1 = stats.shapiro(array1)
    normal_test2 = stats.shapiro(array2)
    print('p1:', normal_test1.pvalue, ' - ', 'p2', normal_test2.pvalue)

    # 判断两个数组是否都服从正态分布
    both_normal = normal_test1.pvalue > alpha and normal_test2.pvalue > alpha

    if both_normal:
        print("服从正态分布，执行双样本t检验")
        t_statistic, p_value = stats.ttest_ind(array1, array2, equal_var=False)
        test_type = "t-test"
    else:
        print("不服从正态分布，执行Mann-Whitney U检验")
        u_statistic, p_value = stats.mannwhitneyu(array1, array2)
        test_type = "Mann-Whitney U test"
    print(test_type, 'p_value:', p_value)
    return test_type, p_value


def samples_test(pair_data1, pair_data2, alpha=0.05,test_mode = 'pair'):
    # 计算配对样本的差值
    differences = np.array(pair_data2) - np.array(pair_data1)

    # 进行Shapiro-Wilk正态性检验
    _, shapiro_pvalue = stats.shapiro(differences)
    print(stats.shapiro(differences))

    # 如果差值符合正态分布
    if shapiro_pvalue > alpha:
        # 执行配对样本t检验
        if test_mode == 'pair':
            
            t_statistic, p_value = stats.ttest_rel(pair_data1, pair_data2)
            print(f"差值服从正态分布，进行配对样本t检验。\nT检验统计量为 {t_statistic}，p值为 {p_value:}")
            
        if test_mode == 'ind':
            t_statistic, p_value = stats.ttest_ind(pair_data1, pair_data2)
            print(f"差值服从正态分布，进行独立样本t检验。\nT检验统计量为 {t_statistic}，p值为 {p_value:}")
        return t_statistic, p_value

    else:
        # 如果差值不符合正态分布，执行Wilcoxon符号秩检验
        wilcoxon_statistic, p_value = stats.wilcoxon(pair_data1, pair_data2, zero_method='pratt')
        print(f"差值不服从正态分布，进行Wilcoxon符号秩检验。\nWilcoxon统计量为 {wilcoxon_statistic}，p值为 {p_value}")
        return wilcoxon_statistic, p_value

    
def FDR(p_values):
    '''多重p值校正
    p_values: 1D array_like
    'bh' is for Benjamini-Hochberg, 'by' is for Benjaminini-Yekutieli'''
    ps_adusted = stats.false_discovery_control(p_values, method = 'bh')
    
    return ps_adusted


def MultiComparison1(data1,data2,data3,title):
    # list_y = ['young']*len(data_y)
    # list_m = ['middle']*len(data_m)
    # list_o = ['old']*len(data_o)
    '''多重样本检验'''
    list1 = [1] * len(data1)
    list2 = [2] * len(data2)
    list3 = [3] * len(data3)
    data_1 = [list1,data1]
    data_2 = [list2,data2]
    data_3 = [list3,data3]
    df_y = pd.DataFrame(np.array(data_1).T, columns=['type','value'])
    df_m = pd.DataFrame(np.array(data_2).T, columns=['type','value'])
    df_o = pd.DataFrame(np.array(data_3).T, columns=['type','value'])
    df = pd.concat([df_y, df_m, df_o], axis=0)
    model = ols('value~C(type)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    print(title)
    print(anova_table)
    mc = MultiComparison(df['value'], df['type'])
    # Tukey's range test to compare means of all pairs of groups
    tukey_result = mc.tukeyhsd(alpha=0.05)
    
    print(tukey_result)
    pvalue = tukey_result.pvalues
    p = {title:pvalue}
    return pvalue,p