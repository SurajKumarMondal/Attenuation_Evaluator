import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

sand_25RD_1m = pd.read_excel("25_RD_Sand_1m.xlsx")
sand_25RD_1m = sand_25RD_1m.dropna()
sand_25RD_1m = sand_25RD_1m.reset_index(drop = True)


sand_25RD_1m["D1_norm"] = sand_25RD_1m["D_1"] / sand_25RD_1m["In_Air"]
sand_25RD_1m["D2_norm"] = sand_25RD_1m["D_2"] / sand_25RD_1m["In_Air"]
sand_25RD_1m["D3_norm"] = sand_25RD_1m["D_3"] / sand_25RD_1m["In_Air"]
sand_25RD_1m["D4_norm"] = sand_25RD_1m["D_4"] / sand_25RD_1m["In_Air"]


m_list_43RD_H = []
m_list_43RD =[]
m_list_25RD = []
R_25RD = []
R_43RD = []
R_43RD_H = []

x = [.066, .141, .216, .291]

X = np.arange(0,0.3,0.001)

# Define the function to fit
def func(x, m):
    x = np.array(x)  # Convert x to a numpy array
    return np.exp(m * x)

for i in range(len(sand_25RD_1m)):
    y_norm = []
    y_norm.append(sand_25RD_1m['D1_norm'][i])
    y_norm.append(sand_25RD_1m['D2_norm'][i])
    y_norm.append(sand_25RD_1m['D3_norm'][i])
    y_norm.append(sand_25RD_1m['D4_norm'][i])

    # Define the data
    y = [sand_25RD_1m['D1_norm'][i], sand_25RD_1m['D2_norm'][i], 
    sand_25RD_1m['D3_norm'][i], sand_25RD_1m['D4_norm'][i]]


    # Fit the curve to the data
    popt, pcov = curve_fit(func, x, y, p0=[-1])

    # Get the value of m and R-squared
    m = popt[0]
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    m_list_25RD.append(m)
    R_25RD.append(r_squared)

    # Print the value of m and R-squared
    print(f'm = {m:.4f}')
    print(f'R-squared = {r_squared:.4f}')

sand_25RD_1m.insert(11, 'm', m_list_25RD)
sand_25RD_1m.insert(12, 'r_squared', R_25RD)
