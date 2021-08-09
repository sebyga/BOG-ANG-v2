
# %% Importing
import pyiast
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.optimize as optim
Rgas = 8.3145*1.2 # J/mol/K
def Arrhenius(dH,T,T_ref):
#    Rgas = 8.3145 # J/mol/K
    dH_J_ar = np.abs(np.array(dH))*1000 # from kJ/mol to J/mol (Positive number)
    T_ref_ar= np.array(T_ref)
    inner_exp = (1/T - 1/T_ref_ar)*dH_J_ar/Rgas*1.2
    ratio_T_Tref = np.exp(inner_exp)
    return ratio_T_Tref

# %% Massbal_iteration
def Massbal_interation(
    dq, dT, comp_in, pyiast_list,dH_ad, T_ref, init_var,
    P_in = 10, T_in = 300,
    Design = {'A': 3.1416, 'L':3,'rho_s':1000, 'epsi':0.4}):
    ## Assign the inputs
    A = Design['A']
    L = Design['L']
    rho_s = Design['rho_s']
    epsi = Design['epsi']
    n = len(pyiast_list)
    x= np.array(init_var)[0:n]
    q = np.array(init_var)[n:2*n]
    T_init = init_var[2*n]
    y_in = np.array(comp_in)
    T_fin = T_init + dT
    Penalty_T = 0
    if T_fin < 100:
        T_fin = 100
        Penalty_T = (100-T_fin)**2

    ## Injected mass (mol) ##
    del_m = rho_s*A*L*(1-epsi)*np.sum(dq) + epsi*A*L/Rgas*(P_in/T_fin - 1/T_init)
    ## Mol in gas phase (mol array) ##
    mi_new = y_in*del_m + x*epsi*A*L*P_in/Rgas/T_init*1E5 - rho_s*(1-epsi)*A*L*np.array(dq)
    ind_invalid = mi_new < 0
    ind_valid = ind_invalid == False
    penalty = 0
    if np.sum(ind_invalid) > 0:
        penalty = penalty + np.sum(mi_new[ind_invalid]**2)
        mi_new[ind_invalid] = 0
    x_new = mi_new / np.sum(mi_new)
    Arr = Arrhenius(dH_ad,T_fin, T_ref)
    P_norm = x_new*P_in*Arr
    ind_posi = mi_new > 0.01
    if np.sum(ind_posi) == 1:
        iso_tmp = np.array(pyiast_list)[ind_posi][0]
        q_new = np.zeros(len(q))
        q_new[ind_posi] = iso_tmp.loading(P_norm[ind_posi])
    elif np.sum(ind_posi) == 0:
        q_new = np.zeros(len(q))
        penalty = penalty + 100
        penalty = penalty*2
    else:
        try:
            q_new = np.zeros(len(q))
            q_new[ind_valid] = pyiast.iast(
                P_norm[ind_valid],
                np.array(pyiast_list)[ind_valid],
                warningoff = True)
        except:
            try:
                q_new = np.zeros(len(q))
                xmol_guess = np.array([1.72, 0.36])/(1.72+0.36)
                q_new[ind_valid] = pyiast.iast(
                    P_norm[ind_valid],np.array(pyiast_list)[ind_valid],
                    adsorbed_mole_fraction_guess = xmol_guess, warningoff = True)
            except:
                check_true = False
                for xx in np.linspace(0.8,0.99,20):
                    try:
                        q_new = np.zeros(len(q))
                        xmol_guess = np.array([xx, 1-xx])
                        q_new[ind_valid] = pyiast.iast(
                        P_norm[ind_valid],np.array(pyiast_list)[ind_valid],
                        adsorbed_mole_fraction_guess = xmol_guess, warningoff = True)
                        check_true = True
                        break
                    except:
                        continue
                if check_true:
                    q_new =  np.array([0,0])
                    print(P_norm[0])
                    print(dH_ad,T_fin, T_ref)
                    q_new[0] = pyiast_list[0].loading(P_norm[0])
    del_q = q_new - q + Penalty_T
    return del_m, del_q, penalty, ind_valid


# %% Heatbal_iteration
def Heatbal_interation(
    del_m_in, dq, dT, comp_in, pyiast_list,dH_ad, T_ref, init_var,
    Cp_solid = 948, Cp_gas = [40.63, 29.22], # Solid = J/kg/K , gas = J/mol/K
    P_in = 10, T_in = 300,
    Design = {'A': 3.1416, 'L':3,'rho_s':1000, 'epsi':0.4}):
    ## Assign the inputs
    A = Design['A']
    L = Design['L']
    rho_s = Design['rho_s']
    epsi = Design['epsi']
    n = len(pyiast_list)
    x= np.array(init_var)[0:n]
    q = np.array(init_var)[n:2*n]
    T_init = init_var[2*n]
    y_in = np.array(comp_in)
    T_fin = T_init + dT

    ## Mass of solid and gas
    m_solid = rho_s*(1-epsi)*A*L # kg for Cp_solid (J/kg/K)
    m_gas_init = rho_s*(1-epsi)*A*L*np.array(q) + A*L*epsi*P_in/Rgas/T_init*1E5*x # [mol, mol, ..] array for Cp_gas (J/mol/K)
    Cp_av_sys = m_solid*Cp_solid + np.sum(Cp_gas*m_gas_init) # J/K
    Cp_av_in = np.sum(del_m_in*y_in*Cp_gas) # J/K
    Heat_ad = rho_s*(1-epsi)*A*L*np.sum(np.array(dq)*np.array(dH_ad))*1000 # kJ --> J
    dT_new = Cp_av_in/Cp_av_sys*(T_in - T_fin) + Heat_ad/Cp_av_sys # K
    #dT_new = 1/Cp_av*()
    return dT_new


# %% storage_tank

def storage_tank(
    comp_in, pyiast_list,dH_ad, T_ref, init_var,
    Cp_solid = 948,Cp_gas = [40.63, 29.22], # solid = J/kg/K || gas = J/mol/K
    P_in = 10, T_in = 300,
    Design = {'A': 3.1416, 'L':3,'rho_s':1000, 'epsi':0.4}):
    #A = Design['A']
    #L = Design['L']
    #rho_s = Desigh['rho_s']
    #epsi = Design['epsi']
    n = len(pyiast_list)
    #x= np.array(init_var)[0:n]
    #q = np.array(init_var)[n:2*n]
    #T_init = init_var[2*n]
    def err_dq_dT(dqdT):
        res_mass = Massbal_interation(
            dqdT[0:n], dqdT[n],
            comp_in, pyiast_list,dH_ad, 
            T_ref,init_var,
            P_in, T_in, Design)
        del_m_tmp = res_mass[0]
        del_q_tmp = res_mass[1]
        Penalty = res_mass[2]
        #Ind_valid = res_mass[3]
        res_heat = Heatbal_interation(
            del_m_tmp, del_q_tmp,dqdT[n],
            comp_in, pyiast_list,dH_ad,
            T_ref,init_var,Cp_solid,Cp_gas,
            P_in,T_in,Design)
        del_T_tmp = res_heat
        obj_val = 10*np.sum((np.array(dqdT[0:n]) - del_q_tmp)**2) + (dqdT[n]-del_T_tmp)**2
        return obj_val + 100*Penalty
    x0 = np.ones(n+1)
    x0[n] = 0.01
    # BOUNDARIES
    bounds = []
    for ii in range(n):
        bounds.append([-80,90])
    bounds.append([-80,150])

    opt_res = optim.minimize(
        err_dq_dT, x0,
        method = 'CG')
    opt_res_best = opt_res
    Tol_setting = 1E-6
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        opt_res = optim.minimize(
        err_dq_dT, x0,
        method = 'Nelder-Mead')
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res

    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        opt_res = optim.minimize(
        err_dq_dT, x0,
        method = 'Powell')
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        opt_res = optim.minimize(
        err_dq_dT, x0,
        bounds = bounds,
        method = 'SLSQP')
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        opt_res = optim.minimize(
        err_dq_dT, x0,
        method = 'TNC')
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        print('SHGO METHOD IS USED !!!')
        opt_res = optim.shgo(
        err_dq_dT, bounds)
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
            print('SHGO METHOD IS PICKED !!!')
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        print('Differential Evolution METHOD IS USED !!!')
        opt_res = optim.differential_evolution(
        err_dq_dT, bounds)
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
            print('Differential Evolution IS PICKED !!!')
    if opt_res_best.fun > Tol_setting:
        #x0 = np.ones(n+1)
        print('Dual Annealing METHOD IS USED !!!')
        opt_res = optim.dual_annealing(
        err_dq_dT, bounds)
        if opt_res_best.fun > opt_res.fun:
            opt_res_best = opt_res
            print('Dual Annealing IS PICKED !!!')
    return opt_res_best


# %% TEST this Library with simple binary case
if __name__ == '__main__':
    p = np.linspace(0,40,41)
    qm_list = [3,1]         # mol/kg
    b_list = [0.2,0.1]      # bar^-1
    dH_adsorption = [20,20] # kJ/mol
    T_reference = [298,298]

    # Inlet conditions ?
    feed_composition = [0.9,0.1]
    T_inlet = 300 # K
    T_inlet = 260 # K
    P_inlet = 3   # bar
    
    # Heat related properties (methane; nitrogen) + (zeolite)
    Cp_g = np.array([40.63,29.22])  # Gas heat capacity: J/mol/K
    Cp_s = 948                      # Solid heat capacity: J/kg/K

    # Molar mass (methane; nitrogen) kg/mol
    #Molar_mass = [0.016, 0.024]

    #print(Arrhenius(dH_adsorption, 300, T_reference))

    pyiast_res_list = []
    for qmm, bb in zip(qm_list,b_list):
        q_tmp = qmm*bb*p/(1+bb*p)
        di_tmp = {'p':p, 'q': q_tmp}
        df_tmp = pd.DataFrame(di_tmp)
        iso_tmp = pyiast.ModelIsotherm(
            df_tmp,'q','p',model='Langmuir',
            param_guess= {'M':qmm,'K': bb})
        pyiast_res_list.append(iso_tmp)

    ## Initial values
    x_gas = [0.9,0.1]
    #q_solid = [1.1771,0.13]
    q_solid = [0.0,0.0]
    T_current = [298]
    Mass_trasnfer_coeff = [1]
    initial_variables = x_gas + q_solid + T_current + Mass_trasnfer_coeff
    
    Delta_q_test = [1.97,0.003]
    Delta_q_test = [0.6156,0.29251]
    Delta_q_test = [0.6156,0.29251]

    Result_StorageTank = storage_tank(
        feed_composition,pyiast_res_list,dH_adsorption,T_reference,
        initial_variables,P_in = P_inlet, T_in = T_inlet)
    print('Funtion Test')
    print('storage_tank')
    print(Result_StorageTank)
    print()

    dq_sol = Result_StorageTank.x[0:len(pyiast_res_list)]
    dT_sol = Result_StorageTank.x[len(pyiast_res_list)]

    #Delta_q_test = [0.02,0.02]
    Massbal_iter_test = Massbal_interation(
        dq_sol, dT_sol, feed_composition,
        pyiast_res_list, dH_adsorption, T_reference,
        initial_variables,P_in = P_inlet, T_in = T_inlet)
    print('Function Test2')
    print('Massbal_iteration')
    print(Massbal_iter_test)
    print()

    Energybal_iter_test = Heatbal_interation(
        Massbal_iter_test[0],Massbal_iter_test[1], dT_sol,
        feed_composition, pyiast_res_list,
        dH_adsorption, T_reference,initial_variables,P_in = P_inlet, T_in = T_inlet
        )
    print('Function Test3')
    print('Heatbal_iteration')
    print(Energybal_iter_test)


