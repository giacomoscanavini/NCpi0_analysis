import math 
import numpy as np
import pandas as pd
import uproot3 as uproot
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import copy
import matplotlib as mpl

# ------------------------------------------------------------------------------------
# Definitions for plots labels (truth)
# ------------------------------------------------------------------------------------
def is_badmatch(df):
    df_ = df[(df.match_completeness_energy<=0.1*df.truth_energyInside)].copy()
    return df_.reset_index(drop=True)

def is_goodmatch(df):
    df_ = df[(df.match_completeness_energy>0.1*df.truth_energyInside)].copy()
    return df_.reset_index(drop=True)

def is_outFV(df):
    df_ = df[((df.match_completeness_energy>0.1*df.truth_energyInside) & (df.truth_vtxInside==0))].copy()
    return df_.reset_index(drop=True)

def is_nueCCinFV(df):
    df_ = df[((df.match_completeness_energy>0.1*df.truth_energyInside) & (abs(df.truth_nuPdg)==12) & (df.truth_isCC==1) & (df.truth_vtxInside==1))].copy()
    return df_.reset_index(drop=True)

def is_numuCCinFV(df):
    df_ = df[((df.match_completeness_energy>0.1*df.truth_energyInside) & (abs(df.truth_nuPdg)==14) & (df.truth_isCC==1) & (df.truth_vtxInside==1) & (df.truth_NprimPio!=1))].copy()
    return df_.reset_index(drop=True)
  
def is_CCpi0inFV(df):
    df_ = df[((df.match_completeness_energy>0.1*df.truth_energyInside) & (abs(df.truth_nuPdg)==14) & (df.truth_isCC==1) & (df.truth_vtxInside==1) & (df.truth_NprimPio==1))].copy()
    return df_.reset_index(drop=True)

def is_NCinFV(df):
    df_ = df[(((df.match_completeness_energy>0.1*df.truth_energyInside) & (df.truth_isCC==0) & (df.truth_vtxInside==1) & (df.truth_NprimPio!=1)))].copy() #|
              #((df.match_completeness_energy>0.1*df.truth_energyInside) & (df.truth_isCC==0) & (df.truth_vtxInside==1) & (df.truth_NprimPio==1) & (df.truth_nuEnergy<275)) |
              #((df.match_completeness_energy>0.1*df.truth_energyInside) & (df.truth_isCC==0) & (df.truth_vtxInside==1) & (df.truth_NprimPio==1) & (df.truth_nuEnergy>4000)))] 
    return df_.reset_index(drop=True)

def is_NCpi0inFV(df):
    df_ = df[((df.match_completeness_energy>0.1*df.truth_energyInside) & (df.truth_isCC==0) & (df.truth_vtxInside==1) & (df.truth_NprimPio==1))].copy() # & (df.truth_nuEnergy>=275) & (df.truth_nuEnergy<4000))]
    return df_.reset_index(drop=True)

def is_QE(df):
    df_ = df[(df.truth_nuScatType==1)].copy()
    return df_.reset_index(drop=True)

def is_RES(df):
    df_ = df[(df.truth_nuScatType==4)].copy()
    return df_.reset_index(drop=True)

def is_DIS(df):
    df_ = df[(df.truth_nuScatType==3)].copy()
    return df_.reset_index(drop=True)

def is_MEC(df):
    df_ = df[(df.truth_nuScatType==10)].copy()
    return df_.reset_index(drop=True)

def is_COH(df):
    df_ = df[(df.truth_nuScatType==5)].copy()
    return df_.reset_index(drop=True)

def is_OTHER(df):
    df_ = df[((df.truth_nuScatType!=1) & (df.truth_nuScatType!=4) & (df.truth_nuScatType!=3) & (df.truth_nuScatType!=10) & (df.truth_nuScatType!=5))].copy()
    return df_.reset_index(drop=True)



# ------------------------------------------------------------------------------------
# Definitions for true types
# ------------------------------------------------------------------------------------
def is_true_nueCC(df):
    df_ = df[((abs(df.truth_nuPdg) == 12) & (df.truth_isCC == 1))].copy()
    return df_.reset_index(drop=True)

def is_true_numuCC(df):
    df_ = df[((abs(df.truth_nuPdg) == 14) & (df.truth_isCC == 1))].copy()
    return df_.reset_index(drop=True)

def is_true_CCpi0(df):
    df_ = df[((abs(df.truth_nuPdg) == 14) & (df.truth_isCC == 1) & (df.truth_NprimPio==1))].copy()
    return df_.reset_index(drop=True)

def is_true_NC(df):
    df_ = df[(df.truth_isCC == 0)].copy()
    return df_.reset_index(drop=True)

def is_true_NCpi0(df):
    df_ = df[((df.truth_isCC == 0) & (df.truth_NprimPio==1))].copy()
    return df_.reset_index(drop=True)

def is_trueFV(df):
    df_ = df[df.truth_vtxInside==1].copy()
    return df_.reset_index(drop=True)

def isnot_trueFV(df):
    df_ = df[df.truth_vtxInside!=1].copy()
    return df_.reset_index(drop=True)

def is_true_0p(df):
    df_ = df[df.truth_num_protons==0].copy()
    return df_.reset_index(drop=True)

def is_true_1p(df):
    df_ = df[df.truth_num_protons==1].copy()
    return df_.reset_index(drop=True)

def is_true_Np(df):
    df_ = df[df.truth_num_protons>0].copy()
    return df_.reset_index(drop=True)

# ------------------------------------------------------------------------------------
# Definitions for Wire-Cell efficiencies (true types + vtx in FV)
# ------------------------------------------------------------------------------------
def is_trueFV_nueCC(df):
    df_ = is_trueFV(is_true_nueCC(df))
    return df_.reset_index(drop=True)

def is_trueFV_numuCC(df):
    df_ = is_trueFV(is_true_numuCC(df))
    return df_.reset_index(drop=True)

def is_trueFV_CCpi0(df):
    df_ = is_trueFV(is_true_CCpi0(df))
    return df_.reset_index(drop=True)

def is_trueFV_NC(df):
    df_ = is_trueFV(is_true_NC(df))
    return df_.reset_index(drop=True)

def is_trueFV_NCpi0(df):
    df_ = is_trueFV(is_true_NCpi0(df))
    return df_.reset_index(drop=True)

# ------------------------------------------------------------------------------------
# Definitions for selections
# ------------------------------------------------------------------------------------
def is_generic(df):
    df_ = df[(df.match_found == 1) & (df.stm_eventtype != 0) & (df.stm_lowenergy == 0) & (df.stm_LM == 0) & (df.stm_TGM == 0) & (df.stm_STM == 0) & (df.stm_FullDead == 0) & (df.stm_clusterlength > 15)]
    return df_.reset_index(drop=True)

def is_nueCC(df):
    df_ = is_generic(df)
    df_ = df_[((df_.numu_cc_flag >= 0) & (df_.nue_score > 7))]
    return df_.reset_index(drop=True)

def is_numuCC(df):
    df_ = is_generic(df)
    df_ = df_[((df_.numu_cc_flag >= 0) & (df_.numu_score > 0.9) & (df_['reco_muonMomentum[3]']>0))]
    return df_.reset_index(drop=True)

def is_CCpi0(df):
    df_ = is_generic(df)
    df_ = is_numuCC(df_)
    df_ = is_pi0(df_)
    return df_.reset_index(drop=True)

def is_NC(df):
    df_ = is_generic(df)
    df_ = df_[(~(df_.cosmict_flag) & (df_.numu_score < 0))]
    return df_.reset_index(drop=True)

def is_NCpi0BDT(df):
    df_ = is_generic(df)
    df_ = df_[((df_.nc_pio_score > 1.816) & (df_.numu_cc_flag >=0) & (df_.kine_pio_energy_1 > 0.) & (df_.kine_pio_energy_2 > 0.))]
    return df_.reset_index(drop=True)

def isnot_NCpi0BDT(df):
    df_ = is_generic(df)
    df_ = df_[~((df_.nc_pio_score > 1.816) & (df_.numu_cc_flag >=0) & (df_.kine_pio_energy_1 > 0.) & (df_.kine_pio_energy_2 > 0.))]
    return df_.reset_index(drop=True)

def is_FC(df):
    df_ = df[df.match_isFC == 1]
    return df_.reset_index(drop=True)

def is_PC(df):
    df_ = df[df.match_isFC == 0]
    return df_.reset_index(drop=True)

def is_pi0(df):
    df_ = df[((df.kine_pio_flag==1 & df.kine_pio_vtx_dis < 9 | df.kine_pio_flag==2) & (df.kine_pio_energy_1 > 40) & (df.kine_pio_energy_2 > 25) & (df.kine_pio_dis_1 < 110) & (df.kine_pio_dis_2 < 120) & (df.kine_pio_angle > 0) & (df.kine_pio_angle < 174) & (df.kine_pio_mass > 22) & (df.kine_pio_mass < 300))]
    return df_.reset_index(drop=True)

def is_0p(df):
    df_ = df[df.reco_num_protons==0]
    #df_ = df[df.reco_Nproton==0]
    return df_.reset_index(drop=True)

def is_1p(df):
    df_ = df[df.reco_num_protons==1]
    return df_.reset_index(drop=True)

def is_Np(df):
    df_ = df[df.reco_num_protons>0]
    #df_ = df[df.reco_Nproton>0]
    return df_.reset_index(drop=True)

# ------------------------------------------------------------------------------------
# Data wrangling functions
# ------------------------------------------------------------------------------------
def gen_run_subrun_list(input_file, name_list):
    # Get useful variables
    T_eval = uproot.open(input_file)['wcpselection/T_eval']
    df_eval = T_eval.pandas.df(['run', 'subrun'], flatten=False)
    df_eval.drop_duplicates(inplace=True) 
    np.savetxt(name_list, df_eval.values, fmt='%d')
'''
def get_reco_Enu_corr(df, family=None):
    em_charge_scale = 0.95
    df.loc[:,'reco_Enu_corr'] = [0]*df.shape[0]
    if(family=='DATA'):
        for ith in range(df.shape[0]):
            total_E = 0
            if(len(df.loc[ith,'kine_energy_particle'])>0):
                for x,y,z in zip(df.loc[ith,'kine_energy_particle'], df.loc[ith,'kine_energy_info'], df.loc[ith,'kine_particle_type']):
                    if(y == 2 and z == 11): total_E +=  x * em_charge_scale
                    else: total_E +=  x
            total_E += df.loc[ith,'kine_reco_add_energy']
            df.loc[ith,'reco_Enu_corr'] = total_E
    else: df.loc[:,'reco_Enu_corr'] = df.kine_reco_Enu
    return df
'''
def get_reco_Enu_corr(df, family=None):
    em_charge_scale = 0.95
    df['reco_Enu_corr'] = np.nan
    if family == 'DATA':
        for idx, row in df.iterrows():
            total_E = 0
            if len(row['kine_energy_particle']) > 0:
                for x, y, z in zip(row['kine_energy_particle'], row['kine_energy_info'], row['kine_particle_type']):
                    if y == 2 and z == 11: total_E += x * em_charge_scale
                    else: total_E += x
            total_E += row['kine_reco_add_energy']
            df.at[idx, 'reco_Enu_corr'] = total_E
    else: df['reco_Enu_corr'] = df['kine_reco_Enu']
    return df

'''
def add_reco_num_protons(df):
    df.loc[:,'reco_num_protons'] = [-999]*df.shape[0]
    df.loc[:,'nonreco_num_protons'] = [-999]*df.shape[0]
    #df.loc[:,'reco_KE_leading_proton'] = [-999]*df.shape[0]
    for ith in range(df.shape[0]):
        Np = 0
        Np_ = 0
        reco_KE_leading_proton = 0
        if(len(df.loc[ith,'kine_energy_particle'])>0):
            for x,z in zip(df.loc[ith,'kine_energy_particle'], df.loc[ith,'kine_particle_type']):
                if(abs(z) == 2212 and x > 35): Np +=1
                if(abs(z) == 2212 and x < 35): Np_ +=1
            df.loc[ith,'reco_num_protons'] = Np
            df.loc[ith,'nonreco_num_protons'] = Np_
    return df
'''
def add_reco_num_protons2(df):
    df['reco_num_protons'] = 0
    df['nonreco_num_protons'] = 0
    df['kine_KE_leading_proton'] = 0.
    df['kine_theta_leading_proton'] = np.nan
    for idx, row in df.iterrows():
        Np = 0
        Np_ = 0
        kine_KE_leading_proton = 0.
        kine_theta_leading_proton = np.nan
        if len(row['reco_mother']) > 0:
            for x, y, z in zip(row['reco_pdg'], row['reco_mother'], row['reco_startMomentum']):
                if abs(x) == 2212 and y==0: 
                    if(np.sqrt(z[0]**2 + z[1]**2 + z[2]**2) > 0): theta = z[2]/np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
                    else: theta = np.nan
                    Ep = z[3]*1000-938.27208816
                    if kine_KE_leading_proton < Ep: 
                        kine_KE_leading_proton = Ep 
                        kine_theta_leading_proton = theta
                    if Ep > 35: Np += 1
                    elif Ep < 35: Np_ += 1
            df.at[idx, 'reco_num_protons'] = Np
            df.at[idx, 'nonreco_num_protons'] = Np_
            df.at[idx, 'kine_KE_leading_proton'] = kine_KE_leading_proton
            df.at[idx, 'kine_theta_leading_proton'] = kine_theta_leading_proton
    return df

def add_reco_num_protons(df):
    df['reco_num_protons'] = -999.
    df['nonreco_num_protons'] = -999.
    df['kine_KE_leading_proton'] = 0.
    for idx, row in df.iterrows():
        Np = 0
        Np_ = 0
        kine_KE_leading_proton = 0.
        for x, z in zip(row['kine_energy_particle'], row['kine_particle_type']):
            if abs(z) == 2212: 
                if kine_KE_leading_proton < x: kine_KE_leading_proton = x 
                if x > 35: Np += 1
                elif x < 35: Np_ += 1
        df.at[idx, 'reco_num_protons'] = Np
        df.at[idx, 'nonreco_num_protons'] = Np_
        df.at[idx, 'kine_KE_leading_proton'] = kine_KE_leading_proton
    return df
'''    
def add_true_num_protons(df):
    df.loc[:,'true_num_protons'] = [-999]*df.shape[0]
    df.loc[:,'true_KE_leading_proton'] = [-999]*df.shape[0]
    for ith in range(df.shape[0]):
        Np = 0
        true_KE_leading_proton = 0
        if(len(df.loc[ith,'truth_mother'])>0):
            for x,y,z in zip(df.loc[ith,'truth_mother'], df.loc[ith,'truth_pdg'], df.loc[ith,'truth_startMomentum']):
                if((x == 0) and (abs(y) == 2212)):
                    ke_prot = z[3]*1000.-938.27
                    if true_KE_leading_proton < ke_prot: true_KE_leading_proton = ke_prot
                    if(ke_prot > 35): Np += 1
            df.loc[ith,'true_num_protons'] = Np
            df.loc[ith,'true_KE_leading_proton'] = true_KE_leading_proton
    return df
'''
def add_true_num_protons(df):
    df['truth_num_protons'] = 0
    df['truth_KE_leading_proton'] = 0
    df['truth_theta_leading_proton'] = np.nan
    for idx, row in df.iterrows():
        Np = 0
        truth_KE_leading_proton = 0
        truth_theta_leading_proton = np.nan
        if len(row['truth_mother']) > 0:
            for x, y, z in zip(row['truth_pdg'], row['truth_mother'], row['truth_startMomentum']):
                if abs(x) == 2212 and y==0: 
                    if(np.sqrt(z[0]**2 + z[1]**2 + z[2]**2) > 0): theta = z[2]/np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
                    else: theta = np.nan
                    Ep = z[3]*1000-938.27208816
                    if truth_KE_leading_proton < Ep: 
                        truth_KE_leading_proton = Ep
                        truth_theta_leading_proton = theta
                    if Ep > 35: Np += 1
            df.at[idx, 'truth_num_protons'] = Np
            df.at[idx, 'truth_KE_leading_proton'] = truth_KE_leading_proton
            df.at[idx, 'truth_theta_leading_proton'] = truth_theta_leading_proton
    return df
'''
def add_reco_info(df, family):
    # add reco number of protons 
    df = add_reco_num_protons(df.reset_index(drop=True))
    df = get_reco_Enu_corr(df.reset_index(drop=True), family)
    # add kine_pio_energy_high, kine_pio_energy_low
    #df.loc[:,'kine_pio_energy_high'] = [-999.]*df.shape[0]
    #df.loc[:,'kine_pio_energy_low'] = [-999.]*df.shape[0]
    for ith in range(df.shape[0]):
        if (df.loc[ith,'kine_pio_energy_1'] > 0 and df.loc[ith,'kine_pio_energy_2'] > 0):
            if (df.loc[ith,'kine_pio_energy_1'] >  df.loc[ith,'kine_pio_energy_2']):
                df.loc[ith,'kine_pio_energy_high'] = df.loc[ith,'kine_pio_energy_1']
                df.loc[ith,'kine_pio_energy_low'] = df.loc[ith,'kine_pio_energy_2']
                df.loc[ith,'kine_pio_theta_high'] = df.loc[ith,'kine_pio_theta_1']
                df.loc[ith,'kine_pio_theta_low'] = df.loc[ith,'kine_pio_theta_2']
                df.loc[ith,'kine_pio_phi_high'] = df.loc[ith,'kine_pio_phi_1']
                df.loc[ith,'kine_pio_phi_low'] = df.loc[ith,'kine_pio_phi_2']
            else: 
                df.loc[ith,'kine_pio_energy_high'] = df.loc[ith,'kine_pio_energy_2']
                df.loc[ith,'kine_pio_energy_low'] = df.loc[ith,'kine_pio_energy_1']
                df.loc[ith,'kine_pio_theta_high'] = df.loc[ith,'kine_pio_theta_2']
                df.loc[ith,'kine_pio_theta_low'] = df.loc[ith,'kine_pio_theta_1']
                df.loc[ith,'kine_pio_phi_high'] = df.loc[ith,'kine_pio_phi_2']
                df.loc[ith,'kine_pio_phi_low'] = df.loc[ith,'kine_pio_phi_1']
    # add pio_mass_gamma, kine_pio_energy_gamma
    df.loc[:,'kine_pio_mass_gamma'] = np.sqrt(2.*df.loc[:,'kine_pio_energy_high'] * df.loc[:,'kine_pio_energy_low'] *(1-np.cos(df.loc[:,'kine_pio_angle']*3.1415926/180.)));
    df.loc[:,'kine_pio_energy_gamma'] = [x+y for x,y in zip(df.loc[:,'kine_pio_energy_high'],df.loc[:,'kine_pio_energy_low'])]
    # add kine_pio_energy, kine_pio_energy_tot, kine_pio_momentum
    pi0_mass = 135
    alpha = abs(df.kine_pio_energy_high - df.kine_pio_energy_low)/(df.kine_pio_energy_high + df.kine_pio_energy_low)
    df.loc[:,'kine_pio_energy'] = pi0_mass * (np.sqrt(2./(1-alpha*alpha)/(1-np.cos(df.kine_pio_angle/180.*np.pi)))-1)
    df.loc[:,'kine_pio_energy_tot'] = pi0_mass + df.kine_pio_energy
    df.loc[:,'kine_pio_momentum'] = np.sqrt(df.kine_pio_energy_tot*df.kine_pio_energy_tot - pi0_mass*pi0_mass)
    # add kine_pio_costheta, kine_nonpio_energy
    p_0 = df.kine_pio_energy_1 + df.kine_pio_energy_2
    # ----------------------------------------------------------
    p1_1 = df.kine_pio_energy_1*np.cos(df.kine_pio_phi_1/180.*np.pi)*np.sin(df.kine_pio_theta_1/180.*np.pi)
    p2_1 = df.kine_pio_energy_2*np.cos(df.kine_pio_phi_2/180.*np.pi)*np.sin(df.kine_pio_theta_2/180.*np.pi)
    p_1 = p1_1 + p2_1
    # ----------------------------------------------------------
    p1_2 = df.kine_pio_energy_1*np.sin(df.kine_pio_phi_1/180.*np.pi)*np.sin(df.kine_pio_theta_1/180.*np.pi)
    p2_2 = df.kine_pio_energy_2*np.sin(df.kine_pio_phi_2/180.*np.pi)*np.sin(df.kine_pio_theta_2/180.*np.pi)
    p_2 = p1_2 + p2_2
    # ----------------------------------------------------------
    p1_3 = df.kine_pio_energy_1*np.cos(df.kine_pio_theta_1/180.*np.pi)
    p2_3 = df.kine_pio_energy_2*np.cos(df.kine_pio_theta_2/180.*np.pi)
    p_3 = p1_3 + p2_3
    # ----------------------------------------------------------
    df.loc[:,'kine_pio_costheta'] = p_3/np.sqrt(p_1**2 + p_2**2 + p_3**2)
    df.loc[:,'kine_nonpio_energy'] = df.reco_Enu_corr - df.kine_pio_energy_gamma
    return df
'''
def add_reco_info(df, family):
    df = add_reco_num_protons2(df.reset_index(drop=True)) # Adds reconstructed number of protons
    df = get_reco_Enu_corr(df.reset_index(drop=True), family) # Gets the reconstructed Enu correction for the given family
    # Gets the maximum and minimum pion energies and their corresponding angles and momenta
    pion_energy_high = np.maximum(df['kine_pio_energy_1'], df['kine_pio_energy_2'])
    pion_energy_low = np.minimum(df['kine_pio_energy_1'], df['kine_pio_energy_2'])
    pion_theta_high = np.where(df['kine_pio_energy_1'] > df['kine_pio_energy_2'], df['kine_pio_theta_1'], df['kine_pio_theta_2'])
    pion_theta_low = np.where(df['kine_pio_energy_1'] > df['kine_pio_energy_2'], df['kine_pio_theta_2'], df['kine_pio_theta_1'])
    pion_phi_high = np.where(df['kine_pio_energy_1'] > df['kine_pio_energy_2'], df['kine_pio_phi_1'], df['kine_pio_phi_2'])
    pion_phi_low = np.where(df['kine_pio_energy_1'] > df['kine_pio_energy_2'], df['kine_pio_phi_2'], df['kine_pio_phi_1'])
    # Adds high and low pion energies, pion mass, and total pion energy
    pion_mass = 135.0
    pion_angle = df['kine_pio_angle'] * np.pi / 180.0
    alpha = np.abs(pion_energy_high - pion_energy_low) / (pion_energy_high + pion_energy_low)
    pion_energy = pion_mass * (np.sqrt(2.0 / (1 - alpha ** 2) / (1 - np.cos(pion_angle))) - 1)
    pion_energy_tot = pion_mass + pion_energy
    pion_momentum = np.sqrt(pion_energy_tot ** 2 - pion_mass ** 2)
    pion_mass_gamma = np.sqrt(2.0 * pion_energy_high * pion_energy_low * (1 - np.cos(pion_angle)))
    df['kine_pio_energy_high'] = pion_energy_high
    df['kine_pio_energy_low'] = pion_energy_low
    df['kine_pio_theta_high'] = pion_theta_high
    df['kine_pio_theta_low'] = pion_theta_low
    df['kine_pio_phi_high'] = pion_phi_high
    df['kine_pio_phi_low'] = pion_phi_low
    df['kine_pio_mass_gamma'] = pion_mass_gamma
    df['kine_pio_energy_gamma'] = pion_energy_high + pion_energy_low
    # Calculate kine_pio_energy, kine_pio_energy_tot, and kine_pio_momentum
    df['kine_pio_energy'] = pion_energy
    df['kine_pio_energy_tot'] = pion_energy_tot
    df['kine_pio_momentum'] = pion_momentum
    # Calculate kine_pio_costheta and kine_nonpio_energy
    p1_1 = pion_energy_high*np.cos(pion_phi_high/180.*np.pi)*np.sin(pion_theta_high/180.*np.pi)
    p2_1 = pion_energy_low*np.cos(pion_phi_low/180.*np.pi)*np.sin(pion_theta_low/180.*np.pi)
    p_1 = p1_1 + p2_1
    # ----------------------------------------------------------
    p1_2 = pion_energy_high*np.sin(pion_phi_high/180.*np.pi)*np.sin(pion_theta_high/180.*np.pi)
    p2_2 = pion_energy_low*np.sin(pion_phi_low/180.*np.pi)*np.sin(pion_theta_low/180.*np.pi)
    p_2 = p1_2 + p2_2
    # ----------------------------------------------------------
    p1_3 = pion_energy_high*np.cos(pion_theta_high/180.*np.pi)
    p2_3 = pion_energy_low*np.cos(pion_theta_low/180.*np.pi)
    p_3 = p1_3 + p2_3
    # ----------------------------------------------------------
    df.loc[:,'kine_pio_costheta'] = p_3/np.sqrt(p_1**2 + p_2**2 + p_3**2)
    df.loc[:,'kine_nonpio_energy'] = df['reco_Enu_corr'] - df['kine_pio_energy_gamma']
    return df
'''
def add_truth_info(df):
    # add true number of protons 
    df = add_true_num_protons(df.reset_index(drop=True))
    # add truth_pio_energy, truth_pio_energy_tot, truth_pio_momentum
    pi0_mass = 135
    alpha = abs(df.truth_pio_energy_1 - df.truth_pio_energy_2)/(df.truth_pio_energy_1 + df.truth_pio_energy_2)
    df.loc[:,'truth_pio_energy'] = pi0_mass * (np.sqrt(2./(1-alpha*alpha)/(1-np.cos(df.truth_pio_angle/180.*np.pi)))-1)
    df.loc[:,'truth_pio_energy_tot'] = pi0_mass + df.truth_pio_energy
    df.loc[:,'truth_pio_momentum'] = np.sqrt(df.truth_pio_energy_tot*df.truth_pio_energy_tot - pi0_mass*pi0_mass)
    for ith in range(df.shape[0]):
        if (df.loc[ith,'truth_pio_energy_1'] > 0 and df.loc[ith,'truth_pio_energy_2'] > 0):
            if (df.loc[ith,'truth_pio_energy_1'] >  df.loc[ith,'truth_pio_energy_2']):
                df.loc[ith,'truth_pio_energy_high'] = df.loc[ith,'truth_pio_energy_1']
                df.loc[ith,'truth_pio_energy_low'] = df.loc[ith,'truth_pio_energy_2']
            else: 
                df.loc[ith,'truth_pio_energy_high'] = df.loc[ith,'truth_pio_energy_2']
                df.loc[ith,'truth_pio_energy_low'] = df.loc[ith,'truth_pio_energy_1']
    # add truth_pio_costheta
    truth_pio_costheta = []
    for x,y,N,P in zip(df.truth_mother, df.truth_pdg, range(len(df.truth_mother)), df.truth_startMomentum):
        index = [1 if (w == 0 and z == 111) else 0 for w,z in zip(list(x),list(y))]
        ind = [i for i in range(len(index)) if index[i] == 1]
        if len(ind) > 0:
            P_pi = [P[j][3] for j in ind]
            ind = ind[P_pi.index(max(P_pi))]
            P_perp = np.sqrt(P[ind][0]*P[ind][0] + P[ind][1]*P[ind][1])
            truth_pio_costheta.append(np.cos(np.arctan2(P_perp,P[ind][2])))#*180./np.pi)
        else: 
            truth_pio_costheta.append(-999.)
    df.loc[:,'truth_pio_costheta'] = truth_pio_costheta
    return df
'''
def add_truth_info(df):
    df = add_true_num_protons(df.reset_index(drop=True)) # Adds true number of protons
    # Gets the maximum and minimum pion energies and their corresponding angles and momenta
    pion_energy_high = np.maximum(df['truth_pio_energy_1'], df['truth_pio_energy_2'])
    pion_energy_low = np.minimum(df['truth_pio_energy_1'], df['truth_pio_energy_2'])
    # Adds high and low pion energies, pion mass, and total pion energy
    pion_mass = 135.0
    pion_angle = df['truth_pio_angle'] * np.pi / 180.0
    alpha = np.abs(pion_energy_high - pion_energy_low) / (pion_energy_high + pion_energy_low)
    pion_energy = pion_mass * (np.sqrt(2.0 / (1 - alpha ** 2) / (1 - np.cos(pion_angle))) - 1)
    pion_energy_tot = pion_mass + pion_energy
    pion_momentum = np.sqrt(pion_energy_tot ** 2 - pion_mass ** 2)
    pion_mass_gamma = np.sqrt(2.0 * pion_energy_high * pion_energy_low * (1 - np.cos(pion_angle)))
    df['truth_pio_energy_high'] = pion_energy_high
    df['truth_pio_energy_low'] = pion_energy_low
    df['truth_pio_mass_gamma'] = pion_mass_gamma
    df['truth_pio_energy_gamma'] = pion_energy_high + pion_energy_low
    # Calculate truth_pio_energy, truth_pio_energy_tot, and truth_pio_momentum
    df['truth_pio_energy'] = pion_energy
    df['truth_pio_energy_tot'] = pion_energy_tot
    df['truth_pio_momentum'] = pion_momentum
    # Calculate truth_pio_costheta
    truth_pio_costheta = []
    for mother, pdg, startMomentum in zip(df.truth_mother, df.truth_pdg, df.truth_startMomentum):
        is_pi0 = (pdg == 111) & (mother == 0)
        if np.any(is_pi0):
            index = np.argmax(startMomentum[is_pi0, 3])
            p_perp = np.sqrt(startMomentum[is_pi0][index, 0]**2 + startMomentum[is_pi0][index, 1]**2)
            truth_pio_costheta.append(np.cos(np.arctan2(p_perp, startMomentum[is_pi0][index, 2])))
        else:
            truth_pio_costheta.append(-999)
    df['truth_pio_costheta'] = truth_pio_costheta
    return df


eval_variables = ['run', 'subrun', 'event',
                  'match_isFC', 'stm_clusterlength', 'match_found', 'stm_eventtype', 
                  'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead']
eval_variables_MC = ['truth_isCC', 'truth_nuPdg', 'truth_nuEnergy', 'truth_vtxInside', 
                     'weight_spline', 'weight_cv', 
                     'truth_energyInside', 'match_completeness_energy']+eval_variables

KINE_variables = ['kine_reco_Enu', 'kine_reco_add_energy',
                  'kine_pio_mass', 'kine_pio_flag',
                  'kine_pio_vtx_dis', 'kine_pio_energy_1',
                  'kine_pio_theta_1', 'kine_pio_phi_1',
                  'kine_pio_theta_2', 'kine_pio_phi_2',
                  'kine_pio_dis_1', 'kine_pio_energy_2',
                  'kine_pio_dis_2', 'kine_pio_angle',
                  'kine_energy_particle', 'kine_energy_info',
                  'kine_particle_type', 'kine_energy_included']

pfeval_variables = ['reco_nuvtxX', 'reco_nuvtxY', 'reco_nuvtxZ', 'reco_muonMomentum','reco_Nproton', 'reco_protonMomentum', 'reco_Ntrack', 'reco_mother', 'reco_pdg', 'reco_startMomentum']
pfeval_variables_MC = ['truth_NprimPio', 'truth_NCDelta', 'truth_pio_energy_1', 'truth_pio_energy_2', 'truth_nuScatType',
                       'truth_pio_angle', 'truth_vtxX', 'truth_vtxY', 'truth_vtxZ',
                       'truth_corr_nuvtxX', 'truth_corr_nuvtxY', 'truth_corr_nuvtxZ',
                       'mc_nu_mode', 'truth_Ntrack', 'reco_Ntrack', #'reco_Nproton', 'reco_protonMomentum',
                       'truth_pdg', 'truth_mother', 'truth_id', 'reco_id', 'truth_startMomentum']+pfeval_variables

BDT_variables = ['numu_score', 'nue_score', 'nc_pio_score', 'cosmict_flag', 'numu_cc_flag']

def gen_dataframe(input_file, POT_goal, family='MC', POT_file=None):
    # family: MC / DATA
    # POT_file: useful if family == DATA

    ############################################################ Import T_pot
    if family=='MC':
        # Calculate POT and scaling factor
        T_pot = uproot.open(input_file)['wcpselection/T_pot']
        df_pot = T_pot.pandas.df(T_pot.keys(), flatten=False)
        #df_pot = df_pot.drop_duplicates(subset=['runNo','subRunNo'])
        if POT_file == None: POT_file = sum(df_pot.pot_tor875)
        elif POT_file != None: POT_file = POT_file
        W_ = POT_goal/POT_file
        print('\033[1m'+'[%s]: POT = %s, W = %s'%(family, POT_file, W_)+'\033[0m')
    elif family=='DATA':
        # Calculate POT and scaling factor
        POT_file = POT_file
        W_ = POT_goal/POT_file
        print('\033[1m'+'[%s]: POT = %s, W = %s'%(family, POT_file, W_)+'\033[0m')
    else: print(' -------- WARNING: Wrong family')
      
  ############################################################ Import T_KINEvars
    T_KINEvars = uproot.open(input_file)['wcpselection/T_KINEvars']
    df_KINEvars = T_KINEvars.pandas.df(KINE_variables, flatten=False)
    if family=='DATA':
        em_charge_scale = 0.95
        df_KINEvars.loc[:,'kine_pio_mass'] = df_KINEvars.kine_pio_mass *em_charge_scale
        df_KINEvars.loc[:,'kine_pio_energy_1'] = df_KINEvars.kine_pio_energy_1 *em_charge_scale
        df_KINEvars.loc[:,'kine_pio_energy_2'] = df_KINEvars.kine_pio_energy_2 *em_charge_scale

  ############################################################ Import T_BDTvars
    T_BDTvars = uproot.open(input_file)['wcpselection/T_BDTvars']
    df_BDTvars = T_BDTvars.pandas.df(BDT_variables, flatten=False) 

  ############################################################ Import T_eval                                
    T_eval = uproot.open(input_file)['wcpselection/T_eval']
    if family=='MC': df_eval = T_eval.pandas.df(eval_variables_MC, flatten=False)
    elif family=='DATA': df_eval = T_eval.pandas.df(eval_variables, flatten=False)
    else: print(' -------- WARNING: Wrong family')
  
  ############################################################ Import T_PFeval
    T_PFeval = uproot.open(input_file)['wcpselection/T_PFeval']
    if family=='MC':
        df_PFeval = T_PFeval.pandas.df(pfeval_variables_MC, flatten=False)
        # Merge variables
        df = pd.concat([df_eval, df_KINEvars, df_BDTvars, df_PFeval], axis=1)
        # Weights + Limit weight values
        df['weight_cv'] = np.where((df.weight_cv <= 0), 1, df.weight_cv)
        df['weight_cv'] = np.where((df.weight_cv > 30), 1, df.weight_cv)
        df['weight_cv'] = np.where((df.weight_cv == np.nan), 1, df.weight_cv)
        df['weight_cv'] = np.where((df.weight_cv == np.inf), 1, df.weight_cv)
        df['weight_cv'] = np.where((df['weight_cv'].isna()), 1, df.weight_cv)
        df['weight_spline'] = np.where((df.weight_spline <= 0), 1, df.weight_spline)
        df['weight_spline'] = np.where((df.weight_spline > 30), 1, df.weight_spline)
        df['weight_spline'] = np.where((df.weight_spline == np.nan), 1, df.weight_spline)
        df['weight_spline'] = np.where((df.weight_spline == np.inf), 1, df.weight_spline)
        df['weight_spline'] = np.where((df['weight_spline'].isna()), 1, df.weight_spline)
        df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']
        df.loc[:,'weight'] = [W_]*df.shape[0] * df['weight_genie']
        # Add true and reco info (pi0, protons)
        df = add_reco_info(df.reset_index(drop=True), family)
        df = add_truth_info(df.reset_index(drop=True))
        #df = df.drop_duplicates(subset=['run','subrun','event']).reset_index(drop=True)
        print('Shape:', df.shape)
        return POT_file, W_, df
    elif family=='DATA':
        df_PFeval = T_PFeval.pandas.df(pfeval_variables, flatten=False)
        # Merge dataframes
        df = pd.concat([df_eval, df_KINEvars, df_BDTvars, df_PFeval], axis=1)
        # Weights + Limit weight values
        df.loc[:,'weight'] = [W_]*df.shape[0]
        # Add reco info (pi0, protons)
        df = add_reco_info(df.reset_index(drop=True), family)
        #df = df.drop_duplicates(subset=['run','subrun','event']).reset_index(drop=True)
        print('Shape:', df.shape)
        return POT_file, W_, df
    else: print(' -------- WARNING: Wrong family')

def merge_files(list_df, POT_goal, list_POT, family='MC'):
    # It works for one or multiple files
    # family: MC / DATA
    POT_tot = sum(list_POT)
    W_ = POT_goal/POT_tot
    print('POT_tot: %s (Normalization factor: %1.4f)'%(POT_tot, W_))
    
    if family == 'MC':
        df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']
        df.loc[:,'weight'] = [W_]*df.shape[0] * df['weight_genie']
    elif family == 'DATA':
        df.loc[:,'weight'] = [W_]*df.shape[0]
    else: print(' -------- WARNING: Wrong family')
    # Check weights are defined
    print('Events (raw): ', df.shape[0],' Events (weights): ', sum(df.weight))
    return W_, df

def apply_rw(df):
    # these reweighting are provided with constraints with numuCC
    # apply on 'truth_energyInside' # MeV
    
    # New weights
    rw_NCpi0_Np = [0.406366, 0.511624, 0.601452, 0.58003, 0.518105, 0.484413, 0.433561, 0.358873, 0.303316, 0.309188, 0.375834, 0.45791, 0.516311, 0.519274, 0.451251]
    rw_NCpi0_0p = [1.38501, 1.37444, 1.36037, 1.34265, 1.26576, 1.05254, 0.694593, 0.340093, 0.180095, 0.29023, 0.537049, 0.742396, 0.810889, 0.72826, 0.567224, 0.443822]
    
    # Old weights
    #rw_NCpi0_Np = [0.253369, 0.383944, 0.562346, 0.65335, 0.674356, 0.660178, 0.620949, 
    #               0.628801, 0.656192, 0.614876, 0.545956, 0.522634, 0.601086, 0.901991, 
    #               1.29757, 1.68813, 1.89426, 2.13387, 2.07376, 2.54674, 2.79741, 2.97804, 2.92738]
    
    #rw_NCpi0_0p = [1.03556, 1.0044, 1.303, 1.51352, 1.45154, 1.02605, 0.534695, 0.274776, 0.454917, 
    #               0.86527, 1.10643, 1.33621, 1.67019, 2.139, 2.85217]
    rw_ = []
    for ith in range(df.shape[0]):
        if (df.loc[ith,'truth_isCC']==0 and df.loc[ith,'truth_NprimPio']>0):
            if (df.loc[ith,'truth_num_protons']==0): 
                # 0p range = [150, 900] MeV
                rw_list = rw_NCpi0_0p
                len_ = len(rw_NCpi0_0p)
                #edges_ = np.linspace(150, 900, len_+1)  # Old edges
                edges_ = np.linspace(100, 900, len_+1)   # New edges
            elif (df.loc[ith,'truth_num_protons']>0):
                # Np range = [250, 1400] MeV
                rw_list = rw_NCpi0_Np
                len_ = len(rw_NCpi0_Np)
                #edges_ = np.linspace(250, 1400, len_+1) # Old edges
                edges_ = np.linspace(250, 1000, len_+1)   # New edges
                
            true_e = df.loc[ith,'truth_energyInside']
            for left, right, ix in zip(edges_, edges_[1:], range(len(edges_))):
                if true_e < edges_[0]: 
                    rw_.append(1)
                    break
                elif true_e > edges_[-1]: 
                    rw_.append(1)
                    break
                elif true_e >= left and true_e < right: 
                    rw_.append(rw_list[ix])
                    break
        else: rw_.append(1)
    df['weight_rw'] = rw_
    df['weight_before_rw'] = df['weight']
    df['weight'] = df['weight'] * df['weight_rw']
    return df



# ------------------------------------------------------------------------------------
# Efficiency & Purity
# ------------------------------------------------------------------------------------
def return_sumw2(list_x, w_x, bins):
    sumw = []
    sumw2 = []
    for left, right in zip(bins, bins[1:]): 
        ix = np.where((list_x >= left) & (list_x <= right))[0] 
        sumw.append(np.sum(w_x[ix]))
        sumw2.append(np.sum(w_x[ix] ** 2))
    return sumw, sumw2

def return_efficiency(list_N, w_N, list_D, w_D, edges):
    N, bins = np.histogram(list_N, bins=edges, weights=w_N)
    sumN, deltaN = return_sumw2(list_N, w_N, edges)
    sigmaN = np.sqrt(deltaN)
    D, bins = np.histogram(list_D, bins=edges, weights=w_D)
    sumD, deltaD = return_sumw2(list_D, w_D, edges)
    sigmaD = np.sqrt(deltaD)
    #print(N)
    #print(sumN)
    #print(deltaN)
    #print(sigmaN)
    ratio = [x/y for x,y in zip(N,D)]
    yerr = []
    for ith in range(len(ratio)):
        if D[ith] > 0: yerr.append(np.sqrt((sigmaN[ith]/D[ith])**2 + (sigmaD[ith]*N[ith]/D[ith]/D[ith])**2))
        else: yerr.append(np.nan)
    return ratio, yerr

def return_purity(list_S, w_S, list_B, w_B, edges):
    list_A = []
    list_A_w2 = []
    for x,w in zip(list_S, w_S):
        A, bins = np.histogram(x, bins=edges, weights=w)
        w2 = return_sumw2(x, w, edges)
        list_A.append(A)
        list_A_w2.append(w2)
    list_C = []
    list_C_w2 = []
    for x,w in zip(list_B, w_B):
        B, bins = np.histogram(x, bins=edges, weights=w)
        w2 = return_sumw2(x, w, edges)
        list_C.append(B)
        list_C_w2.append(w2)
    N = []
    N_w2 = []
    for jth in range(len(list_A[0])):
        value = 0
        w2 = 0
        for ith in range(len(list_A)):
            value += list_A[ith][jth]
            w2 += list_A_w2[ith][jth]
        N.append(value)   
        N_w2.append(w2)   
    D = []
    D_w2 = []
    for jth in range(len(list_C[0])):
        value = 0
        w2 = 0
        for ith in range(len(list_C)):
            value += list_C[ith][jth]
            w2 += list_C_w2[ith][jth]
        D.append(value)   
        D_w2.append(w2)  
    D = [x+y for x,y in zip(N,D)]
    D_w2 = [x+y for x,y in zip(N_w2,D_w2)]
    sigmaN = np.sqrt(N_w2)
    sigmaD = np.sqrt(D_w2)
    ratio = [x/y for x,y in zip(N,D)]
    yerr = []
    for ith in range(len(x)):
        if D[ith] > 0: yerr.append(np.sqrt((sigmaN[ith]/D[ith])**2 + (sigmaD[ith]*N[ith]/D[ith]/D[ith])**2))
        else: yerr.append(np.nan)
    return ratio, yerr



# ------------------------------------------------------------------------------------
# Cross Section Functions
# ------------------------------------------------------------------------------------
def find_y(x,xvalues,yvalues):
    xvalues = [round(x,3) for x in xvalues]
    xvalues_ = xvalues.copy()
    xvalues_.append(x)
    xvalues_.sort()
    index = 0
    for ith,y in zip(range(len(xvalues_)),xvalues_):
        if y==x: 
            index = ith
            break
    if x in xvalues: 
        return yvalues[index]
    else: 
        low=index-1
        high=index
        m = (yvalues[high]-yvalues[low])/(xvalues[high]-xvalues[low])
        x_new = x-xvalues[low]
        return yvalues[low]+m*x_new

def interpolate(x1,x2,y1,y2):
    N=20
    deltax = (x2-x1)/N
    deltay = (y2-y1)/N
    X = [x1+deltax*ith for ith in range(N)]
    Y = [y1+deltay*ith for ith in range(N)]
    return X,Y

def get_mean_x(xlow, xhigh):
    X,Y = interpolate(xlow,xhigh,find_y(xlow,xvalues,yvalues),find_y(xhigh,xvalues,yvalues))
    yx = sum([a*b for a,b in zip(X,Y)])
    y = sum(Y)
    return yx/y

def binning(edges, weighted_bin=False):
    if weighted_bin: # asymmetric error
        diff_x = [get_mean_x(x,y) for x,y in zip(edges[:], edges[1:])]
        diff_ex_left = [abs(y-x) for x,y in zip(diff_x, edges)]
        diff_ex_right = [abs(y-x) for x,y in zip(edges[1:], diff_x)]
        diff_ex = np.array(list(zip(diff_ex_left, diff_ex_right))).T
    else: 
        diff_x = [x+abs(y-x)/2. for x,y in zip(edges[:], edges[1:])]
        diff_ex = [abs(y-x)/2. for x,y in zip(edges[:], edges[1:])]
    return diff_x, diff_ex

def corr_from_cov(cov):
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    return corr

def print_xs_info(folder, verbose=True):
    f1 = open('/home/gs627/LEEana/wiener_svd/'+folder+'/import_xsec.txt', 'r')
    f2 = open('/home/gs627/LEEana/wiener_svd/'+folder+'/import_frac_uncertainties.txt', 'r')
    #file_wiener = '/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root'
    #file_output = '/home/gs627/LEEana/wiener_svd/'+folder+'/output.root'
    content1 = f1.read()
    if verbose: print(content1)
    f1.close()
    content2 = f2.read()
    if verbose: print(content2)
    f2.close()
    
def reco_cov_matrix(folder):
    hcov_det = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_det'].values
    hcov_flux = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_flux'].values
    hcov_genie = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_genie'].values
    hcov_geant4 = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_geant4'].values
    hcov_rw = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_rw'].values
    hcov_rw_cor = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_rw_cor'].values
    hcov_rw_tot = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_rw_tot'].values
    hcov_tot = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hcov_tot'].values
    
    corr_det = corr_from_cov(hcov_det)
    corr_flux = corr_from_cov(hcov_flux)
    corr_genie = corr_from_cov(hcov_genie)
    corr_geant4 = corr_from_cov(hcov_geant4)
    corr_rw = corr_from_cov(hcov_rw)
    corr_rw_cor = corr_from_cov(hcov_rw_cor)
    corr_rw_tot = corr_from_cov(hcov_rw_tot)
    corr_tot = corr_from_cov(hcov_tot)
    
    A = [hcov_det, hcov_flux, hcov_genie, hcov_geant4, hcov_rw, hcov_rw_cor, hcov_rw_tot, hcov_tot]
    B = [corr_det, corr_flux, corr_genie, corr_geant4, corr_rw, corr_rw_cor, corr_rw_tot, corr_tot]
    return A, B
    
def get_matrices(folder):    
    hR = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/wiener.root')['hR'].values
    bias = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/output.root')['bias'].values
    smear = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/output.root')['smear'].values
    unfcov = uproot.open('/home/gs627/LEEana/wiener_svd/'+folder+'/output.root')['unfcov'].values
    return hR, bias, smear, unfcov

def plot_R(hR, ch, save_name=None):
    lines = True
    text = True
    plt.figure(figsize=(10,6.5))
    plt.imshow(hR, aspect="auto", origin='lower', cmap='viridis', norm=mpl.colors.LogNorm())   
    #my_cmap = copy.copy(plt.cm.get_cmap('viridis')) # get a copy of the default color map
    #my_cmap.set_bad(alpha=0) # hide 'bad' values 
    #lattice = hR
    #lattice[lattice == 0] = np.nan # insert 'bad' values into your lattice 
    #plt.imshow(lattice, aspect="auto", origin='lower', cmap=my_cmap)
    if lines: 
        jth = hR.shape[0]/ch
        for ith in range(1,ch+1): 
            #plt.axhline(jth*ith-1-0.5, color='white') # last bin
            plt.axhline(jth*ith-0.5, color='white')    # overflow bin
    # NEED FIXING
    #if text:
    #    mid_x = -1
    #    mid_y = [jth/2 + jth*x for x in range(ch)]
    #    for y in mid_y: plt.text(mid_x, y, 'HI', color='red')   
    plt.xlabel('True bin index', loc='right')
    plt.ylabel('Reconstructed bin index', loc='top')
    plt.colorbar(pad=0.01)
    plt.tight_layout()
    if save_name: plt.savefig('xs_plots/'+save_name+'_matR.png')
    if save_name: plt.savefig('xs_plots/'+save_name+'_matR.pdf')
    plt.show()

def plot_add_smear(smear, save_name=None):
    plt.figure(figsize=(7.5,6.5))
    #plt.title('Additional Smearing Matrix')
    plt.imshow(smear, aspect="auto", origin='lower', cmap='binary')
    my_cmap = copy.copy(plt.cm.get_cmap('viridis')) # get a copy of the default color map
    my_cmap.set_bad(alpha=0) # hide 'bad' values 
    lattice = smear
    lattice[lattice == 0] = np.nan # insert 'bad' values into your lattice 
    plt.imshow(lattice, aspect="auto", origin='lower', cmap=my_cmap)   
    plt.xlabel('True bin index', loc='right')
    plt.ylabel('True bin index', loc='top')
    plt.colorbar(pad=0.01)
    plt.tight_layout()
    if save_name: plt.savefig('xs_plots/'+save_name+'_add_smear.png')
    if save_name: plt.savefig('xs_plots/'+save_name+'_add_smear.pdf')
    plt.show()

def plot_bias(bias, save_name=None):
    plt.figure(figsize=(7.5,6.5))
    plt.title('Intrinsic Bias')
    x = [x for x,y in enumerate(bias)]
    plt.step(x, bias, where='pre', color='blue', lw=2)
    plt.axhline(0, color='grey', ls='--', lw=1, alpha=0.5)
    plt.xlim(x[0],x[-1])
    plt.xlabel('True bin index', loc='right')
    plt.ylabel('True bin index', loc='top')
    plt.tight_layout()
    if save_name: plt.savefig('xs_plots/'+save_name+'_bias.png')
    if save_name: plt.savefig('xs_plots/'+save_name+'_bias.pdf')
    plt.show()
    
def plot_true_cov_corr(unfcov, save_name=None):
    plt.figure(figsize=(7.5,6.5))
    #plt.title('Unfolded Covariance')
    plt.imshow(unfcov, aspect="auto", origin='lower', cmap='binary')
    my_cmap = copy.copy(plt.cm.get_cmap('viridis')) # get a copy of the default color map
    my_cmap.set_bad(alpha=0) # hide 'bad' values 
    lattice = unfcov
    lattice[lattice == 0] = np.nan # insert 'bad' values into your lattice 
    plt.imshow(lattice, aspect="auto", origin='lower', cmap=my_cmap, norm=mpl.colors.LogNorm())   
    plt.xlabel('True bin index', loc='right')
    plt.ylabel('True bin index', loc='top')
    plt.colorbar(pad=0.01)
    plt.tight_layout()
    if save_name: plt.savefig('xs_plots/'+save_name+'_true_cov.png')
    if save_name: plt.savefig('xs_plots/'+save_name+'_true_cov.pdf')
    plt.show()
    
    plt.figure(figsize=(7.5,6.5))
    #plt.title('Correlation Coefficient')
    plt.imshow(corr_from_cov(unfcov), aspect="auto", origin='lower', cmap='binary')
    my_cmap = copy.copy(plt.cm.get_cmap('viridis')) # get a copy of the default color map
    my_cmap.set_bad(alpha=0) # hide 'bad' values 
    lattice = corr_from_cov(unfcov)
    lattice[lattice == 0] = np.nan # insert 'bad' values into your lattice 
    plt.imshow(lattice, aspect="auto", origin='lower', cmap=my_cmap)    
    plt.xlabel('True bin index', loc='right')
    plt.ylabel('True bin index', loc='top')
    plt.colorbar(pad=0.01)
    plt.clim(-1, 1)
    plt.tight_layout()
    if save_name: plt.savefig('xs_plots/'+save_name+'_true_corr.png')
    if save_name: plt.savefig('xs_plots/'+save_name+'_true_corr.pdf')
    plt.show()
    
def plot_reco_cov_matrix(COV, CORR, ch, save_name=None):
    names = ['Detector', 'Flux', 'Genie', 'GEANT4', 'Reweighting_uncorr', 'Reweighting_corr', 'Reweighting_tot', 'Total']
    lines = True
    for zth,name in enumerate(names):
        plt.figure(figsize=(7.5,6.5))
        #plt.title(name+' Covariance')
        plt.imshow(COV[zth], aspect="auto", origin='lower')#, norm=mpl.colors.LogNorm())
        plt.xlabel('Reco bin index', loc='right')
        plt.ylabel('Reco bin index', loc='top')
        if lines: 
            jth = COV[zth].shape[0]/ch
            for ith in range(1,ch): 
                plt.axhline(jth*ith-0.5, color='white')    # overflow bin
                plt.vlines(jth*ith-0.5, 0-0.5, COV[zth].shape[0]-0.5, color='white')    # overflow bin
        plt.colorbar(pad=0.01)
        plt.tight_layout()
        if save_name: plt.savefig('xs_plots/'+save_name+'_reco_cov_'+name+'.png')
        if save_name: plt.savefig('xs_plots/'+save_name+'_reco_cov_'+name+'.pdf')
        plt.show()

        plt.figure(figsize=(7.5,6.5))
        #plt.title(name+' Correlation')
        plt.imshow(CORR[zth], aspect="auto", origin='lower') 
        plt.xlabel('Reco bin index', loc='right')
        plt.ylabel('Reco bin index', loc='top')
        if lines: 
            jth = COV[zth].shape[0]/ch
            for ith in range(1,ch): 
                plt.axhline(jth*ith-0.5, color='white')    # overflow bin
                plt.vlines(jth*ith-0.5, 0-0.5, COV[zth].shape[0]-0.5, color='white')    # overflow bin
        plt.colorbar(pad=0.01)
        plt.clim(-1, 1)
        plt.tight_layout()
        if save_name: plt.savefig('xs_plots/'+save_name+'_reco_corr_'+name+'.png')
        if save_name: plt.savefig('xs_plots/'+save_name+'_reco_corr_'+name+'.pdf')
        plt.show()

def visualize_xs_matrix(folder, ch=4, make_plots=False, save_name=None):
    COV, CORR = reco_cov_matrix(folder)
    #COV = [hcov_det, hcov_flux, hcov_genie, hcov_geant4, hcov_rw, hcov_rw_cor, hcov_rw_tot, hcov_tot]
    #CORR = [corr_det, corr_flux, corr_genie, corr_geant4, corr_rw, corr_rw_cor, corr_rw_tot, corr_tot]
    # Plot cov and corr currently missing
    hR, bias, smear, unfcov = get_matrices(folder)
    if make_plots:
        plot_R(hR, ch, save_name)
        plot_add_smear(smear, save_name)
        plot_bias(bias, save_name)
        plot_true_cov_corr(unfcov, save_name)
        
        plot_reco_cov_matrix(COV, CORR, ch, save_name)
        
def return_frac_err_tot(frac_err_stat, frac_err_mcstat, frac_err_dirt, frac_err_flux, frac_err_det, frac_err_geant4, frac_err_genie, frac_err_rw=None, frac_err_rw_cor=None):
    frac_err_pot = [0.022 for x in frac_err_stat]
    frac_err_tar = [0.011 for x in frac_err_stat]
    if frac_err_rw:
        frac_err_tot = [np.sqrt(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2+i**2+j**2+k**2) for a,b,c,d,e,f,g,h,i,j,k in zip(frac_err_stat,frac_err_mcstat,
                                                                                                                     frac_err_dirt,frac_err_flux,
                                                                                                                     frac_err_det,#frac_err_xs,
                                                                                                                     frac_err_geant4, frac_err_genie,
                                                                                                                     frac_err_rw,frac_err_rw_cor,
                                                                                                                     frac_err_pot,frac_err_tar)]
    else: 
        frac_err_tot = [np.sqrt(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2+i**2) for a,b,c,d,e,f,g,h,i,j,k in zip(frac_err_stat,frac_err_mcstat,
                                                                                                                     frac_err_dirt,frac_err_flux,
                                                                                                                     frac_err_det,#frac_err_xs,
                                                                                                                     frac_err_geant4, frac_err_genie,
                                                                                                                     #frac_err_rw,frac_err_rw_cor,
                                                                                                                     frac_err_pot,frac_err_tar)]
    return frac_err_tot

def plot_total_xs(edges, GENIE_y, y, yerr_stat, yerr_tot, yerr_sys, chi2, c, POT, xlabel, ylabel, xrange, yrange, weighted_bin=False, save_name=None):
    
    diff_x, diff_ex = binning(edges, weighted_bin)
    fig = plt.figure(figsize=(15,10.5))
    # ---------------------------------------------------------------------------------------------------
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.0)
    gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.07, height_ratios=(3.5,1))
    ax1 = fig.add_subplot(gsA[0,0])
    ax1.errorbar(diff_x, y[0], xerr=[0.], yerr=yerr_tot[0], color='magenta', label='Xp (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.errorbar(diff_x, y[0], xerr=[0.], yerr=yerr_stat[0], color='black', label='Xp (Stat.)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.hlines(GENIE_y[0], xrange[0], xrange[-1], colors='red', linestyles='solid', label=r'GENIE v3 MicroBooNE tune', lw=2)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\sigma$ [$10^{-39}$ cm$^{2}$ / nucleon]')
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    plt.xticks([])
    #ax1.legend(loc='lower center', ncol=1)

    ax2 = fig.add_subplot(gsA[1,0])
    ax2.errorbar(diff_x, [x/y for x,y in zip([y[0]],[GENIE_y[0]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[0]],[GENIE_y[0]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax2.errorbar(diff_x, [x/y for x,y in zip([y[0]],[GENIE_y[0]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[0]],[GENIE_y[0]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax2.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax2.set_xlabel('')
    ax2.axes.xaxis.set_ticklabels([])
    ax2.set_ylabel('Data/Prediction')
    ax2.set_xlim(xrange)
    ax2.set_ylim(0.,2)
    plt.xticks([])
    # ---------------------------------------------------------------------------------------------------
    gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.07, height_ratios=(3.5,1))
    ax3 = fig.add_subplot(gsB[0,0])
    ax3.errorbar(diff_x, y[1], xerr=[0.], yerr=yerr_tot[1], color='magenta', label='0p (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax3.errorbar(diff_x, y[1], xerr=[0.], yerr=yerr_stat[1], color='black', label='0p (Stat.)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax3.hlines(GENIE_y[1], xrange[0], xrange[-1], colors='red', linestyles='solid', label=r'GENIE v3 MicroBooNE tune', lw=2)
    ax3.axes.xaxis.set_ticklabels([])
    ax3.set_xlim(xrange)
    ax3.set_ylim(yrange)
    plt.xticks([])
    plt.yticks([])
    #ax3.legend(loc='upper center', ncol=1)

    ax4 = fig.add_subplot(gsB[1,0])
    ax4.errorbar(diff_x, [x/y for x,y in zip([y[1]],[GENIE_y[1]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[1]],[GENIE_y[1]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax4.errorbar(diff_x, [x/y for x,y in zip([y[1]],[GENIE_y[1]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[1]],[GENIE_y[1]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax4.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax4.set_xlabel('')
    ax4.axes.xaxis.set_ticklabels([])
    ax4.set_xlim(xrange)
    ax4.set_ylim(0.,2)
    plt.xticks([])
    plt.yticks([])
    # ---------------------------------------------------------------------------------------------------
    gsC = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], hspace=0.07, height_ratios=(3.5,1))
    ax5 = fig.add_subplot(gsC[0,0])
    ax5.errorbar(diff_x, y[2], xerr=[0.], yerr=yerr_tot[2], color='magenta', label='Np (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax5.errorbar(diff_x, y[2], xerr=[0.], yerr=yerr_stat[2], color='black', label='Np (Stat.)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax5.hlines(GENIE_y[2], xrange[0], xrange[-1], colors='red', linestyles='solid', label=r'GENIE v3 MicroBooNE tune', lw=2)
    ax5.axes.xaxis.set_ticklabels([])
    ax5.set_xlim(xrange)
    ax5.set_ylim(yrange)
    plt.xticks([])
    plt.yticks([])
    #ax5.legend(loc='upper center', ncol=1, prop={'size': 12})
    plt.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT), loc='best', ncol=1, frameon=True, framealpha=1.0)

    ax6 = fig.add_subplot(gsC[1,0])
    ax6.errorbar(diff_x, [x/y for x,y in zip([y[2]],[GENIE_y[2]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[2]],[GENIE_y[2]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax6.errorbar(diff_x, [x/y for x,y in zip([y[2]],[GENIE_y[2]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[2]],[GENIE_y[2]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax6.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax6.set_xlabel('')
    ax6.axes.xaxis.set_ticklabels([])
    ax6.set_xlim(xrange)
    ax6.set_ylim(0.,2)
    plt.xticks([])
    plt.yticks([])
    # ---------------------------------------------------------------------------------------------------
    if save_name: 
        plt.savefig('xs_plots/'+save_name+'.png')
        plt.savefig('xs_plots/'+save_name+'.pdf')
    plt.xticks([])
    plt.show()

def plot_single_xs(edges, GENIE_y, y, yerr_stat, yerr_tot, yerr_sys, chi2, c, POT, xlabel, ylabel, 
                   xrange, yrange, yrange2, unfcov, weighted_bin=False, save_name=None, generator=None):
    
    diff_x, diff_ex = binning(edges, weighted_bin)

    fig = plt.figure(figsize=(10,6.5))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.08)
    # ---------------------------------------------------------------------------------------------------
    # Panel A (left)
    gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.07, height_ratios=(3.5,1))
    # ---------------------------------------------------------------------------------------------------
    # Panel A top (left, top)
    ax1 = fig.add_subplot(gsA[0,0])
    ax1.errorbar(diff_x, y, xerr=diff_ex, yerr=yerr_tot, color=c, label='Data (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.errorbar(diff_x, y, xerr=diff_ex, yerr=yerr_stat, color='black', label='Data (Stat.)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    #ax1.plot(diff_x, GENIE_y, color='darkgreen', label='GENIE v3 MicroBooNE tune', markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    if generator: 
        if 'nuwro' in generator and len(generator['nuwro'])>0: ax1.plot(diff_x, generator['nuwro'], color='blue', label='NuWro 19.02.1 (%1.1f/%d)'%(calc_GoF(generator['nuwro'], y, unfcov),len(y)), markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
        if 'geniev2' in generator and len(generator['geniev2'])>0: ax1.plot(diff_x, generator['geniev2'], color='orange', label='GENIE v2.12.10 (%1.1f/%d)'%(calc_GoF(generator['geniev2'], y, unfcov),len(y)), markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
        if 'geniev3' in generator and len(generator['geniev3'])>0: ax1.plot(diff_x, generator['geniev3'], color='lime', label='GENIE v3.00.06 (%1.1f/%d)'%(calc_GoF(generator['geniev3'], y, unfcov),len(y)), markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
        if 'neut' in generator and len(generator['neut'])>0: ax1.plot(diff_x, generator['neut'], color='red', label='NEUT 5.4.0.1 (%1.1f/%d)'%(calc_GoF(generator['neut'], y, unfcov),len(y)), markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    ax1.axes.xaxis.set_ticklabels([])
    #ax1.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT)+'\n'+r'$\chi^{2}$/ndf=%1.1f/%d'%(calc_GoF(GENIE_y, y, unfcov),len(y)), loc='best', ncol=1)
    ax1.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT), loc='best', ncol=1)
    # ---------------------------------------------------------------------------------------------------
    # Panel A bottom (left, bottom)
    ax2 = fig.add_subplot(gsA[1,0])
    ratio = [x/y for x,y in zip(y,GENIE_y)]
    ratio_err = [x/y for x,y in zip(yerr_tot,GENIE_y)]
    ax2.errorbar(diff_x, ratio, xerr=diff_ex, yerr=ratio_err, color=c, fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.errorbar(diff_x, ratio, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat,GENIE_y)], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax2.set_xlabel(xlabel, loc='right')
    ax2.set_ylabel('Data/Pred.', labelpad=20)
    ax2.set_xlim(xrange)
    ax2.set_ylim(yrange2)
    # ---------------------------------------------------------------------------------------------------
    #plt.tight_layout()
    if save_name: 
        plt.savefig('xs_plots/'+save_name+'.png')
        plt.savefig('xs_plots/'+save_name+'.pdf')
    plt.show()
    sys = [np.sqrt(x**2 - y**2) for x,y in zip(yerr_tot, yerr_stat)] 
    print('stat/sys = ')
    print([round(x/y,3) for x,y in zip(yerr_stat,sys)])
    
def chi2_decomposition(pred, meas, unfcov, save_name=None):
    w, v = np.linalg.eig(unfcov)
    matrix_meas_temp = np.array(meas)
    matrix_pred_temp = np.array(pred)
    x = np.linspace(0,len(w)-1, 1001)
    y1 = [-1,1]
    y2 = [-2,2]
    y3 = [-3,3]
    plt.axhline(y3[0], color='red', alpha=0.2)
    plt.axhline(y3[1], color='red', alpha=0.2)
    plt.axhline(y2[0], color='orange', alpha=0.2)
    plt.axhline(y2[1], color='orange', alpha=0.2)
    plt.axhline(y1[0], color='green', alpha=0.2)
    plt.axhline(y1[1], color='green', alpha=0.2)
    plt.axhline(0, color='grey', alpha=0.2, ls='--')
    plt.fill_between(x, y3[0], y2[0], color='red', alpha=0.2)
    plt.fill_between(x, y2[0], y1[0], color='gold', alpha=0.2)
    plt.fill_between(x, y1[0], y1[1], color='lime', alpha=0.2)
    plt.fill_between(x, y1[1], y2[1], color='gold', alpha=0.2)
    plt.fill_between(x, y2[1], y3[1], color='red', alpha=0.2)
    plt.xlim(0,len(w)-1)
    plt.ylim(-5,5)
    matrix_delta_lambda = [x-y for x,y in zip(np.matmul(matrix_meas_temp,v),np.matmul(matrix_pred_temp,v))]
    print(sum([(x/np.sqrt(y))**2 for x,y in zip(matrix_delta_lambda,w)])) # Should be the chi-square value
    
    plt.plot(range(len(w)),[x/np.sqrt(y) for x,y in zip(matrix_delta_lambda,w)], marker='s', markersize=6, markerfacecolor='blue', markeredgecolor='black', linestyle = 'None')
    plt.text(len(w)-6, 3.25, r'Overall $\chi^{2}$/ndf = %1.1f/%i'%(calc_GoF(pred, meas, unfcov), len(w)))
    plt.xlabel('Bin Index', loc='right')
    plt.ylabel(r'$\epsilon_i$ value', loc='top')
    if save_name: 
        plt.savefig('xs_plots/'+save_name+'_chi2_decomp.png')
        plt.savefig('xs_plots/'+save_name+'_chi2_decomp.pdf')
    plt.tight_layout()
    plt.show()

def plot_2D_xs(edges, GENIE_y, y, yerr_stat, yerr_tot, yerr_sys, chi2, c, POT, xlabel, ylabel, 
               xrange, yrange, weighted_bin=False, save_name=None):
    
    diff_x, diff_ex = binning(edges, weighted_bin)    
    N = len(y)//4 # true bins per cross section
    
    fig = plt.figure(figsize=(20,13))
    fig.suptitle('MicroBooNE Preliminary\n'+'POT='+str(POT)+'\n'+r'$\chi^{2}$/ndf='+'%1.1f'%(float(chi2))+'/'+str(len(y)))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.08)
    gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.07, height_ratios=(3.5,1))
    # Panel A top (left, top) legend, ylabel, no xlabel
    ax1 = fig.add_subplot(gsA[0,0])
    ax1.errorbar(diff_x, y[:N], xerr=diff_ex, yerr=yerr_tot[:N], color=c, fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.errorbar(diff_x, y[:N], xerr=diff_ex, yerr=yerr_stat[:N], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.plot(diff_x, GENIE_y[:N], color='darkgreen', markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    ax1.axes.xaxis.set_ticklabels([])
    plt.xticks([])
    ax1.legend(title=r'cos$\theta$ < 0', loc='best', ncol=1)
    # ---------------------------------------------------------------------------------------------------
    # Panel A bottom (left, bottom) ylabel, xlabel
    ax2 = fig.add_subplot(gsA[1,0])
    ratio1 = [x/y for x,y in zip(y[:N],GENIE_y[:N])]
    ratio_err1 = [x/y for x,y in zip(yerr_tot[:N],GENIE_y[:N])]
    ax2.errorbar(diff_x, ratio1, xerr=diff_ex, yerr=ratio_err1, color=c, fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.errorbar(diff_x, ratio1, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat[:N],GENIE_y[:N])], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax2.set_xlabel(xlabel, loc='right')
    ax2.set_ylabel('Data/Prediction', labelpad=20)
    ax2.set_xlim(xrange)
    ax2.set_ylim(0, 2)
    # ---------------------------------------------------------------------------------------------------
    gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.07, height_ratios=(3.5,1))
    # Panel B top (right, top) legend, no ylabel, no xlabel
    ax3 = fig.add_subplot(gsB[0,0])
    ax3.errorbar(diff_x, y[N:N*2], xerr=diff_ex, yerr=yerr_tot[N:N*2], color=c, fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax3.errorbar(diff_x, y[N:N*2], xerr=diff_ex, yerr=yerr_stat[N:N*2], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax3.plot(diff_x, GENIE_y[N:N*2], color='darkgreen', markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    ax3.set_xlim(xrange)
    ax3.set_ylim(yrange)
    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])
    ax3.legend(title=r'0 < cos$\theta$ < 0.58', loc='best', ncol=1)
    # ---------------------------------------------------------------------------------------------------
    # Panel B bottom (right, bottom) no ylabel, xlabel
    ax4 = fig.add_subplot(gsB[1,0])
    ratio2 = [x/y for x,y in zip(y[N:N*2],GENIE_y[N:N*2])]
    ratio_err2 = [x/y for x,y in zip(yerr_tot[N:N*2],GENIE_y[N:N*2])]
    ax4.errorbar(diff_x, ratio2, xerr=diff_ex, yerr=ratio_err2, color=c, fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax4.errorbar(diff_x, ratio2, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat[N:N*2],GENIE_y[N:N*2])], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax4.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax4.set_xlabel(xlabel, loc='right')
    ax4.set_xlim(xrange)
    ax4.set_ylim(0, 2)
    ax4.axes.yaxis.set_ticklabels([])
    plt.yticks([])
    # ---------------------------------------------------------------------------------------------------
    gsC = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], hspace=0.07, height_ratios=(3.5,1))
    # Panel C top (left, top) legend, ylabel, no xlabel
    ax5 = fig.add_subplot(gsC[0,0])
    ax5.errorbar(diff_x, y[N*2:N*3], xerr=diff_ex, yerr=yerr_tot[N*2:N*3], color=c, fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax5.errorbar(diff_x, y[N*2:N*3], xerr=diff_ex, yerr=yerr_stat[N*2:N*3], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax5.plot(diff_x, GENIE_y[N*2:N*3], color='darkgreen', markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    ax5.set_ylabel(ylabel)
    ax5.set_xlim(xrange)
    ax5.set_ylim(yrange)
    ax5.axes.xaxis.set_ticklabels([])
    ax5.legend(title=r'0.58 < cos$\theta$ < 0.85', loc='best', ncol=1)
    # ---------------------------------------------------------------------------------------------------
    # Panel A bottom (left, bottom) ylabel, xlabel
    ax6 = fig.add_subplot(gsC[1,0])
    ratio3 = [x/y for x,y in zip(y[N*2:N*3],GENIE_y[N*2:N*3])]
    ratio_err3 = [x/y for x,y in zip(yerr_tot[N*2:N*3],GENIE_y[N*2:N*3])]
    ax6.errorbar(diff_x, ratio3, xerr=diff_ex, yerr=ratio_err3, color=c, fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax6.errorbar(diff_x, ratio3, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat[N*2:N*3],GENIE_y[N*2:N*3])], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax6.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax6.set_xlabel(xlabel, loc='right')
    ax6.set_ylabel('Data/Prediction', labelpad=20)
    ax6.set_xlim(xrange)
    ax6.set_ylim(0, 2)
    # ---------------------------------------------------------------------------------------------------
    gsD = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], hspace=0.07, height_ratios=(3.5,1))
    # Panel B top (right, top) legend, no ylabel, no xlabel
    ax7 = fig.add_subplot(gsD[0,0])
    ax7.errorbar(diff_x, y[N*3:], xerr=diff_ex, yerr=yerr_tot[N*3:], color=c, fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax7.errorbar(diff_x, y[N*3:], xerr=diff_ex, yerr=yerr_stat[N*3:], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax7.plot(diff_x, GENIE_y[N*3:], color='darkgreen', markersize=10, lw=2, markeredgewidth=2, alpha=0.5)
    ax7.set_xlim(xrange)
    ax7.set_ylim(yrange)
    ax7.axes.xaxis.set_ticklabels([])
    ax7.axes.yaxis.set_ticklabels([])
    ax7.legend(title=r'cos$\theta$ > 0.85', loc='best', ncol=1)
    # ---------------------------------------------------------------------------------------------------
    # Panel B bottom (right, bottom) no ylabel, xlabel
    ax8 = fig.add_subplot(gsD[1,0])
    ratio4 = [x/y for x,y in zip(y[N*3:],GENIE_y[N*3:])]
    ratio_err4 = [x/y for x,y in zip(yerr_tot[N*3:],GENIE_y[N*3:])]
    ax8.errorbar(diff_x, ratio4, xerr=diff_ex, yerr=ratio_err4, color=c, fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax8.errorbar(diff_x, ratio4, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat[N*3:],GENIE_y[N*3:])], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax8.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax8.set_xlabel(xlabel, loc='right')
    ax8.set_xlim(xrange)
    ax8.set_ylim(0, 2)
    ax8.axes.yaxis.set_ticklabels([])
    plt.yticks([])
    if save_name: 
        plt.savefig('xs_plots/'+save_name+'.png')
        plt.savefig('xs_plots/'+save_name+'.pdf')
    plt.show()

def plot_single_xs_err(edges, POT, xlabel, ylabel, frac_err_stat, frac_err_mcstat, frac_err_dirt, frac_err_flux, 
                       frac_err_det, frac_err_geant4, frac_err_genie, frac_err_rw, frac_err_rw_cor,
                       xrange, yrange, weighted_bin=False, save_name=None):
    
    diff_x, diff_ex = binning(edges, weighted_bin)

    frac_err_pot = [0.022]*len(diff_x)
    frac_err_tar = [0.011]*len(diff_x)
    frac_err_tot = [np.sqrt(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2+i**2+j**2+k**2) for a,b,c,d,e,f,g,h,i,j,k in zip(frac_err_stat,frac_err_mcstat,
                                                                                                                     frac_err_dirt,frac_err_flux,
                                                                                                                     frac_err_det,#frac_err_xs,
                                                                                                                     frac_err_geant4, frac_err_genie,
                                                                                                                     frac_err_rw,frac_err_rw_cor,
                                                                                                                     frac_err_pot,frac_err_tar)]

    fig = plt.figure(figsize=(9,7))
    plt.step(edges, [frac_err_stat[0]]+frac_err_stat, color='black', label='Stat.', lw=2, ls=':', where='pre', alpha=0.75)
    plt.step(edges, [frac_err_mcstat[0]]+frac_err_mcstat, color='springgreen', label='MC stat.', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_dirt[0]]+frac_err_dirt, color='brown', label='Dirt', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_flux[0]]+frac_err_flux, color='red', label='Flux', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_det[0]]+frac_err_det, color='magenta', label='Detector', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_geant4[0]]+frac_err_geant4, color='royalblue', label='GEANT4', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_genie[0]]+frac_err_genie, color='deepskyblue', label='GENIE', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_rw[0]]+frac_err_rw, color='darkgoldenrod', label='Reweight', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_rw_cor[0]]+frac_err_rw_cor, color='gold', label='Reweight corr.', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_pot[0]]+frac_err_pot, color='slategrey', label='POT', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_tar[0]]+frac_err_tar, color='cyan', label='N. targets', lw=2, where='pre', alpha=0.75)
    plt.step(edges, [frac_err_tot[0]]+frac_err_tot, color='black', label='Total', lw=3, where='pre', alpha=0.75)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks()
    plt.yticks()
    plt.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT), loc='best', ncol=3)
    # ---------------------------------------------------------------------------------------------------
    plt.tight_layout()
    if save_name: 
        plt.savefig('xs_plots/'+save_name+'.png')
        plt.savefig('xs_plots/'+save_name+'.pdf')
    plt.show()
    
def normalize_xs(edges, y, yerr_tot):
    delta_x = [abs(y-x)/2. for x,y in zip(edges[:], edges[1:])]
    area = 0
    for ith in range(len(delta_x)):
        area += abs(y[ith]*delta_x[ith])
        
    y = [item/area for item in y]
    yerr_tot = [item/area for item in yerr_tot]
    return y, yerr_tot


# ------------------------------------------------------------------------------------
# NUISANCE functions
# ------------------------------------------------------------------------------------
def is_NUISANCE_NC_noE(df):
    df_ = df[(df.cc == 0)]
    #df_ = df[(df.cc == 0)]
    return df_.reset_index(drop=True)


def is_NUISANCE_NC(df):
    df_ = df[(df.cc == 0) & (df.Enu_true > 0.275) & (df.Enu_true <= 4)]
    #df_ = df[(df.cc == 0)]
    return df_.reset_index(drop=True)

def is_NUISANCE_numuCC(df):
    df_ = df[(df.cc == 1)]
    return df_.reset_index(drop=True)

def is_NUISANCE_1pi0(df):
    df_ = df[(df.true_NprimPio == 1)]
    #df_ = df_.explode('true_pio_momentum').reset_index(drop=True) # Useful if true_pio_momentum is a list
    #return df_.explode('true_pio_costheta').reset_index(drop=True) # Useful if true_pio_costheta is a list
    return df_.reset_index(drop=True)

def is_NUISANCE_Npi0(df):
    df_ = df[(df.true_NprimPio > 0)]
    #df_ = df_.explode('true_pio_momentum').reset_index(drop=True)
    #return df_.explode('true_pio_costheta').reset_index(drop=True)
    return df_.reset_index(drop=True)

def is_NUISANCE_0p(df):
    df_ = df[(df.true_num_protons == 0)]
    return df_.reset_index(drop=True)

def is_NUISANCE_Np(df):
    df_ = df[(df.true_num_protons > 0)]
    return df_.reset_index(drop=True)

def count_pi0(df):
    truth_NprimPio = []
    true_tot_protons = []
    true_num_protons = []
    true_tot_protons_KEmax = []
    true_pio_momentum = []
    true_pio_costheta = []
 
    # Final state particles
    for n,p,x,y,z in zip(df.nfsp, df.pdg, df.px, df.py, df.pz):
    # Vertex particles
    #for n,p,x,y,z in zip(df.nvertp, df.pdg_init, df.px_init, df.py_init, df.pz_init):
        tot_p = 0
        num_p = 0
        num_pi0 = 0
        mom = -999.
        cos = -999.
        p_KEmax = -999.
        if(n>0):
            if(111 in p):
                num_pi0 = p.tolist().count(111)
                mom_ = -999.
                cos_ = -999.
                for pth,xth,yth,zth in zip(p,x,y,z):
                    if(pth==111):
                        mom_ = np.sqrt(xth**2 + yth**2 + zth**2)
                        cos_ = zth/np.sqrt(xth**2 + yth**2 + zth**2)
                        if(mom_ > mom): 
                            mom = mom_
                            cos = cos_
            if(2212 in p):
                ke_p = []
                for pth,xth,yth,zth in zip(p,x,y,z):
                    if(pth==2212):
                        m_p = 0.93827
                        e_p = np.sqrt((xth**2 + yth**2 + zth**2) + m_p**2)
                        ke_p.append(e_p - m_p)
                p_KEmax = max(ke_p)
                num_p = sum([ith>0.035 for ith in ke_p])
                tot_p = p.tolist().count(2212)
        
        truth_NprimPio.append(num_pi0)
        true_tot_protons.append(tot_p)
        true_num_protons.append(num_p)
        true_tot_protons_KEmax.append(p_KEmax)
        true_pio_momentum.append(mom)
        true_pio_costheta.append(cos)
                
    df['true_NprimPio'] = truth_NprimPio
    df['true_tot_protons'] = true_tot_protons
    df['true_num_protons'] = true_num_protons
    df['true_tot_protons_KEmax'] = true_tot_protons_KEmax
    df['true_pio_momentum'] = true_pio_momentum
    df['true_pio_costheta'] = true_pio_costheta
    return df

def NUISANCE_xs_total(df, flux0_10, flux0_4, nucleon=True):
    fScale = df.fScaleFactor[0]*flux0_10/flux0_4
    ftgtA = df.tgta[0]
    if nucleon==True: 
        # [1/nucleon]
        total_xs_pi0 = df.shape[0] *(fScale)
    else: 
        # [1/Ar]
        total_xs_pi0 = df.shape[0] *(fScale*ftgtA)
    return total_xs_pi0

def NUISANCE_xs_diff(df, flux0_10, flux0_4, edges, types='mom', nucleon=True):
    #types: {'mom', 'cos'}
    fScale = df.fScaleFactor[0]*flux0_10/flux0_4
    ftgtA = df.tgta[0]
    deltas = [round(y-x,4) for x,y in zip(edges, edges[1:])]

    diff_xs_pi0 = []
    if types=='mom': 
        for left, right in zip(edges, edges[1:]): diff_xs_pi0.append(df[(df.true_pio_momentum > round(left,4)) & (df.true_pio_momentum <= round(right,4))].shape[0])
    elif types=='cos': 
        for left, right in zip(edges, edges[1:]): diff_xs_pi0.append(df[(df.true_pio_costheta > round(left,4)) & (df.true_pio_costheta <= round(right,4))].shape[0])
    
    if nucleon==True: 
        # [1/nucleon]
        diff_xs_pi0 = [fScale*x/y for x,y in zip(diff_xs_pi0, deltas)]
    elif nucleon==False: 
        # [1/Ar]
        diff_xs_pi0 = [fScale*ftgtA*x/y for x,y in zip(diff_xs_pi0, deltas)]
    #return diff_xs_pi0
    # Already normalized to 1e-39
    return [x/1e-39 for x in diff_xs_pi0]

def NUISANCE_xs_2D_diff(df, flux0_10, flux0_4, edges, slice, types='mom', nucleon=True):
    #types: {'mom'}
    fScale = df.fScaleFactor[0]*flux0_10/flux0_4
    ftgtA = df.tgta[0]
    deltas = [round(y-x,4) for x,y in zip(edges, edges[1:])]

    df = df[(df.true_pio_costheta >= slice[0]) & (df.true_pio_costheta < slice[1])]

    diff_xs_pi0 = []
    if types=='mom': 
        for left, right in zip(edges, edges[1:]): diff_xs_pi0.append(df[(df.true_pio_momentum > round(left,4)) & (df.true_pio_momentum <= round(right,4))].shape[0])
    
    if nucleon==True: 
        # [1/nucleon]
        diff_xs_pi0 = [fScale*x/y for x,y in zip(diff_xs_pi0, deltas)]
    elif nucleon==False: 
        # [1/Ar]
        diff_xs_pi0 = [fScale*ftgtA*x/y for x,y in zip(diff_xs_pi0, deltas)]
    #return diff_xs_pi0
    # Already normalized to 1e-39
    return [x/1e-39 for x in diff_xs_pi0]

def calc_GoF(M_pred, M_data, cov):
    M = np.matrix([w-x for w,x in zip(M_pred, M_data)])
    Mt = M.transpose()
    cov_inv = np.linalg.inv(np.matrix(cov))
    Mret = np.matmul(M, np.matmul(cov_inv, Mt))
    return Mret[0,0]

def plot_total_xs_predictions(GENIE_y, y, yerr_stat, yerr_tot, 
                              geniev3_total, geniev2_total, neut_total, nuwro_total,
                              POT, factor, save_name=None):
    fig = plt.figure(figsize=(14,10))
    # ---------------------------------------------------------------------------------------------------
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.0)
    gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.07, height_ratios=(3.5,1))
    ax1 = fig.add_subplot(gsA[0,0])

    ax1.hlines(GENIE_y[0], 0, 2, colors='red', linestyles='solid', lw=2)
    ax1.hlines(geniev3_total[0]/factor, 0, 2, colors='orange', linestyles='solid', lw=2)
    ax1.hlines(geniev2_total[0]/factor, 0, 2, colors='gold', linestyles='solid', lw=2)
    ax1.hlines(neut_total[0]/factor, 0, 2, colors='blue', linestyles='solid', lw=2)
    ax1.hlines(nuwro_total[0]/factor, 0, 2, colors='lime', linestyles='solid', lw=2)
    ax1.errorbar([1], y[0], xerr=[0.], yerr=yerr_tot[0], color='magenta', label=r'WC $NC\pi^{0}$ (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.errorbar([1], y[0], xerr=[0.], yerr=yerr_stat[0], color='black', label=r'WC $NC\pi^{0}$ (Stat. only)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)

    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$\sigma_{NC\pi^{0}}$ [$10^{-38}$ cm$^{2}$ / Ar]', fontsize=14)
    ax1.set_xlim(0,2)
    ax1.set_ylim(0,2.5)
    plt.xticks([])
    plt.yticks(fontsize=14)
    ax1.legend(loc='upper center', ncol=1, prop={'size': 14}, frameon=True, framealpha=1.0)

    ax2 = fig.add_subplot(gsA[1,0])
    ax2.errorbar([1], [x/y for x,y in zip([y[0]],[GENIE_y[0]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[0]],[GENIE_y[0]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax2.errorbar([1], [x/y for x,y in zip([y[0]],[GENIE_y[0]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[0]],[GENIE_y[0]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax2.hlines(1, 0, 2, ls='--', color='black', lw=2, alpha=0.4)
    ax2.set_xlabel('')
    ax2.axes.xaxis.set_ticklabels([])
    ax2.set_ylabel('Data/Prediction', fontsize=14)
    ax2.set_xlim((0,2))
    ax2.set_ylim((0.,2))
    plt.xticks([])
    # ---------------------------------------------------------------------------------------------------
    gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.07, height_ratios=(3.5,1))
    ax3 = fig.add_subplot(gsB[0,0])

    ax3.hlines(GENIE_y[1], 0, 2, colors='red', linestyles='solid', label='GENIE v3 MicroBooNE tune', lw=2)
    ax3.hlines(geniev3_total[1]/factor, 0, 2, colors='orange', linestyles='solid', label='GENIE v3.00.06', lw=2)
    ax3.hlines(geniev2_total[1]/factor, 0, 2, colors='gold', linestyles='solid', label='GENIE v2.12.10', lw=2)
    ax3.hlines(neut_total[1]/factor, 0, 2, colors='blue', linestyles='solid', label='NEUT 5.4.0.1', lw=2)
    ax3.hlines(nuwro_total[1]/factor, 0, 2, colors='lime', linestyles='solid', label='NuWro 19.02.1', lw=2)
    ax3.errorbar([1], y[1], xerr=[0.], yerr=yerr_tot[1], color='magenta', label=r'WC $NC\pi^{0}$ 0p (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax3.errorbar([1], y[1], xerr=[0.], yerr=yerr_stat[1], color='black', label=r'WC $NC\pi^{0}$ 0p (Stat. only)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)

    ax3.axes.xaxis.set_ticklabels([])
    ax3.set_xlim(0,2)
    ax3.set_ylim(0,2.5)
    plt.xticks([])
    plt.yticks([])
    plt.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT), loc='upper center', ncol=1, prop={'size': 14}, frameon=True, framealpha=1.0)

    ax4 = fig.add_subplot(gsB[1,0])
    ax4.errorbar([1], [x/y for x,y in zip([y[1]],[GENIE_y[1]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[1]],[GENIE_y[1]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax4.errorbar([1], [x/y for x,y in zip([y[1]],[GENIE_y[1]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[1]],[GENIE_y[1]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax4.hlines(1, 0, 2, ls='--', color='black', lw=2, alpha=0.4)
    ax4.set_xlabel('')
    ax4.axes.xaxis.set_ticklabels([])
    ax4.set_xlim((0,2))
    ax4.set_ylim((0.,2))
    plt.xticks([])
    plt.yticks([])
    # ---------------------------------------------------------------------------------------------------
    gsC = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], hspace=0.07, height_ratios=(3.5,1))
    ax5 = fig.add_subplot(gsC[0,0])

    ax5.hlines(GENIE_y[2], 0, 2, colors='red', linestyles='solid', lw=2)
    ax5.hlines(geniev3_total[2]/factor, 0, 2, colors='orange', linestyles='solid', lw=2)
    ax5.hlines(geniev2_total[2]/factor, 0, 2, colors='gold', linestyles='solid', lw=2)
    ax5.hlines(neut_total[2]/factor, 0, 2, colors='blue', linestyles='solid', lw=2)
    ax5.hlines(nuwro_total[2]/factor, 0, 2, colors='lime', linestyles='solid', lw=2)
    ax5.errorbar([1], y[2], xerr=[0.], yerr=yerr_tot[2], color='magenta', label=r'WC $NC\pi^{0}$ Np (Total)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax5.errorbar([1], y[2], xerr=[0.], yerr=yerr_stat[2], color='black', label=r'WC $NC\pi^{0}$ Np (Stat. only)', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)

    ax5.axes.xaxis.set_ticklabels([])
    ax5.set_xlim(0,2)
    ax5.set_ylim(0,2.5)
    plt.xticks([])
    plt.yticks([])
    ax5.legend(loc='upper center', ncol=1, prop={'size': 14}, frameon=True, framealpha=1.0)

    ax6 = fig.add_subplot(gsC[1,0])
    ax6.errorbar([1], [x/y for x,y in zip([y[2]],[GENIE_y[2]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_tot[2]],[GENIE_y[2]])], color='magenta', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax6.errorbar([1], [x/y for x,y in zip([y[2]],[GENIE_y[2]])], xerr=[0.], yerr=[x/y for x,y in zip([yerr_stat[2]],[GENIE_y[2]])], color='black', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax6.hlines(1, 0, 2, ls='--', color='black', lw=2, alpha=0.4)
    ax6.set_xlabel('')
    ax6.axes.xaxis.set_ticklabels([])
    ax6.set_xlim((0,2))
    ax6.set_ylim((0.,2))
    plt.xticks([])
    plt.yticks([])
    # ---------------------------------------------------------------------------------------------------
    if save_name: plt.savefig(save_name+'.png')
    if save_name: plt.savefig(save_name+'.pdf')
    plt.xticks([])
    plt.show()

def plot_diff_xs_predictions(GENIE_y, y, yerr_stat, yerr_tot, unfcov,  
                             geniev3_diff, geniev2_diff, neut_diff, nuwro_diff,
                             edges, POT, factor, xlabel, ylabel, xrange, yrange, save_name=None):
    
    diff_x = [x+(y-x)/2 for x,y in zip(edges,edges[1:])]
    diff_ex = [(y-x)/2 for x,y in zip(edges,edges[1:])]
    
    fig = plt.figure(figsize=(12,9))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.08)
    # ---------------------------------------------------------------------------------------------------
    # Panel A (left)
    gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.07, height_ratios=(3.5,1))
    # ---------------------------------------------------------------------------------------------------
    # Panel A top (left, top)
    ax1 = fig.add_subplot(gsA[0,0])
    ax1.errorbar(diff_x, y, xerr=diff_ex, yerr=yerr_tot, color='magenta', label='Total uncertainty', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)
    ax1.errorbar(diff_x, y, xerr=diff_ex, yerr=yerr_stat, color='black', label='Stat. uncertainty', fmt='.', markersize=10, capsize=5, lw=2, markeredgewidth=2)

    chi2_GENIE = calc_GoF(GENIE_y, y, unfcov)
    chi2_geniev3 = calc_GoF([x/factor for x in geniev3_diff], y, unfcov)
    chi2_geniev2 = calc_GoF([x/factor for x in geniev2_diff], y, unfcov)
    chi2_neut = calc_GoF([x/factor for x in neut_diff], y, unfcov)
    chi2_nuwro = calc_GoF([x/factor for x in nuwro_diff], y, unfcov)
    ax1.plot(diff_x, GENIE_y, color='red', markersize=10, lw=2, markeredgewidth=2, alpha=0.5, label='[%1.1f/%i] GENIE v3 MicroBooNE tune'%(chi2_GENIE, len(diff_x)))
    ax1.plot(diff_x, [x/factor for x in geniev3_diff], color='orange', markersize=10, lw=2, markeredgewidth=2, alpha=0.5, label='[%1.1f/%i] GENIE v3.00.06'%(chi2_geniev3, len(diff_x)))
    ax1.plot(diff_x, [x/factor for x in geniev2_diff], color='gold', markersize=10, lw=2, markeredgewidth=2, alpha=0.5, label='[%1.1f/%i] GENIE v2.12.10'%(chi2_geniev2, len(diff_x)))
    ax1.plot(diff_x, [x/factor for x in neut_diff], color='blue', markersize=10, lw=2, markeredgewidth=2, alpha=0.5, label='[%1.1f/%i] NEUT 5.4.0.1'%(chi2_neut, len(diff_x)))
    ax1.plot(diff_x, [x/factor for x in nuwro_diff], color='lime', markersize=10, lw=2, markeredgewidth=2, alpha=0.5, label='[%1.1f/%i] NuWro 19.02.1'%(chi2_nuwro, len(diff_x)))

    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    ax1.axes.xaxis.set_ticklabels([])
    plt.yticks(fontsize=14)
    ax1.legend(title='MicroBooNE Preliminary\n'+'POT='+str(POT), loc='best', ncol=1, prop={'size': 14})
    # ---------------------------------------------------------------------------------------------------
    # Panel A bottom (left, bottom)
    ax2 = fig.add_subplot(gsA[1,0])
    ratio = [x/y for x,y in zip(y,GENIE_y)]
    ratio_err = [x/y for x,y in zip(yerr_tot,GENIE_y)]
    ax2.errorbar(diff_x, ratio, xerr=diff_ex, yerr=ratio_err, color='magenta', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.errorbar(diff_x, ratio, xerr=diff_ex, yerr=[x/y for x,y in zip(yerr_stat,GENIE_y)], color='black', fmt='.', markersize=10, capsize=3, lw=2, markeredgewidth=2)
    ax2.hlines(1, xrange[0], xrange[-1], ls='--', color='black', lw=2, alpha=0.4)
    ax2.set_xlabel(xlabel, loc='right', fontsize=14)
    ax2.set_ylabel('Data/Prediction', labelpad=20, fontsize=14)
    ax2.set_xlim(xrange)
    ax2.set_ylim(0, 2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # ---------------------------------------------------------------------------------------------------
    if save_name: plt.savefig(save_name+'.png')
    if save_name: plt.savefig(save_name+'.pdf')
    #plt.tight_layout()
    plt.show()