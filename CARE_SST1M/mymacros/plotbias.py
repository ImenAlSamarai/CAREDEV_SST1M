import biasCurve
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
#import GridSpec

fig, axes1 = plt.subplots()

#fig = plt.figure()
#axes = plt.subplot2grid((2, 1), (0, 0))

######### CARE ##########

#axes1 = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan = 2)
#axes2 = plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan = 2)

dict_res =  {'threshold':[],'rate':[],'rate_err':[], 'group_rate':[],'group_rate_err':[] }
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_ter.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample200_length1000_DOWN.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample200_length1000_UP.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample500.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample500_length2000.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample500_length2000_lesstrials.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample500_length1000_lesstrials.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample250_length1500_lesstrials.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample200_length2500_3MHz_04.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample200_length2500_3MHz_sigmae14.root')
#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_startsample200_length2500266MHz_sigmae14.root')


#th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_length2500_NSB266_sigmae04_10000trials.root.root')
th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('../../output/mytest/BiasCurve/Roots/careRun000001.root')
#DOWN
for i,t in enumerate(th):
                if t in dict_res['threshold']:
                    dict_res['rate'][dict_res['threshold'].index(t)].append(ra[i])
                    dict_res['rate_err'][dict_res['threshold'].index(t)].append(ra_er[i])
                    dict_res['group_rate'][dict_res['threshold'].index(t)].append(gr[i])
                    dict_res['group_rate_err'][dict_res['threshold'].index(t)].append(gr_er[i])
                else:
                    dict_res['threshold'].append(t)
                    dict_res['rate'].append([ra[i]])
                    dict_res['rate_err'].append([ra_er[i]])
                    dict_res['group_rate'].append([gr[i]])
                    dict_res['group_rate_err'].append([gr_er[i]])

rate = [item for sublist in dict_res['rate'] for item in sublist]
rate_err= [item for sublist in dict_res['rate_err'] for item in sublist]
thresh = []
for i in dict_res['threshold']:
    thresh=np.append(thresh, i*7)


rate_blind = []
for j in rate:
    rate_blind =np.append(rate_blind,j*1./50.)


rateerr_blind = []
for j in rate_err:
    rateerr_blind =np.append(rateerr_blind,j*1./50.)

axes1.errorbar(thresh, rate_blind , yerr=rateerr_blind, label = 'CARE (x-talk = 8%)',color = 'r', linestyle = '-')

#UP
dict_res1 =  {'threshold':[],'rate':[],'rate_err':[], 'group_rate':[],'group_rate_err':[] }
#th_UP, ra_UP, ra_er_UP, gr_UP, gr_er_UP = biasCurve.getSingleBiasCurve('test1105_length2500_NSB266_sigmae14_10000trials.root')
#th_UP, ra_UP, ra_er_UP, gr_UP, gr_er_UP = biasCurve.getSingleBiasCurve('test1105_length2500_NSB266_sigmae04_10000trials._sigma_4percent.root')
'''th_UP, ra_UP, ra_er_UP, gr_UP, gr_er_UP = biasCurve.getSingleBiasCurve('test1105_length2500_NSB266_sigmae04_xtalk006_10000trials..root')

for i_UP,t_UP in enumerate(th_UP):
                if t_UP in dict_res1['threshold']:
                    dict_res1['rate'][dict_res['threshold'].index(t_UP)].append(ra_UP[i_UP])
                    dict_res1['rate_err'][dict_res['threshold'].index(t_UP)].append(ra_er_UP[i_UP])
                    dict_res1['group_rate'][dict_res['threshold'].index(t_UP)].append(gr_UP[i_UP])
                    dict_res1['group_rate_err'][dict_res['threshold'].index(t_UP)].append(gr_er_UP[i_UP])
                else:
                    dict_res1['threshold'].append(t_UP)
                    dict_res1['rate'].append([ra_UP[i_UP]])
                    dict_res1['rate_err'].append([ra_er_UP[i_UP]])
                    dict_res1['group_rate'].append([gr_UP[i_UP]])
                    dict_res1['group_rate_err'].append([gr_er_UP[i_UP]])

rate_UP = [item for sublist in dict_res1['rate'] for item in sublist]
rate_err_UP= [item for sublist in dict_res1['rate_err'] for item in sublist]
thresh_UP = []
for i_UP in dict_res1['threshold']:
    thresh_UP=np.append(thresh_UP,7 * i_UP)


rate_blind_UP = []
for j_UP in rate_UP:
    rate_blind_UP =np.append(rate_blind_UP,j_UP*1./50.)


rateerr_blind_UP = []
for j_UP in rate_err_UP:
    rateerr_blind_UP =np.append(rateerr_blind_UP,j_UP*1./50.)

axes1.errorbar(thresh_UP, rate_blind_UP , yerr=rateerr_blind_UP, label = 'CARE (x-talk = 6%) ',color = 'r', linestyle = ':')



axes1.legend()

'''

######### TOY ##########

# Toy BC: read NPZ file
# mean and sigma of inputs approximated with a Gaussian function, this is to compare with CARE whichuses such an approximation
BC_toy_gauss = np.load('test_bias_curve_gauss.npz')
threshold_gauss = BC_toy_gauss['thresholds']
rate_toy_gauss = BC_toy_gauss['rate']
raterr_toy_gauss = BC_toy_gauss['rate_error']

# Real measured values per pixel (gain, dark count rate, sigma_e)
BC_toy_perpix = np.load('test_bias_curve_perpix.npz')
threshold_perpix = BC_toy_perpix['thresholds']
rate_toy_perpix = BC_toy_perpix['rate']
raterr_toy_perpix = BC_toy_perpix['rate_error']


# NEW Real measured values per pixel (gain, dark count rate, sigma_e)
BC_toy_perpix_05112018 = np.load('test_bias_curve_perpix_05112018.npz')
threshold_perpix_05112018 = BC_toy_perpix_05112018['thresholds']
rate_toy_perpix_05112018 = BC_toy_perpix_05112018['rate']
raterr_toy_perpix_05112018 = BC_toy_perpix_05112018['rate_error']

# NEW Real measured values per pixel (gain, dark count rate, sigma_e) with 10% less sigma_e
BC_toy_perpix_05112018_se01 = np.load('test_bias_curve_perpix_05112018_se01.npz')
threshold_perpix_05112018_se01 = BC_toy_perpix_05112018_se01['thresholds']
rate_toy_perpix_05112018_se01 = BC_toy_perpix_05112018_se01['rate']
raterr_toy_perpix_05112018_se01 = BC_toy_perpix_05112018_se01['rate_error']


######### Data ##########


hdul = fits.open('DigicamSlowControl_20180506_000.fits.gz')
#print hdul.info()
data = np.array(hdul[1].data)
#print(data['biasCurveTriggerRate'].shape)
#print(data['biasCurvePatch7Threshold'].shape)
print(hdul[1].data.names)
#fig = plt.figure()
#plt.hist(data['Crate1_T'][0], bins = 50)
#plt.hist(data['Crate2_T'][0], bins = 50)
#plt.hist(data['Crate3_T'][0], bins = 50)
#plt.show()
#axes1.plot(data['biasCurvePatch7Threshold'][0],data['biasCurveTriggerRate'][0], label = 'Data',color =  'k')



###### Data from txt ##########

data_1 = np.loadtxt("bias_curve_light_off_mask_cluster.txt")
data_2 = np.loadtxt("bias_curve_dark_50_samples.txt")
#data_2 = np.loadtxt("bias_curve_light_off.txt")
axes1.errorbar(data_1[:,0],data_1[:,1], yerr= 1./np.sqrt(10.*data_1[:,1]), label = 'DATA',color =  'k')
#axes1.plot(data_2[:,0],data_2[:,1], label = 'DATA05142018.txt',color =  'k',linestyle = '--')


# Plotting


#axes1.errorbar(threshold_gauss, rate_toy_gauss * 1E9, yerr=raterr_toy_gauss * 1E9,
 #             label='TOY(Gauss. dispersion of gain, $\sigma_e$ )',color =  'r')
#axes1.errorbar(threshold_perpix, rate_toy_perpix * 1E9, yerr=raterr_toy_perpix * 1E9,
#label='TOY (Per pixel parameters in camera)', color= 'g')
axes1.errorbar(threshold_perpix_05112018, rate_toy_perpix_05112018 * 1E9, yerr=raterr_toy_perpix_05112018 * 1E9,
              label='TOY MC', color = 'g')
#axes1.errorbar(threshold_perpix_05112018_se01, rate_toy_perpix_05112018_se01 * 1E9, yerr=raterr_toy_perpix_05112018_se01 * 1E9,
#label='TOY MC (10% less electronic noise)', color = 'g', linestyle = ':')

axes1.legend()
axes1.grid()
axes1.set_yscale('log')
axes1.set_ylim(1.e2, 1.e7)
axes1.set_xlim(0., 110.)
axes1.set_xlim(0., 110.)

axes1.set_xlabel('Threshold [ADC]')
axes1.set_ylabel('Trigger rate [Hz]')
axes1.set_xticks(np.arange(0, 110 + 1, 10.0))
#axes1.set_xticks(np.arange(0, 200 + 1, 10.0))
#axes1.set_ylabel('Ratio')
#axes1.grid()


#fig.subplots_adjust(hspace=0.)
#plt.setp(axes2.get_xticklabels(), visible=True)
#plt.setp(axes1.get_xticklabels(), visible=False)
#The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
#axes2.set_yticks(axes2.get_yticks()[1:])



#axes2.plot(data['biasCurvePatch7Threshold'][0], (rate_toy_perpix * 1E9)/(data['biasCurveTriggerRate'][0]),
# label='TOY (Per pixel parameters in camera)', color= 'k')
#axes2.plot(data['biasCurvePatch7Threshold'][0], np.ones(len(threshold_perpix)),
#label='TOY (Per pixel parameters in camera)', color= 'k', linestyle = ':')
'''axes2.plot(threshold_perpix, rate_toy_perpix * 1E9/rate_blind,
              label='TOY (Per pixel parameters in camera)', color= 'r')

'''




plt.show()
