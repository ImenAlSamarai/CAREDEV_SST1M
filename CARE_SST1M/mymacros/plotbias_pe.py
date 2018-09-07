import biasCurve
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

def func(x, a, b,c):

    return a *np.exp(-b * x) + c


fig = plt.figure()
axes1 = fig.add_subplot(111)
axes2 = axes1.twiny()
# Add some extra space for the second axis at the bottom
fig.subplots_adjust(bottom=0.2)

######### CARE ##########


dict_res =  {'threshold':[],'rate':[],'rate_err':[], 'group_rate':[],'group_rate_err':[] }
th, ra, ra_er, gr, gr_er = biasCurve.getSingleBiasCurve('test1105_length2500_NSB266_sigmae04_10000trials.root.root')

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
    thresh=np.append(thresh,7 * i)


rate_blind = []
for j in rate:
    rate_blind =np.append(rate_blind,j*1./50.)


rateerr_blind = []
for j in rate_err:
    rateerr_blind =np.append(rateerr_blind,j*1./50.)

axes1.plot(thresh, rate_blind ,color = 'r', linestyle = '-')
axes1.fill_between(thresh,rate_blind-rateerr_blind, rate_blind+rateerr_blind, alpha = 0.3, color = 'r',label = 'CARE')
axes1.legend()



######### TOY ##########


# NEW Real measured values per pixel (gain, dark count rate, sigma_e)
BC_toy_perpix_05112018 = np.load('test_bias_curve_perpix_05112018.npz')
threshold_perpix_05112018 = BC_toy_perpix_05112018['thresholds']
rate_toy_perpix_05112018 = BC_toy_perpix_05112018['rate']
raterr_toy_perpix_05112018 = BC_toy_perpix_05112018['rate_error']

axes1.plot(threshold_perpix_05112018, rate_toy_perpix_05112018 * 1E9, color = 'g')
axes1.fill_between(threshold_perpix_05112018, rate_toy_perpix_05112018 * 1E9-raterr_toy_perpix_05112018 * 1E9, rate_toy_perpix_05112018 * 1E9+raterr_toy_perpix_05112018 * 1E9, label='TOY MC', color = 'g', alpha = 0.3)
######### Data ##########

data_1 = np.loadtxt("bias_curve_light_off_mask_cluster.txt")
data_2 = np.loadtxt("bias_curve_dark_50_samples.txt")
axes1.errorbar(data_1[:,0],data_1[:,1], yerr= 1./np.sqrt(10.*data_1[:,1]), label = 'DATA',color =  'k')


# Plotting

axes1.legend()
axes1.grid()
axes1.set_yscale('log')
axes1.set_ylim(1.e2, 1.e7)
axes1.set_xlim(0., 110.)
axes1.set_xlim(0., 110.)

axes1.set_xlabel('Threshold/ Cluster [ADC]')
axes1.set_ylabel('Trigger rate [Hz]')
axes1.set_xticks(np.arange(0, 110 + 1, 10.0))


new_tick_locations = np.linspace(0.,110,11.6)

def tick_function(X):
    V = data_1[:,0]/5.8
    return ["%.1f" % z for z in V]


# Move twinned axis ticks and label from top to bottom
axes2.xaxis.set_ticks_position("bottom")
axes2.xaxis.set_label_position("bottom")

# Offset the twin axis below the host
axes2.spines["bottom"].set_position(("axes", -0.15))

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
axes2.set_frame_on(True)
axes2.patch.set_visible(False)
for sp in axes2.spines.itervalues():
    sp.set_visible(False)
axes2.spines["bottom"].set_visible(True)

plt.title('Trigger rates vs. threshold (1 cluster = 21 pixels)')
axes2.set_xlim(axes1.get_xlim())
axes2.set_xticks(new_tick_locations)
axes2.set_xticklabels(tick_function(new_tick_locations))
axes2.set_xlabel('Threshold/ Cluster [PE]')

##the original way to fit

'''index = np.where(data_1[:,1]<2.e6)
fit_region = data_1[index,0].flatten()
y_fitregion=  data_1[index,1].flatten()

print y_fitregion.shape
print fit_region.shape
popt, pcov = curve_fit(func, thresh, rate_blind, p0=(50, -100, 1))
yy = func(fit_region, *popt)
plt.plot(fit_region,y_fitregion,':', fit_region, yy)
print popt'''
plt.show()

