#/usr/bin/python2
import ROOT


def getSingleBiasCurve(rootfile):
    '''
    Return lists containing the bias curve points from a root file produced by CARE BiasCurve function

    Keyword arguments:
    rootfile : the fullpath of the rootfile containing the bias curve results
    '''
    
    ## Open the file
    f = ROOT.TFile(rootfile,'r')
    
    ## Get the point vectors
    biasCurve_vect = f.Get('BiasCurve/TVecBiasCurve')
    biasCurveError_vect = f.Get('BiasCurve/TVecBiasCurveError')
    biasCurveScanPoints_vect = f.Get('BiasCurve/TVecBiasCurveScanPoints')
    biasGroupRateVSTh_vect = f.Get('BiasCurve/TVecGroupRateVsThreshold')
    biasGroupRateVSThError_vect = f.Get('BiasCurve/TVecGroupRateVsThresholdError')
    numTel_vect = f.Get('BiasCurve/TVecNumTelescopes')
 


    print "==== Dump data stored in",f.GetName()+'/BiasCurve'
    print len(biasCurve_vect),len(biasCurveError_vect),len( biasCurveScanPoints_vect),len(biasGroupRateVSTh_vect),len(biasGroupRateVSThError_vect),len(numTel_vect)
    print "==== BiasCurve"
    for i,val in enumerate(biasCurve_vect) : print i,biasCurveScanPoints_vect[i],val,'+/-',biasCurveError_vect[i]
    print "==== GroupRateVSThError"
    for i,val in enumerate(biasGroupRateVSTh_vect) : print i,val,'+/-',biasGroupRateVSThError_vect[i]
    print "==== Num Tel"
    for val in numTel_vect: print val
    
    
    ## Fill list by looping over the points
    thresholds, rates, errors = [], [], []
    group_rates, group_errors = [], []
    for i,val in enumerate(biasCurve_vect) :
        thresholds.append(biasCurveScanPoints_vect[i])
        rates.append(val)
        errors.append(biasCurveError_vect[i])
        group_rates.append(biasGroupRateVSTh_vect[i])
        group_errors.append(biasGroupRateVSThError_vect[i])
    f.Close()
    ## return 
    return thresholds, rates, errors, group_rates, group_errors


def getBiasCurve(runNames,baseDir):
    '''
    Return a directory containing the bias curve points from a set of identical CARE BiasCurve runs

    Keyword arguments:
    baseDir : the GORCA output directory
    runNames : a list of names of run to consider or the name of a single run  

    '''
    print runNames 
    import glob
    ## Loop over the run to combine
    dict_res =  {'threshold':[],'rate':[],'rate_err':[], 'group_rate':[],'group_rate_err':[] }
    for runName in runNames:
        ## Loop on the files in this run.
        for i,f in enumerate(glob.glob(baseDir+'/'+runName+'/BiasCurve/Roots/Run*.root')):
            ## Collect the data
            th, ra, ra_er, gr, gr_er = getSingleBiasCurve(f)
            for i,t in enumerate(th):
                if t in dict_res['threshold']:
                    dict_res['rate'][dict_res['threshold'].index(t)].append(ra[i])
                    dict_res['rate_err'][dict_res['threshold'].index(t)].append(ra_er[i])
                    dict_res['group_rate'][dict_res['threshold'].index(t)].append(gr[i])
                    dict_res['group_rate_err'][dict_res['threshold'].index(t)].append(gr_er[i])
                else:
                    #print 'new Threshold',t,'append',[ra[i]]
                    dict_res['threshold'].append(t)
                    dict_res['rate'].append([ra[i]])
                    dict_res['rate_err'].append([ra_er[i]])
                    dict_res['group_rate'].append([gr[i]])
                    dict_res['group_rate_err'].append([gr_er[i]])

    ### little manipulation of the list to return the appropriate mean,combined erros etc...
    lenMax = -np.inf
    ## Get the maximum number of runs containing the same threshold
    for x in dict_res['rate']:
        lenMax = max(lenMax,len(x))
    
    for k in dict_res.keys():
        if k == 'threshold':
            dict_res[k]=np.array(dict_res[k])
            continue
        ## Fill with 0. list of data for a given threshold that have less runs than the max
        for x in dict_res[k]:
            while len(x)!=lenMax: x.append(np.nan)
        ## Transpose
        dict_res[k]=np.array(dict_res[k])
        ## Combine errors
        if k.count('err')>0.5:
            dict_res[k]=np.sqrt(np.nansum(np.power(dict_res[k],2),axis=1)/float(dict_res[k].shape[0]))
        ## Get mean
        else:
            print dict_res[k]
            dict_res[k]=np.nanmean(dict_res[k],axis=1)
    
    ## Now reorder
    order = np.argsort(dict_res['threshold'])
    for k in dict_res.keys():
        dict_res[k]=dict_res[k][order]
    return dict_res 


if __name__ == '__main__':
    
    import numpy as np
    from ROOT import TFile
    from ROOT import kBlack,kBlue,kRed,kGreen,kYellow,kCyan,kMagenta
    
    from optparse import OptionParser
    
    parser = OptionParser()
    
    parser.add_option("--gorcadirectory", dest="basedir",
                      help="""The directory containing the root files produced from CARE bias curve macro
                      This macro assumes the format is the following: <gorcadirectory>/<runname>//BiasCurve/Roots/Run*.root""",
                      default = '/data/datasets/CTA/GORCA_Output/biasCurve_N1D1_M1_7_19_NSB_1_2_5_Max')
    
    parser.add_option("-r","--runnames", dest="runnames",
                      help="List of runs to consider: ':' separates different settings ',' separates subruns")
    
    parser.add_option("-b","--batch", dest="batch", default=False,
                      action='store_true',help="Run in batch mode")

    parser.add_option("-s","--save", dest="save",
                      help="Name of the saved pdf file, if not specified the file is not saved",default='')
    
    
    parser.add_option("-l","--runlabels", dest="runlabels",
                      help="labels of the different settings, separated by ':'")
    
    parser.add_option("-t","--title", dest="title",
                      help="Title of the plot")
    
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")

    parser.add_option("-g", "--gain", dest="gain", type = 'float',
                      help="Gain of the FADC")

    parser.add_option( "--line", dest="line", default = '-,--,-.,:',
                      help="Line style")
    
    parser.add_option( "--mark", dest="mark", default = 'o,s,^,v,<,>,x',
                      help="Mark style")
    
    parser.add_option( "--color", dest="color", default = 'k,b,r,g,y,c,m',
                      help="Mark style")

    global options
    options, args = parser.parse_args()
    
    options.lines  = lambda i: options.line.split(',')[i%4]
    options.marks  = lambda i: options.mark.split(',')[i%7]
    colROOT = {'k':kBlack,'b':kBlue,'r':kRed,'g':kGreen,'y':kYellow,'c':kCyan,'m':kMagenta}
    #markROOT = {'o':,'s':kBlue,'^':kRed,'v':kGreen,'<':kYellow,'>':kCyan,'x':kMagenta}
    options.colorsROOT  = lambda i: colROOT[options.color.split(',')[i%7]]

    import sys,inspect,os
    sys.path.append(os.environ['GORCAROOT']+'/GANA/modules')
    #from gana_rootlogon import baseROOTPlotDefinition
    #baseROOTPlotDefinition()
    import gana_plots as gp
    
    # Get the groups of runs and corresponding labels
    labels = [s for s in options.runlabels.split(':')]
    dictBiases = {}
    for i,l in enumerate(labels):
        print l
        dictBiases[l] = getBiasCurve([s for s in options.runnames.split(':')[i].split(',')],options.basedir)

    ## Create the corresponding plots
    plots = {'can':{},'graph':{},'legend':{},'axis':{}}
    gp.createBiasCurvePlot(dictBiases,labels,options,plots)
    if options.save != '': plots['can']['biasCurve'].SaveAs(options.save)

    if not options.batch: raw_input("\n == Press enter to close ==\n")
