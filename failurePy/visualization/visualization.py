"""
Module of our various visualization functions. These are a little rough around the edged, but now in a common place
"""
#Long plotting library pylint: disable=too-many-lines
import os
import sys
import pickle
import numbers
import warnings
import numpy as onp
#Ignore matplotlib warnings'
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

#Import the saved data, this is not compatible with python < 3.9!!
import failurePy

warnings.filterwarnings( "ignore", module = r"matplotlib\..*" )


def plotAvgsOnly(avgRewards,avgSuccessRates,avgWallClockTimes,nSimulationsPerTrees,nTrialsPerPoint,noise,solverNames, #Main plotting function, so pylint: disable=too-many-statements,too-many-branches
                 successRateAdjustmentFlag=False,systemName=None,dt=1,plotSuccessRatesAndWallClockTimesFlag=False):
    """
    Method that loads data to plot and calls plotting functions as needed

    Parameters
    ----------
    avgRewards : array, shape(numSolvers,numNSimulationsPerTrees,numExperimentSteps+1)
        The average rewards for each solver and number of simulations per tree, for each time step (including initial time)
    avgSuccessRates : array, shape(numSolvers,numNSimulationsPerTrees)
        The average success rate for each solver and number of simulations per tree
    avgWallClockTimes : array, shape(numSolvers,numNSimulationsPerTrees)
        The average time for each solver and number of simulations per tree
    nSimulationsPerTrees : array, shape(numNSimulationsPerTrees)
        The number of simulations performed before returning an action (when not running on time out mode).
        This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
    nTrialsPerPoint : int
        The number of repeated trials per configuration.
    noise : float
        Level of sigma for this collection of experiments
    solverNames : list
        List of the names of each solver that was used
    successRateAdjustmentFlag : bool (default=False)
        If true, rewards are scaled by the success rate
    systemName : str (default=None)
        Name of the system for the plot title, if any
    dt : float (default=1)
        The time between time steps of the experiment
    plotSuccessRatesAndWallClockTimesFlag : boolean (default=False)
        If true plot extra data on success rates and wall clock times
    """

    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    #numPoints = len(nSimulationsPerTrees)
    offset = .15

    #Plotting success rates and wall clock times
    if plotSuccessRatesAndWallClockTimesFlag:

        #Set random to 1 simulation, so it plots on a log scale well
        semiLogNSimulationsPerTrees = onp.copy(nSimulationsPerTrees)
        if semiLogNSimulationsPerTrees[0] == 0:
            semiLogNSimulationsPerTrees[0] = 1

        dummyFig, ax = plt.subplots(nrows=2,ncols=1,figsize=(15,8))

        #Plot success rates
        legendHandles = []
        for iSolver, solverName in enumerate(solverNames):
            legendHandles.append(ax[0].semilogx(semiLogNSimulationsPerTrees,100*avgSuccessRates[iSolver],label=solverName,marker="o")[0])
        ax[0].legend(handles=legendHandles)
        ax[0].set_ylabel("Success Rate (%)")

        #Plot wall clock times
        legendHandles = []
        for iSolver, solverName in enumerate(solverNames):
            legendHandles.append(ax[1].semilogx(semiLogNSimulationsPerTrees,avgWallClockTimes[iSolver],label=solverName,marker="o")[0])

        ax[1].legend(handles=legendHandles)
        ax[1].set_ylabel("Wall Clock Time/Iteration (s)")

        ax[-1].set_xlabel("Number of Simulations per Tree")

        ax[0].set_title(f"Success Rates and Wall Clock Times\n{nTrialsPerPoint} experiments averaged per data point")


    #Now plot reward trajectories
    dummyFig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,8))

    #Make x-axis
    timeSteps = onp.arange(0,dt*(len(avgRewards[0,0])),dt)
    #Labels for legend
    legendHandles = []

    #Legend Entries
    lineStyles = ["-",":","--","dashdot"]
    markers = ["o","^","x","*"]
    for iSolver, solverName in enumerate(solverNames):

        if successRateAdjustmentFlag:
            successFactor = avgSuccessRates[iSolver,0]
        else:
            successFactor = 1
        legendHandles.append(ax.plot(timeSteps[0],avgRewards[0,0,0]*successFactor,label=solverName,ls=lineStyles[iSolver],marker=markers[iSolver],c="black")[0])

    #Set color cycle
    setColorCycler(len(nSimulationsPerTrees),ax)

    #Loop through solvers
    for iSolver, solverName in enumerate(solverNames):
        #Loop through experiments
        for jNSimsPerTreeTrial,nSimulationsPerTree in enumerate(nSimulationsPerTrees):
            #Adjust by success rates if needed
            if successRateAdjustmentFlag:
                successFactor = avgSuccessRates[iSolver,jNSimsPerTreeTrial]
            else:
                successFactor = 1
            #Check for random policy
            if nSimulationsPerTree == 0:
                #Only plot once
                if iSolver == 0:
                    label = "Random Action"
                    #Plot rewards, modulated by success rate

                    handle = ax.plot(timeSteps,avgRewards[iSolver,jNSimsPerTreeTrial,:]*successFactor,label=label,ls="--",marker="*")[0]
                    legendHandles.append(handle)
                #Need to advance color cycler
                else:
                    ax.plot([],[])
                continue

            label = f"N = {nSimulationsPerTree}"
            handle = ax.plot(timeSteps+offset*jNSimsPerTreeTrial,avgRewards[iSolver,jNSimsPerTreeTrial,:]*successFactor,label=label,ls=lineStyles[iSolver],marker=markers[iSolver])[0]
            #Hack to make N=200 green
            if nSimulationsPerTree==200:
                handle = ax.plot(timeSteps+offset*jNSimsPerTreeTrial,avgRewards[iSolver,jNSimsPerTreeTrial,:]*successFactor,label=label,ls="-",marker="o",color="green")[0]

            #Make legend only first time through, since labeling by nSimsPerTree
            if iSolver == 0:
                legendHandles.append(handle)


    ax.legend(handles=legendHandles,facecolor="gainsboro",loc="lower right")
    ax.set_xlabel("Simulation Time Step")
    if successRateAdjustmentFlag:
        ax.set_ylabel("Reward * Success Rate")
    else:
        ax.set_ylabel("Reward")
    if systemName is not None:
        ax.set_title(systemName)
    elif noise is None:
        ax.set_title(f"Rewards After Each Time Step (N Simulations per Time Step)\n{nTrialsPerPoint} experiments averaged per data point")
    else:
        ax.set_title(f"Rewards After Each Time Step (N Simulations per Time Step)\n{nTrialsPerPoint} " + fr"experiments averaged per data point. $\sigma_w$ = {noise}")
    ax.set_xticks(onp.arange(0,20,2))

    #Set the color to gray and turn on grid
    ax.set_facecolor(".8")
    plt.grid(True)
    #Normalize y to 1
    #ax.set_ylim(0,1)
    #print(wcts)
    #print(successRates)
    #print(Ns)
    #print(steps)


def setColorCycler(repeatNum,ax):
    """
    Creates a color cycler with CMR color map with specified repeat frequency and assigns to the provided axis
    """
    ax.set_prop_cycle('color',plt.cm.CMRmap(onp.linspace(0,1,repeatNum)))

#Plotting functions

def plotRewardStd(avgRewards,avgSuccessRates,sigmaRewards,nSimulationsPerTrees,nTrialsPerPoint,noise,solverNames,successRateAdjustmentFlag=True,#Main plotting function, so pylint: disable=too-many-statements,too-many-branches
                  systemName=None,dt=1,tSteps=20,cumulativeFlag=False,experimentIndexes=None,fixedYTicksFlag=True):

    """
    Method that loads data to plot and calls plotting functions as needed. Plots reward data showing 1 sigma std.

    Parameters
    ----------
    avgRewards : array, shape(numSolvers,numNSimulationsPerTrees,numExperimentSteps+1)
        The average rewards for each solver and number of simulations per tree, for each time step (including initial time)
    avgSuccessRates : array, shape(numSolvers,numNSimulationsPerTrees,numExperimentSteps+1)
        The average success rate for each solver and number of simulations per tree
    sigmaRewards : array, shape(numSolvers,numNSimulationsPerTrees)
        The 1 sigma bounds for the rewards
    nSimulationsPerTrees : array, shape(numNSimulationsPerTrees)
        The number of simulations performed before returning an action (when not running on time out mode).
        This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
    nTrialsPerPoint : int
        The number of repeated trials per configuration.
    noise : float
        Level of sigma for this collection of experiments
    solverNames : list
        List of the names of each solver that was used
    successRateAdjustmentFlag : bool (default=False)
        If true, rewards are scaled by the success rate
    systemName : str (default=None)
        Name of the system for the plot title, if any
    dt : float (default=1)
        The time between time steps of the experiment
    plotSuccessRatesAndWallClockTimesFlag : boolean (default=False)
        If true plot extra data on success rates and wall clock times
    cumulativeFlag : boolean (default=False)
        If true, plot the cumulative reward
    """

    if solverNames[0] == "SFEAST":
        solverNames[0] = "s-FEAST"
    solverGroupName = solverNames[0] #Will be block of solvers for each nSimulationsPerTrees

    #Check for random or not (this is assumed to only be on s-FEAST solvers)
    if nSimulationsPerTrees[0] == 0:
        solverNames[0] = "Random"
    else:
        solverNames = solverNames[1:]

    #Format baselines as needed
    if "greedy" in solverNames:
        solverNames[solverNames.index("greedy")] = "Greedy"
    if "cbf" in solverNames:
        solverNames[solverNames.index("cbf")] = "CBF"
    if "scp" in solverNames:
        solverNames[solverNames.index("scp")] = "SCP"

    font = {'size'   : 16}
    matplotlib.rc('font', **font)

    #Now plot reward trajectories
    dummyFig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7.5,4),dpi=400)

    #Make x-axis
    timeSteps = onp.arange(0,dt*(len(avgRewards[0,0])),dt)
    #Labels for legend
    legendHandles = []

    #Legend Entries
    lineStyles = ["-",":","--","dashdot"]
    markers = ["o","^","x","*"]
    #Idea, make each baseline a different color
    solverColors = ["black","green","peru","lightskyblue"]
    markersize = 2.5

    for iSolver, solverName in enumerate(solverNames):
        if successRateAdjustmentFlag:
            successFactor = avgSuccessRates[iSolver,0]
        else:
            successFactor = 1
        legendHandles.append(ax.plot(timeSteps[0],avgRewards[0,0,0]*successFactor,label=solverName,ls=lineStyles[iSolver],marker=markers[iSolver],ms=markersize,c=solverColors[iSolver])[0])

    #Add name for group of solvers
    emptyRectHandle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',visible=False,label=solverGroupName)
    legendHandles.append(emptyRectHandle)

    #Set color cycle, adding extra lines to account for matching colors with safety (bit hacky)
    numLines = 8 #len(nSimulationsPerTrees)+len(solverNames)-1
    setColorCycler(numLines,ax)

    offset = .1

    rwbds = onp.zeros((len(solverNames),len(nSimulationsPerTrees),2,tSteps+1))
    rwbds[:,:,0,:] = avgRewards -sigmaRewards #
    rwbds[:,:,1,:] = avgRewards + sigmaRewards
    if experimentIndexes is None:
        experimentIndexes = onp.arange(len(nSimulationsPerTrees))

    #print(avgRewards,avgSuccessRates)
    #Loop through solvers
    for iSolver, solverName in enumerate(solverNames):
        #Loop through experiments
        for jOffset, jNSimsPerTreeExperiment in enumerate(experimentIndexes):
            nSimulationsPerTree = nSimulationsPerTrees[jNSimsPerTreeExperiment]
            #Check if we scale by success
            if successRateAdjustmentFlag:
                successFactor = avgSuccessRates[iSolver,0]
            else:
                successFactor = 1

            #For baselines, want to have their colors not on cycler, and don't want to double add to legend
            if iSolver == 0:
                label = f"N = {nSimulationsPerTree}"
                #Plot rewards, modulated by success rate

                handle = ax.plot(timeSteps+offset*(jOffset-iSolver%2),avgRewards[iSolver,jNSimsPerTreeExperiment,:]*successFactor,ls=lineStyles[iSolver],marker=markers[iSolver],
                                    ms=markersize,label=label)[0]
                #Hack to make N=200 green
                #if nSimulationsPerTree==200:
                #    handle = ax.plot(timeSteps+offset*jNSimsPerTreeTrial,avgRewards[iSolver,jNSimsPerTreeTrial,:]*successFactor,label=label,ls="-",marker="o",color="green")[0]

                #Custom error bar
                ax.vlines(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,0,:]*successFactor,rwbds[iSolver,jNSimsPerTreeExperiment,1,:]*successFactor,
                            ls=lineStyles[iSolver],alpha=1,color=handle.get_color(),zorder=200+jOffset - 10* iSolver)
                #Add markers at end
                ax.scatter(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,1,:]*successFactor,marker=markers[iSolver],s=markersize**2,
                            color=handle.get_color(),zorder=200+jOffset - 10* iSolver)#,alpha=.5)
                ax.scatter(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,0,:]*successFactor,marker=markers[iSolver],s=markersize**2,
                            color=handle.get_color(),zorder=200+jOffset - 10* iSolver)#,alpha=.5)

                #Make legend only first time through, since labeling by nSimsPerTree
                #if iSolver == 0:
                if nSimulationsPerTree != 0:
                    legendHandles.append(handle)
            else:
                ax.plot(timeSteps+offset*(jOffset-iSolver%2),avgRewards[iSolver,jNSimsPerTreeExperiment,:]*successFactor,label=label,ls=lineStyles[iSolver],marker=markers[iSolver],
                            ms=markersize,c=solverColors[iSolver])

                #Custom error bar
                ax.vlines(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,0,:]*successFactor,rwbds[iSolver,jNSimsPerTreeExperiment,1,:]*successFactor,
                            ls=lineStyles[iSolver],color=solverColors[iSolver],alpha=1,zorder=200+jOffset - 10* iSolver)
                #Add markers at end
                ax.scatter(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,0,:]*successFactor,marker=markers[iSolver],s=markersize**2,
                            color=solverColors[iSolver],zorder=200+jOffset - 10* iSolver)#,alpha=.5)
                ax.scatter(timeSteps+offset*(jOffset-iSolver%2),rwbds[iSolver,jNSimsPerTreeExperiment,1,:]*successFactor,marker=markers[iSolver],s=markersize**2,
                            color=solverColors[iSolver],zorder=200+jOffset - 10* iSolver)#,alpha=.5)


    ax.legend(handles=legendHandles,facecolor="gainsboro",loc="lower left",prop={'size': 10}).set_zorder(1000)
    ax.set_xlabel("Simulation Time Step")
    if successRateAdjustmentFlag:
        ax.set_ylabel("Reward * Success Rate")
    else:
        ax.set_ylabel("Reward")
    if cumulativeFlag:
        cumulative = "Cumulative "
    else:
        cumulative = " "
    if systemName is not None:
        ax.set_title(systemName)
    elif noise is None:
        ax.set_title(f"{cumulative}Rewards After Each Time Step (N Simulations per Time Step)\n{nTrialsPerPoint} experiments averaged per data point")
    else:
        ax.set_title(f"{cumulative}Rewards After Each Time Step (N Simulations per Time Step)\n{nTrialsPerPoint} " + fr"experiments averaged per data point. $\sigma_w$ = {noise}")
    ax.set_xticks(onp.arange(0,timeSteps[-1],2))
    #Most plots use this by default anyways, making it explicit for consistency
    if fixedYTicksFlag:
        ax.set_yticks(onp.array([0.,0.2,0.4,0.6,0.8,1.0]))

    #Set the color to gray and turn on grid
    ax.set_facecolor(".8")
    plt.grid(True)

def plotData(experimentName,solverNames,noise,successRateAdjustmentFlag=False,systemName=None,plotSuccessRatesAndWallClockTimesFlag=False):
    """
    Method that loads data to plot and calls plotting functions as needed

    Parameters
    ----------
    experimentName : str
        Name of the experiment (top level saved data directory)
    solverNames : list
        List of the names of each solver that was used
    noise : float
        Level of sigma for this collection of experiments
    successRateAdjustmentFlag : bool (default=False)
        If true, rewards are scaled by the success rate
    systemName : str (default=None)
        Name of the system for the plot title, if any
    """


    nSimulationsPerTrees, nTrialsPerPoint, avgRewards, avgSuccessRates, avgWallClockTimes, dummyAvgSteps, dummySigmaRewards = loadDataSummary(experimentName,solverNames)

    plotAvgsOnly(avgRewards,avgSuccessRates,avgWallClockTimes,nSimulationsPerTrees,nTrialsPerPoint,noise,solverNames,
                 successRateAdjustmentFlag,systemName,plotSuccessRatesAndWallClockTimesFlag=plotSuccessRatesAndWallClockTimesFlag)
    plt.show()


def plotDataRewardStd(experimentName,solverNames,noise,successRateAdjustmentFlag=False,systemName=None,tSteps=20,cumulativeFlag=False,baselineExpName=None,experimentIndexes=None):
    """
    Method that loads data to plot and calls plotting functions as needed. Plots reward data showing 1 sigma std.

    Parameters
    ----------
    experimentName : str
        Name of the experiment (top level saved data directory)
    solverNames : list
        List of the names of each solver that was used
    noise : float
        Level of sigma for this collection of experiments
    successRateAdjustmentFlag : bool (default=False)
        If true, rewards are scaled by the success rate
    systemName : str (default=None)
        Name of the system for the plot title, if any
    """

    nSimulationsPerTrees, nTrialsPerPoint, avgRewards, avgSuccessRates, dummyAvgWallClockTimes, dummyAvgSteps, sigmaRewards = loadDataSummary(experimentName,solverNames,cumulativeFlag,
                                                                                                                                                baselineExpName,tSteps=tSteps )
    plotRewardStd(avgRewards,avgSuccessRates,sigmaRewards,nSimulationsPerTrees,nTrialsPerPoint,noise,solverNames,successRateAdjustmentFlag,systemName,tSteps=tSteps,
            cumulativeFlag=cumulativeFlag,experimentIndexes=experimentIndexes)
    plt.show()


#Alt way to plot only some: add experimentIndexes=None):
def loadDataSummary(experimentName,solverNames,cumulativeFlag=False,baselineExpName=None,altRewardFlag=True,savedDataDirPath=None,tSteps=None): #Allow to be do everything pylint: disable=too-many-statements,too-many-branches
    """
    Method to load saved data that is reused by several functions

    Parameters
    ----------
    experimentName : str
        Name of the experiment (top level saved data directory)
    solverNames : list
        List of the names of each solver that was used
    cumulativeFlag : boolean (default=False)
        If true, plot the cumulative reward
    baselineExpName : str (default=None)
        If provided, loads in additional solver experiments that are all baselines (1 nSim per tree, always 1)
    altRewardFlag : str (default=False)
        If true, looks for alternative reward data instead of the default. Incompatible with cumulative sum currently. Does not check for existence first
    savedDataDirPath : str (default=None)
        Path to saved data. Will try to look in local failurePy installation if none provided, as that is the default

    Returns
    -------
    nSimulationsPerTrees : array, shape(numNSimulationsPerTrees)
        The number of simulations performed before returning an action (when not running on time out mode).
        This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
    nTrialsPerPoint : int
        The number of repeated trials per configuration.
    avgRewards : array, shape(numSolvers,numNSimulationsPerTrees,numExperimentSteps+1)
        The average rewards for each solver and number of simulations per tree, for each time step (including initial time)
    avgSuccessRates : array, shape(numSolvers,numNSimulationsPerTrees)
        The average success rate for each solver and number of simulations per tree
    avgWallClockTimes : array, shape(numSolvers,numNSimulationsPerTrees)
        The average time for each solver and number of simulations per tree
    avgSteps : array, shape(numSolvers,numNSimulationsPerTrees)
        The average steps for each solver and number of simulations per tree
    sigmaRewards : array, shape(numSolvers,numNSimulationsPerTrees)
        The 1 sigma bounds for the rewards
    """

    #Check path provided
    if savedDataDirPath is None:
        savedDataDirPath = loadExperimentFromDefaultSaveData(experimentName)

    #First we need to collect the data
    experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)

    #Check if there is a baseline directory to load from too
    if baselineExpName is not None:
        baselineExperimentDataDirPath = os.path.join(savedDataDirPath,baselineExpName)
        try:
            baselineSolverNames = next(os.walk(baselineExperimentDataDirPath))[1]
        except StopIteration as exception:
            directoryEmptyError = f"Directory {baselineExperimentDataDirPath} does not exist or is empty."
            raise FileNotFoundError(directoryEmptyError) from exception
        #Hack to sort baselines consistently (will format when plotting legend)
        if "greedy" in baselineSolverNames and "cbf" in baselineSolverNames and "scp" in baselineSolverNames:
            baselineSolverNames = ["greedy", "cbf", "scp"]


        numBaselines = len(baselineSolverNames)
        solverNames += baselineSolverNames #Updates solverNames by reference!
    else:
        numBaselines = 0

    #We take for granted that the number of simulations per trees is the same for each solver, as is the number of trials per point. If this is None, haven't found them yet
    nSimulationsPerTrees = None
    nTrialsPerPoint = None

    for iSolver, solverName in enumerate(solverNames):
        #print(iSolver,len(solverNames),numBaselines,solverNames)
        if iSolver >= len(solverNames) - numBaselines:
            solverDirectoryPath = os.path.join(baselineExperimentDataDirPath,solverName)
        else:
            solverDirectoryPath = os.path.join(experimentDataDirPath,solverName)
        if not os.path.exists(solverDirectoryPath):
            raise FileNotFoundError(f"Directory {solverDirectoryPath} not found, check if the correct save directory and experiment are given")

        if nSimulationsPerTrees is None:
            #Get NSimulationsPerTrees using os.walk to read the directory names
            nSimulationsPerTrees = onp.array(next(os.walk(solverDirectoryPath))[1])
            nSimulationsPerTrees = nSimulationsPerTrees.astype(int)
            nSimulationsPerTrees = onp.sort(nSimulationsPerTrees)
            ##Alternate way for plotting only some of the experiments
            #if experimentIndexes is not None:
            #    NSimulationsPerTrees = NSimulationsPerTrees[experimentIndexes]

        for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
            #Initialize average data arrays if we haven't yet
            if nTrialsPerPoint is None:
                #Get  nTrialsPerPoint
                nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                nTrialsPerPoint = len(next(os.walk(nSimPath))[1])


                #Load first data dict
                experimentDataPath = os.path.join(nSimPath,"averageData.dict")
                with open(experimentDataPath,'rb') as experimentDataFile:
                    experimentAverageDataDict = pickle.load(experimentDataFile)

                #Now initialize the average data now that we know the number of experiments
                avgRewards = onp.zeros((len(solverNames),len(nSimulationsPerTrees),len(experimentAverageDataDict["avgRewards"])))
                avgSuccessRates = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                avgWallClockTimes = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                avgSteps = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                sigmaRewards = onp.zeros((len(solverNames),len(nSimulationsPerTrees),len(experimentAverageDataDict["avgRewards"])))
            #Baselines load a little different (SUPER HACKY)
            elif iSolver > 0 and jNSimsPerTreeExperiment == 0:
                nSimPath =  os.path.join(solverDirectoryPath,str(1)) #Always 1 for baselines
                experimentDataPath = os.path.join(nSimPath,"averageData.dict")
                with open(experimentDataPath,'rb') as experimentDataFile:
                    experimentAverageDataDict = pickle.load(experimentDataFile)
            #Fill baselines with nans for other "num sims per tre" SUPER HACKY here, just trying to quickly plot.
            elif iSolver > 0:
                avgSuccessRates[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgWallClockTimes[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgSteps[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgRewards[iSolver,jNSimsPerTreeExperiment] += onp.nan
                sigmaRewards[iSolver,jNSimsPerTreeExperiment] += onp.nan
                continue #Don't load anything
            #Otherwise just load
            else:
                #Load data dict
                nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                experimentDataPath = os.path.join(nSimPath,"averageData.dict")
                with open(experimentDataPath,'rb') as experimentDataFile:
                    experimentAverageDataDict = pickle.load(experimentDataFile)

            #print(experimentAverageDataDict)

            #In either case, get data
            avgSuccessRates[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgSuccessRate"]
            avgWallClockTimes[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgWallClockTime"]
            avgSteps[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgSteps"]
            if cumulativeFlag:
                #If EKF diverged = failure (also need to make sure success set to 0)
                experimentAverageDataDict["cumulativeAvgRewards"][onp.isnan(experimentAverageDataDict["cumulativeAvgRewards"])] = 0

                avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["cumulativeAvgRewards"]
                sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["cumulativeVarRewards"])
            elif altRewardFlag:
                #If EKF diverged = failure (also need to make sure success set to 0)
                experimentAverageDataDict["avgAltRewards"][onp.isnan(experimentAverageDataDict["avgAltRewards"])] = 0

                avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgAltRewards"]
                sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["varAltRewards"])
            else:
                #If EKF diverged = failure (also need to make sure success set to 0)
                experimentAverageDataDict["avgRewards"][onp.isnan(experimentAverageDataDict["avgRewards"])] = 0

                avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgRewards"]
                sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["varRewards"])
    #Allow for only considering some of the data
    if tSteps is not None:
        timeStepsInData = len(avgRewards[0,0]) - 1
        if timeStepsInData > tSteps:
            avgRewards = avgRewards[:,:,:tSteps+1]
            sigmaRewards = sigmaRewards[:,:,:tSteps+1]
        elif timeStepsInData <  tSteps:
            tooManyTimeStepsRequested = f"Saved data only has {timeStepsInData} time steps. {tSteps} requested"
            raise ValueError(tooManyTimeStepsRequested)

    return nSimulationsPerTrees, nTrialsPerPoint, avgRewards, avgSuccessRates, avgWallClockTimes, avgSteps, sigmaRewards

def plotMultipleWallClockTimes(experimentNameList,solverNames,labels=None,savedDataDirPath=None,sigmaFlag=False): #plotting method pylint: disable=too-many-statements,too-many-branches
    """
    Method that loads data to plot and calls plotting functions as needed

    Parameters
    ----------
    experimentName : List of strs
        Names of the experiment (top level saved data directory)
    solverNames : list
        List of the names of each solver that was used
    systemName : str (default=None)
        Name of the system for the plot title, if any
    labels : list (default=None)
        List of strings to label the plots with.
    savedDataDirPath : str (default=None)
        Path to saved data. Will try to look in local failurePy installation if none provided, as that is the default
    sigmaFlag : boolean (default=False)
        When true, 1 std shown for the different times measured
    """

    #Check path provided
    if savedDataDirPath is None:
        savedDataDirPath = loadExperimentFromDefaultSaveData(experimentNameList[0])

    experimentWallClockTimesList = []
    experimentSigmaWallClockTimesList = []

    for experimentName in experimentNameList:
        #First we need to collect the data
        experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)

        #We take for granted that the number of simulations per trees is the same for each solver, as is the number of trials per point. If this is None, haven't found them yet
        nSimulationsPerTrees = None
        nTrialsPerPoint = None

        for jSolver, solverName in enumerate(solverNames):
            solverDirectoryPath = os.path.join(experimentDataDirPath,solverName)
            if not os.path.exists(solverDirectoryPath):
                raise FileNotFoundError("Directory not found, check if the correct save directory and experiment are given")

            if nSimulationsPerTrees is None:
                #Get nSimulationsPerTrees using os.walk to read the directory names
                nSimulationsPerTrees = onp.array(next(os.walk(solverDirectoryPath))[1])
                nSimulationsPerTrees = nSimulationsPerTrees.astype(int)
                nSimulationsPerTrees = onp.sort(nSimulationsPerTrees)

            for jNSimsPerTreeTrial, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
                #Initialize average data arrays if we haven't yet
                if nTrialsPerPoint is None:
                    #Get  nTrialsPerPoint
                    nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                    nTrialsPerPoint = len(next(os.walk(solverDirectoryPath))[1])

                    #Load first data dict
                    experimentDataPath = os.path.join(nSimPath,"averageData.dict")
                    with open(experimentDataPath,'rb') as experimentDataFile:
                        experimentAverageDataDict = pickle.load(experimentDataFile)

                    #Now initialize the average data now that we know the number of experiments
                    #Only want clock times
                    avgWallClockTimes = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                    sigmaWallClockTimes = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))

                #Otherwise just load
                else:
                    #Load data dict
                    nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                    experimentDataPath = os.path.join(nSimPath,"averageData.dict")
                    with open(experimentDataPath,'rb') as experimentDataFile:
                        experimentAverageDataDict = pickle.load(experimentDataFile)

                #print(experimentAverageDataDict)

                #In either case, get data
                avgWallClockTime = experimentAverageDataDict["avgWallClockTime"]
                sigmaWallClockTime = onp.sqrt(experimentAverageDataDict["varWallClockTime"])
                if not isinstance(avgWallClockTime, numbers.Number):
                    avgWallClockTime = avgWallClockTime[1] #It's from the old data (backwards compatibility with pervious versions of failurePy)
                avgWallClockTimes[jSolver,jNSimsPerTreeTrial] = avgWallClockTime
                sigmaWallClockTimes[jSolver,jNSimsPerTreeTrial] = sigmaWallClockTime
        #Append
        experimentWallClockTimesList.append(avgWallClockTimes)
        experimentSigmaWallClockTimesList.append(sigmaWallClockTimes)

    #Now let's make the plot

    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    dummyFig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,8))

    #Make x-axis
    #Set random to 1 simulation, so it plots on a log scale well
    semiLogNSimulationsPerTrees = onp.copy(nSimulationsPerTrees)

    #Labels for legend
    legendHandles = []

    #Legend Entries
    #lineStyles = ["-",":"]
    markers = ["o","^","x","*"]

    #Get labels
    if labels is None:
        labels = []
        for iExperiment in len(experimentNameList):
            labels.append(None)

    #Loop through solvers and experiments and plot
    #print(solverNames)
    for jSolver, solverName in enumerate(solverNames):
        for iExperiment,experimentName in enumerate(experimentNameList):
            label = labels[iExperiment]
            avgWallClockTimes = experimentWallClockTimesList[iExperiment]
            sigmaWallClockTimes = experimentSigmaWallClockTimesList[iExperiment]
            handle = ax.plot(semiLogNSimulationsPerTrees,avgWallClockTimes[jSolver],label=label,marker=markers[jSolver])[0]
            #Plot error bars
            if sigmaFlag:
                sigma = sigmaWallClockTimes[jSolver]
                #print("sigmas",sigma)
                ax.scatter(semiLogNSimulationsPerTrees,avgWallClockTimes[jSolver]+sigma,label=labels[iExperiment],marker=markers[jSolver],color=handle.get_color(),alpha=.8)
                ax.scatter(semiLogNSimulationsPerTrees,avgWallClockTimes[jSolver]-sigma,label=labels[iExperiment],marker=markers[jSolver],color=handle.get_color(),alpha=.8)
                ax.vlines(semiLogNSimulationsPerTrees,avgWallClockTimes[jSolver]-sigma,avgWallClockTimes[jSolver]+sigma,ls="-",color=handle.get_color(),alpha=.8)
            if jSolver == 0:
                legendHandles.append(handle)

    #legendHandles.append(ax.hlines(1,semiLogNSimulationsPerTrees[0],semiLogNSimulationsPerTrees[-1],ls="--",label="Real Time"))

    ax.legend(handles=legendHandles)
    ax.set_xlabel("Number of Simulations per Tree")

    ax.set_ylabel("Wall Clock Time/Iteration (s)")

    ax.set_title("Average wall clock time on Jetson Orin Dev Kit\n20 experiments averaged per data point")

    #ax.set_xticks(onp.arange(0,20,2))

    #Set the color to gray and turn on grid
    ax.set_facecolor(".8")
    plt.grid(True)

    #print(experimentWallClockTimesList)

    plt.show()

def plotTrajectories(experimentName,solverName,figureSavePath="trajectoryRender.pdf",savedDataDirPath=None):
    """
    Visualize trajectories on x-y space to show qualitative differences of high vs. low planning

    Parameters
    ----------
    experimentName : str
        Name of the experiment (top level saved data directory)
    solverName : string
        Names of the solver that was used (for now only one solver at a time can be plotted)
    noise : float
        Level of sigma for this collection of experiments
    systemName : str (default=None)
        Name of the system for the plot title, if any
    savedDataDirPath : str (default=None)
        Path to saved data. Will try to look in local failurePy installation if none provided, as that is the default
    """

    #Check path provided
    if savedDataDirPath is None:
        savedDataDirPath = loadExperimentFromDefaultSaveData(experimentName)

    #Need to load raw data, not summary, so have to go deeper.

    #First we need to collect the data
    experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)
    solverDirectoryPath = os.path.join(experimentDataDirPath,solverName)

    nTrialsPerPoint = None


    if not os.path.exists(solverDirectoryPath):
        raise FileNotFoundError("Directory not found, check if the correct save directory and experiment are given")

    #We take for granted that the number of simulations per trees is the same for each solver, as is the number of trials per point.
    #Get nSimulationsPerTrees using os.walk to read the directory names
    nSimulationsPerTrees = onp.array(next(os.walk(solverDirectoryPath))[1])
    nSimulationsPerTrees = nSimulationsPerTrees.astype(int)
    nSimulationsPerTrees = onp.sort(nSimulationsPerTrees)

    for jExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
        nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))

        #Initialize average data arrays if we haven't yet
        if nTrialsPerPoint is None:
            #Get  nTrialsPerPoint and
            #Get trial numbers using os.walk to read the directory names
            trialNums = onp.array(next(os.walk(nSimPath))[1])
            trialNums = trialNums.astype(int)
            trialNums = onp.sort(trialNums)

            nTrialsPerPoint = len(trialNums)

            #Load first data summary dict to get number of time steps (tf)
            experimentDataPath = os.path.join(nSimPath,"averageData.dict")
            with open(experimentDataPath,'rb') as experimentDataFile:
                experimentAverageDataDict = pickle.load(experimentDataFile)
            #tf : int (default=20)
            #    Final time of each experiment, if not terminated before then
            tf = len(experimentAverageDataDict["avgRewards"])

            #Create state arrays
            states = onp.zeros((len(nSimulationsPerTrees),nTrialsPerPoint,tf,2))

        #Now that we are initialized, get state list for each trial
        for kTrial,trialNum in enumerate(trialNums):
            kTrialPath = os.path.join(os.path.join(nSimPath,str(trialNum)),"trialData.dict")
            #Load data dict
            with open(kTrialPath,'rb') as trialDataFile:
                trialDataDict = pickle.load(trialDataFile)
            physicalStateList = trialDataDict["physicalStateList"]
            #Loop over states and grab just the positions (0th and 2nd state, as x, vx, y, vy)
            #Should be tf of these
            lState = 0
            for lState, state in enumerate(physicalStateList):
                states[jExperiment,kTrial,lState,:] = state[0::2] #pylint: disable=possibly-used-before-assignment
            #Make sure we don't go back to origin if done
            for mState in range(tf-lState):
                states[jExperiment,kTrial,mState+lState,:] = states[jExperiment,kTrial,lState,:]

            ##Loop through#Make sure we don't go back to origin if done
            #for lState, state in enumerate(states):
            #    if lState != 0 and onp.all(state == 0):
            #        states[jExperiment,kTrial,lState] = states[jExperiment,kTrial,lState-1]

    #After loading, we can plot each experiment (reusing code here, doesn't seem to translate well to an import)
    plottingBounds  = onp.array([
        [-150, 150],
        [-150, 150],
    ])

    for iExperiment,nSimulationsPerTree in enumerate(nSimulationsPerTrees):

        dummyFig, ax = plt.subplots(figsize=(15,15) ,dpi=1000)

        #Make a collection of line segments
        colors = -1* onp.arange(-1,0,1/(tf-1))

        for kSegment in range(tf-1):
            lineSegments = []
            for jTrial in range(len(trialNums)):

                lineSegments.append([[states[iExperiment,jTrial,kSegment,0],states[iExperiment,jTrial,kSegment,1]],[states[iExperiment,jTrial,kSegment+1,0],states[iExperiment,jTrial,kSegment+1,1]]])

            stateTrajectoryLineCollection = LineCollection(lineSegments,
                linewidth=1.5, colors=[colors[kSegment],1-colors[kSegment],0], alpha=0.9, zorder=kSegment)
            ax.add_collection(stateTrajectoryLineCollection)

        ax.set_xlim(plottingBounds[0,:])
        ax.set_ylim(plottingBounds[1,:])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Trajectories of SFEAST-{nSimulationsPerTree} Experiments")
        ax.set_facecolor("#481B6D")

    pdfOutput = PdfPages(os.path.join(os.getcwd(),figureSavePath))
    for iFig in plt.get_fignums():
        pdfOutput.savefig(plt.figure(iFig),dpi=1000)
        #plt.close(plt.figure(i))
    pdfOutput.close()

    plt.show()




def loadExperimentFromDefaultSaveData(experimentName):
    """
    Module that attempts to load the experiment from the default failurePy/SavedData directory
    """
    #Not backwards compatible, but this is only called if the savedDataDirectory is not provided
    if sys.version_info < (3,9):
        raise ValueError("Loading saved data from the default SavedData directory is not supported for python versions < 3.9. Provide savedDataDirPath to the method call instead")
    from failurePy.load.packageLoader import getPackageSubDirectoryPath  # pylint: disable=import-outside-toplevel #Guarded import, only if needed
    defaultSavedDataDirPath = getPackageSubDirectoryPath(failurePy,"SavedData")
    if not os.path.exists(os.path.join(defaultSavedDataDirPath,experimentName)):
        pathDoesNotExistInDefaultDirectory = f"The specified experiment {experimentName} does not exist in the default SavedData directory ({defaultSavedDataDirPath}). " +\
                                            "Please provide savedDataDirPath containing this experiment"
        raise FileNotFoundError(pathDoesNotExistInDefaultDirectory)

    return defaultSavedDataDirPath



#Plotting functions
def setUpPlottingCommonFeatures(solverNames,numLines,tFinal,nrows=1,ncols=1,solverGroupName=None):#,logX=False):
    """
    Creates common elements between plots
    """
    if ncols == 1:
        fontSize = 16
    elif ncols == 2:
        fontSize = 16
    else:
        fontSize = 12
    font = {'size'   : fontSize}
    matplotlib.rc('font', **font)

    if ncols == 1:
        figsizeX = 15/2
    else:
        figsizeX = 15
    if nrows == 1 and ncols == 2:
        figsizeY = 4
    else:
        figsizeY = nrows*4

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(figsizeX,figsizeY),squeeze=False,dpi=400) #So we always iterate, user can squeeze later (or we could)

    #Labels for legend
    legendHandles = []

    #Label Solvers
    #Legend Entries
    lineStyles = ["-",":","--","dashdot"]
    markers = ["o","^","x","*"]
    #Idea, make each baseline a different color
    solverColors = ["black","green","peru","lightskyblue"]

    for iRow in range(nrows):
        for jCol in range(ncols):
            #Set color cycle
            setColorCycler(numLines,axs[iRow,jCol])
            axs[iRow,jCol].set_xlim(-0.5,tFinal)
            axs[iRow,jCol].set_facecolor(".8")
            axs[iRow,jCol].grid(True)
            #if logX:
            #    axs[iRow,jCol].log


    for iSolver, solverName in enumerate(solverNames):
        #Going to plot off screen (using fact we never have negative time)
        #Place legend only in bottom right plot
        legendHandles.append(axs[-1,-1].plot(-1,0,label=solverName,ls=lineStyles[iSolver],marker=markers[iSolver],c=solverColors[iSolver])[0])

    #Create group name for our method if provided (this will make a black entry that can be used to group the solvers below it)
    if solverGroupName is not None:
        emptyRectHandle = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',visible=False,label=solverGroupName)
        legendHandles.append(emptyRectHandle)

    return fig, axs, legendHandles, iSolver+1, lineStyles, markers, solverColors #Will error out before this if it hasn't looped pylint: disable=undefined-loop-variable

def makePlotLegendLowerLeft(ax,legendHandles,fontSize=10):
    "Makes lower right legend"
    ax.legend(handles=legendHandles,facecolor="gainsboro",loc="lower left",prop={'size': fontSize})


def plotAvgSafetyOverTrialsByEvalMethod(timeSteps,empSafetyVals,trueSafetyVals,nInfSafetyVals,solverNames,nSimulationsPerTreesList,figTitleExpName=None,
                                        beliefAlphaSafetyValues=None,experimentIndexes=None):
    """
    Creates comparison plots for each N for the different safety eval methods (averages)
    """
    numLines = 4

    numSubPlots = len(nSimulationsPerTreesList)
    if numSubPlots > 6:
        ncols = 3
        nrows = int(onp.ceil( numSubPlots /3))
    elif numSubPlots > 1:
        ncols = 2
        nrows = int(onp.ceil( numSubPlots /2))
    else:
        ncols = 1
        nrows = 1

    fig,axs,legendHandles, numSolvers, lineStyles, markers, dummySolverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=nrows,ncols=ncols)
    iRow = 0
    kCol = 0
    if experimentIndexes is None:
        experimentIndexes = onp.arange(len(nSimulationsPerTreesList))
    for jNSimsPerTreeExperiment in experimentIndexes:
        nSimulationsPerTree = nSimulationsPerTreesList[jNSimsPerTreeExperiment]
        for mSolver in range(numSolvers):
            #print(onp.shape(lineStyles),onp.shape(markers),onp.shape(empSafetyVals),onp.shape(axs))
            empSafety, =  axs[iRow,kCol].plot(timeSteps,onp.average(empSafetyVals[mSolver,jNSimsPerTreeExperiment,:,:],axis=0),
                                                            ls=lineStyles[mSolver],marker=markers[mSolver],label="Sample-Based Evaluation")
            nInfSafety, = axs[iRow,kCol].plot(timeSteps,onp.average(nInfSafetyVals[mSolver,jNSimsPerTreeExperiment,:,:],axis=0),
                                                            ls=lineStyles[mSolver],marker=markers[mSolver],label=r"~M=$\infty$ Evaluation")
            if beliefAlphaSafetyValues is not None:
                alphaBeliefSafety, = axs[iRow,kCol].plot(timeSteps,onp.average(beliefAlphaSafetyValues[0,jNSimsPerTreeExperiment,:,:],axis=0),
                                                            ls=lineStyles[mSolver],marker=markers[mSolver],label="Belief Direct Sampling, M=2000")
            trueSafety, = axs[iRow,kCol].plot(timeSteps,onp.average(trueSafetyVals[mSolver,jNSimsPerTreeExperiment,:,:],axis=0),
                                                            ls=lineStyles[mSolver],marker=markers[mSolver],label="True Safety",color="blue")
            if jNSimsPerTreeExperiment == 0 and mSolver == 0: #Only plot legend once
                legendHandles.append(empSafety)
                legendHandles.append(nInfSafety)
                if beliefAlphaSafetyValues is not None:
                    legendHandles.append(alphaBeliefSafety)
                legendHandles.append(trueSafety)
        axs[iRow,kCol].set_title(f"N={nSimulationsPerTree}")
        axs[iRow,kCol].set_ylim(0,1.05)
        axs[iRow,kCol].hlines(.95,0,15,linestyle=((0, (5, 10))))
        kCol += 1
        if kCol >= ncols:
            kCol = 0
            iRow += 1
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Different Safety Evaluations for Various Planning Levels for {figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1)
        fig.subplots_adjust(top=.95)
    plt.show()

def plotTrueAndEstimatedSafetyOursVsBaselines(timeSteps,ourEmpSafetyVals,ourTrueSafetyVals,baselineEmpSafetyVals,baselineTrueSafetyVals,
                                              solverNamesOursFirst,nSimulationsPerTreesList,figTitleExpName=None,threshold=.95,experimentIndexes=None):
    """
    Creates plots with baselines!
    """
    numLinesSFEAST = len(nSimulationsPerTreesList)
    if experimentIndexes is None:
        experimentIndexes = onp.arange(numLinesSFEAST)
    else:
        numLinesSFEAST = len(experimentIndexes)


    numBaselines = len(baselineEmpSafetyVals)
    numLines = numLinesSFEAST + numBaselines
    fig,axs,legendHandles, dummyNumSolvers, lineStyles, markers, solverColors = setUpPlottingCommonFeatures(solverNamesOursFirst,numLines,timeSteps[-1],nrows=1,ncols=2)
    for jNSimsPerTreeExperiment in experimentIndexes:
        nSimulationsPerTree = nSimulationsPerTreesList[jNSimsPerTreeExperiment]
        #plot true safety
        axs[0,0].plot(timeSteps,onp.average(ourTrueSafetyVals[0,jNSimsPerTreeExperiment,:,:],axis=0),ls=lineStyles[0],marker=markers[0])
        #Plot and label estimated safety
        handle, = axs[0,1].plot(timeSteps,onp.average(ourEmpSafetyVals[0,jNSimsPerTreeExperiment,:,:],axis=0),ls=lineStyles[0],marker=markers[0],label=f"N={nSimulationsPerTree}")
        #print(numSolvers)
        #Baselines
        legendHandles.append(handle)
    for mSolver in range(numBaselines):
        #plot true safety
        axs[0,0].plot(timeSteps,onp.average(baselineTrueSafetyVals[mSolver,0,:,:],axis=0),ls=lineStyles[mSolver+1],marker=markers[mSolver+1],c=solverColors[mSolver+1])
        #Plot and label estimated safety
        axs[0,1].plot(timeSteps,onp.average(baselineEmpSafetyVals[mSolver,0,:,:],axis=0),ls=lineStyles[mSolver+1],marker=markers[mSolver+1],c=solverColors[mSolver+1])

    axs[0,0].hlines(threshold,0,15,linestyle=((0, (5, 10))))
    axs[0,1].hlines(threshold,0,15,linestyle=((0, (5, 10))))

    axs[0,0].set_title("Average True Safety")
    axs[0,1].set_title("Average Estimated Safety")
    axs[0,0].set_ylabel("Fraction of Safe Trials")
    axs[0,0].set_xlabel("Simulation Time Step")
    axs[0,1].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Average Safety Across Different Planning Levels\n{figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1.1)
        fig.subplots_adjust(top=.85)
    plt.show()

def plotTrueSafetyOursVsBaselines(timeSteps,ourTrueSafetyVals,baselineTrueSafetyVals,
                                              solverNamesOursFirst,nSimulationsPerTreesList,figTitleExpName=None,threshold=.95,experimentIndexes=None):
    """
    Creates plots with baselines!
    """

    #Get our method's name
    ourName = solverNamesOursFirst[0]

    numLinesSFEAST = len(nSimulationsPerTreesList)
    if experimentIndexes is None:
        experimentIndexes = onp.arange(numLinesSFEAST)
    else:
        numLinesSFEAST = len(experimentIndexes)


    numBaselines = len(baselineTrueSafetyVals)

    numLines = numLinesSFEAST + numBaselines
    #Make solver names for legend
    solverNames = solverNamesOursFirst
    #Check if we have a random (N=0) experiment as if so we need to add random to our legend
    if nSimulationsPerTreesList[experimentIndexes[0]] == 0:
        solverNames[0] = "Random"
    else:
        solverNames = solverNames[1:]

    fig,axs,legendHandles, dummyNumSolvers, lineStyles, markers, solverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=1,ncols=1,solverGroupName=ourName)
    #print(legendHandles)
    for jNSimsPerTreeExperiment in experimentIndexes:
        nSimulationsPerTree = nSimulationsPerTreesList[jNSimsPerTreeExperiment]
        #plot true safety
        handle, = axs[0,0].plot(timeSteps,onp.average(ourTrueSafetyVals[0,jNSimsPerTreeExperiment,:,:],axis=0),ls=lineStyles[0],marker=markers[0],label=f"N={nSimulationsPerTree}")
        #No error bars because this is a percentage of the trials safe, so an error bar doesn't make sense

        #Only add to legend of s-FEAST, not random
        if nSimulationsPerTree != 0:
            legendHandles.append(handle)
    for mSolver in range(numBaselines):
        #plot true safety
        axs[0,0].plot(timeSteps,onp.average(baselineTrueSafetyVals[mSolver,0,:,:],axis=0),ls=lineStyles[mSolver+1],marker=markers[mSolver+1],c=solverColors[mSolver+1])


    axs[0,0].hlines(threshold,0,15,linestyle=((0, (5, 10))))

    axs[0,0].set_ylabel("Fraction of Safe Trials")
    axs[0,0].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Average Safety Across Different Planning Levels\n{figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1.1)
        fig.subplots_adjust(top=.85)
    else: #This is for paper where we aren't doing side by side
        axs[0,0].set_title("Average Safety for Each Policy")

    plt.show()

def plotTrueAndEstimatedSafetyPerN(timeSteps,empSafetyVals,trueSafetyVals,solverNames,nSimulationsPerTreesList,figTitleExpName=None,threshold=.95,experimentIndexes=None):
    """
    Creates comparison plots across each N for the different safety eval methods (averages)
    """

    numLines = len(nSimulationsPerTreesList)
    if experimentIndexes is None:
        experimentIndexes = onp.arange(numLines)
    else:
        numLines = len(experimentIndexes)
    fig,axs,legendHandles, numSolvers, lineStyles, markers, dummySolverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=1,ncols=2)

    for jNSimsPerTreeExperiment in experimentIndexes:
        nSimulationsPerTree = nSimulationsPerTreesList[jNSimsPerTreeExperiment]
    #for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTreesList):
        #print(numSolvers)
        for mSolver in range(numSolvers):
            #plot true safety
            axs[0,0].plot(timeSteps,onp.average(trueSafetyVals[mSolver,jNSimsPerTreeExperiment,:,:],axis=0),ls=lineStyles[mSolver],marker=markers[mSolver])
            #Plot and label estimated safety
            handle, = axs[0,1].plot(timeSteps,onp.average(empSafetyVals[mSolver,jNSimsPerTreeExperiment,:,:],axis=0),ls=lineStyles[mSolver],marker=markers[mSolver],label=f"N={nSimulationsPerTree}")
            #Back out false alive statuses. We know that if
        legendHandles.append(handle)
    axs[0,0].hlines(threshold,0,15,linestyle=((0, (5, 10))))
    axs[0,1].hlines(threshold,0,15,linestyle=((0, (5, 10))))

    axs[0,0].set_title("Avg True Safety")
    axs[0,1].set_title("Avg Estimated Safety")
    axs[0,0].set_ylabel("Fraction of Safe Trials")
    axs[0,0].set_xlabel("Simulation Time Step")
    axs[0,1].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Average Safety Across Different Planning Levels for {figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1.1)
        fig.subplots_adjust(top=.85)
    plt.show()

def plotEstimatedFalsePositiveAndNegativeRates(timeSteps,estimatedFalsePositives,estimatedFalseNegatives,solverNames,nSimulationsPerTreesList,figTitleExpName=None):
    """
    Plot false positive and negatives. False positives should be very close to zero (at least below alpha=5%)
    """
    numLines = len(nSimulationsPerTreesList)
    fig,axs,legendHandles, dummyNumSolvers, dummyLineStyles, dummyMarkers, dummySolverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=2,ncols=2)
    for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTreesList):
        #plot false positives
        axs[0,0].plot(timeSteps,onp.sum(estimatedFalsePositives[0,jNSimsPerTreeExperiment,:,:,0],axis=0))
        #plot false negatives
        axs[0,1].plot(timeSteps,onp.sum(estimatedFalseNegatives[0,jNSimsPerTreeExperiment,:,:,0],axis=0))

        #plot false positives
        axs[1,0].plot(timeSteps,onp.sum(estimatedFalsePositives[0,jNSimsPerTreeExperiment,:,:,1],axis=0))
        #plot false negatives
        handle, = axs[1,1].plot(timeSteps,onp.sum(estimatedFalseNegatives[0,jNSimsPerTreeExperiment,:,:,1],axis=0),label=f"N={nSimulationsPerTree}")
        legendHandles.append(handle)

    axs[0,0].set_title("False Positive Estimated Safety Rates")
    axs[0,1].set_title("False Negative Estimated Safety Rates")
    axs[1,0].set_title("False Positive Alive Status")
    axs[1,1].set_title("False Negative Alive Status")
    axs[0,0].set_ylabel("Total False Positive/Negatives")
    axs[1,0].set_ylabel("Total False Positive/Negatives")
    axs[1,0].set_xlabel("Simulation Time Step")
    axs[1,1].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Estimated Safety Error Rates for {figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1)
        fig.subplots_adjust(top=.85)
    plt.show()


def plotBeliefAlphaFalsePositiveAndNegativeRates(timeSteps,beliefAlphaFalsePositives,beliefAlphaFalseNegatives,solverNames,nSimulationsPerTreesList,figTitleExpName=None):
    """
    Plot false positive and negatives. False positives should be very close to zero (at least below alpha=5%)
    """
    numLines = len(nSimulationsPerTreesList)
    fig,axs,legendHandles, dummyNumSolvers, dummyLineStyles, dummyMarkers, dummySolverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=2,ncols=2)
    for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTreesList):
        #plot false positives
        axs[0,0].plot(timeSteps,onp.sum(beliefAlphaFalsePositives[0,jNSimsPerTreeExperiment,:,:,0],axis=0))
        #plot false negatives
        axs[0,1].plot(timeSteps,onp.sum(beliefAlphaFalseNegatives[0,jNSimsPerTreeExperiment,:,:,0],axis=0))

        #plot false positives
        axs[1,0].plot(timeSteps,onp.sum(beliefAlphaFalsePositives[0,jNSimsPerTreeExperiment,:,:,1],axis=0))
        #plot false negatives
        handle, = axs[1,1].plot(timeSteps,onp.sum(beliefAlphaFalseNegatives[0,jNSimsPerTreeExperiment,:,:,1],axis=0),label=f"N={nSimulationsPerTree}")
        legendHandles.append(handle)

    axs[0,0].set_title("False Positive Belief Alpha Safety Rates")
    axs[0,1].set_title("False Negative Belief Alpha Safety Rates")
    axs[1,0].set_title("False Positive Alive Status")
    axs[1,1].set_title("False Negative Alive Status")
    axs[0,0].set_ylabel("Total False Positive/Negatives")
    axs[1,0].set_ylabel("Total False Positive/Negatives")
    axs[1,0].set_xlabel("Simulation Time Step")
    axs[1,1].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Belief Alpha Safety Error Rates for {figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1)
        fig.subplots_adjust(top=.85)
    plt.show()

def plotTrueAndBelievedLastCollisionFree(nSimulationsPerTrees,avgTrueLastCollisionFreeTSteps,avgBelievedLastCollisionFreeTSteps,figTitleExpName=None):
    """
    Plots the true and believed last time before collision (on average) as a function of nSimulationsPerTrees
    """

    dummyFig,ax = plt.subplots(1,1,figsize=(7.5,5))

    #Plot true and believed. Slight off set to avoid singularity at n = 0
    if nSimulationsPerTrees[0] == 0:
        nSimulationsPerTrees[0] = 1
    handle1, = ax.semilogx(nSimulationsPerTrees,avgTrueLastCollisionFreeTSteps[0],color="k",label="Ground Truth")
    handle2, = ax.semilogx(nSimulationsPerTrees,avgBelievedLastCollisionFreeTSteps[0],color="b",label="Believed")
    if figTitleExpName is not None:
        ax.set_title(f"Average Number of Collision Free Time Steps,\n {figTitleExpName}")
    else:
        ax.set_title("Average Number of Collision Free Time Steps")
    ax.set_ylabel("Last collision Free Time Step")
    ax.set_xlabel("Number of Simulations Per Tree")
    #ax.set_xscale("log")
    ax.set_xlim(.5,nSimulationsPerTrees[-1]+30)
    ax.set_facecolor(".8")
    ax.grid(True)
    makePlotLegendLowerLeft(ax,[handle1,handle2])

    plt.show()

def plotTrueAndEstimatedSafetyPerM(timeSteps,empSafetyVals,trueSafetyVals,solverNames,mSafetySamplesPerNodes,nSimulationsPerTree,figTitleExpName=None,experimentIndexes=None):
    """
    Creates comparison plots for fixed N and varying safety sample M in the tree for the different safety eval methods (averages)
    """
    numLines = len(mSafetySamplesPerNodes)
    if experimentIndexes is None:
        experimentIndexes = onp.arange(numLines)
    else:
        numLines = len(experimentIndexes)
    fig,axs,legendHandles, dummyNumSolvers, dummyLineStyles, dummyMarkers, dummySolverColors = setUpPlottingCommonFeatures(solverNames,numLines,timeSteps[-1],nrows=1,ncols=2)
    for jMSampleExperiment in experimentIndexes:
        mSafetySamples = mSafetySamplesPerNodes[jMSampleExperiment]
        #plot true safety
        axs[0,0].plot(timeSteps,onp.average(trueSafetyVals[0,jMSampleExperiment,:,:],axis=0))
        #Plot and label estimated safety
        handle, = axs[0,1].plot(timeSteps,onp.average(empSafetyVals[0,jMSampleExperiment,:,:],axis=0),label=f"M={mSafetySamples}")
        #Back out false alive statuses. We know that if
        legendHandles.append(handle)

    axs[0,0].set_title("Avg True Safety")
    axs[0,1].set_title("Avg Estimated Safety")
    axs[0,0].set_ylabel("Fraction of Safe Trials")
    axs[0,0].set_xlabel("Simulation Time Step")
    axs[0,1].set_xlabel("Simulation Time Step")
    makePlotLegendLowerLeft(axs[-1,-1],legendHandles)
    if figTitleExpName is not None:
        figureTitle = fig.suptitle(f"Average Safety Across Different Planning Levels for N={nSimulationsPerTree}\n{figTitleExpName}")
        #Adjust spacing
        figureTitle.set_y(1)
        fig.subplots_adjust(top=.85)
    plt.show()
