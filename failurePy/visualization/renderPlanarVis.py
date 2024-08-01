"""
Old render_2d_vis.py file from v1, converted to match v2 conventions for clarity.

Main method moved to wrapper

Some other minor changes to improve readability/remove unused code
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, ConnectionStyle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap
import cv2


plt.rcParams.update({'font.size': 10})

#ax is widely used with matplotlib for axis, so will make an exception. Similarly lots of plotting lines so we have an exception for number of statements.
def drawSpacecraft(physicalState, action, beliefTuple, possibleFailures, rootNode, ax, plottingBounds, legendFlag=False,futureAction=None,rotationFlag=False): # pylint: disable=invalid-name,too-many-statements
    """
    Function that draws a rendition of a planar spacecraft. Now renders in a "mini-map" like section of the plot,
    to better show scale.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state
    action : array, shape(numAct)
        Action taken to get to this state.
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures,numAct+numSen)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state.
                OVERLOAD: Reused to hold a ground truth flag as GroundTruth for plotting the true failure
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are sorted over this
    rootNode : BeliefNode
        Root node of the tree that is constructed to plan the next action
    ax : matplotlib axis
        Axis to draw s/c in
    plottingBounds : array, shape(2,2)
        Bounds of the axis
    scale : float (default=20)
        How much the s/c should be scaled up
    legendFlag : boolean (default=False)
        Whether or not a legend is shown
    futureAction : array, shape(numAct) (default=None)
        If provided, the action that will be taken next
    rotationFlag : boolean (default=False)
        Whether or not this is a 3DOF spacecraft
    """
    # check whether we are plotting ground truth or experiment
    groundTruthFlag = bool(isinstance(beliefTuple[1],str) and beliefTuple[1] == "groundTruth")

    numAct, numSen = getNumActNumSen(rotationFlag,action)

    edgecolor = "black"
    linewidth = 0.2
    treeColor = "xkcd:dark gray" #"white"#"xkcd:dark gray"
    failureThreshold = 0.22 #This is the maximum failure chance that a component can have before it starts to show as red. (ie, 22% failure chance shows as white)
    numMeshForPositionGaussian = 200 #How many points in each axis will be sampled when making the position Gaussian
    positionGaussianExtent = 5 # what is interesting gaussian region?

    # make special colormap, using our threshold to set the linear scaling on the colors. Currently 0-.78 -> red - white. .78-1 -> white to green
    colorMapColors = ["red","white","green"]
    anchorValues = [0,1-failureThreshold,1.]
    colorMapAnchorPointList = list(zip(anchorValues,colorMapColors))
    failureBeliefColorMap = LinearSegmentedColormap.from_list('rg',colorMapAnchorPointList, N=256)

    # get failure probabilities:
    failureProbabilities = computeComponentFailureProbabilities(beliefTuple, possibleFailures, numAct, numSen)

    #Now instead going to always transform to same spot, except for small icon to indicate spot (just small version of s/c?)


    #Translate to upper right corner, use plotting bounds to set this to top third and right third
    #Set scale to 1/4 of axis size (worst case)
    xSize = plottingBounds[0,1]-plottingBounds[0,0]
    ySize = plottingBounds[1,1]-plottingBounds[1,0]
    miniMapBodyScale = np.min([xSize/5,ySize/5])
    #Draw box and background as 1/3 of axis size. z order or artist order?
    miniMapBackgroundSize = miniMapBodyScale*5/3
    miniMapXUpperRight = xSize/6 *5 - miniMapBackgroundSize/2 + plottingBounds[0,0]#Referenced off the bottom left corner!
    miniMapYUpperRight = ySize/6 *5 - miniMapBackgroundSize/2 + plottingBounds[1,0]

    #Disable belief and safety only
    miniMapBackground = Rectangle((miniMapXUpperRight,miniMapYUpperRight), miniMapBackgroundSize, miniMapBackgroundSize,
        facecolor="gainsboro", alpha=1, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_artist(miniMapBackground)
#
    miniMapBodyXUpperRight = xSize/6 *5 - miniMapBodyScale/2 + plottingBounds[0,0]#Referenced off the bottom left corner!
    miniMapBodyYUpperRight = ySize/6 *5 - miniMapBodyScale/2 + plottingBounds[1,0]

    #Set rotation to 0 if not 3DOF
    if rotationFlag:
        angle = physicalState[4]
    else:
        angle = 0

    #Draw minimap (disable for belief and safety only)
    _drawAndScaleSpacecraft(miniMapBodyScale,miniMapBodyXUpperRight, miniMapBodyYUpperRight,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,
                           failureProbabilities,action,futureAction,angle,ax)

    #Draw actual size s/c, 1 meter size
    bodyXUpperRight = physicalState[0]-1/2
    bodyYUpperRight = physicalState[2]-1/2
    bodyScale = 1
    #Disable for belief and safety only and hardware vis of mini map
    _drawAndScaleSpacecraft(bodyScale,bodyXUpperRight, bodyYUpperRight,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,
                           failureProbabilities,action,futureAction,angle,ax,simple=True)

    #Draw mini s/c to scale, scale down safety region to match!

    # Translate between drawing frame and physical frame (as artist frame origin is spacecraft lower left corner)
    #bodyXUpperRight = physicalState[0]-bodyScale/2
    #bodyYUpperRight = physicalState[2]-bodyScale/2
    #print(bodyXUpperRight,bodyYUpperRight,bodyScale)


    #Plot physical state x,y as a circle of radius 1 (trying smaller .1 to make it easier to see )
    #ax.add_artist(Circle((physicalState[0],physicalState[2]),radius=.1,
    #        facecolor="black", edgecolor="black", linewidth=linewidth))

    #Actual size spacecraft is shown as just a tick instead
    #ax.vlines(x=physicalState[0], ymin=physicalState[2]-.5, ymax=physicalState[2]+.5, linewidth=.2, color="k")
    #ax.hlines(y=physicalState[2], xmin=physicalState[0]-.5, xmax=physicalState[0]+.5, linewidth=.2, color="k")
    #Dot instead?
    ax.plot(physicalState[0],physicalState[2],"ko",markersize=.1)

    # position belief
    if not groundTruthFlag:
        plotPositionBelief(ax,positionGaussianExtent,physicalState,beliefTuple,numMeshForPositionGaussian)

    # tree
    if rootNode is not None:
        tree = rootNodeToTree(rootNode)
        segments = treeToSegments(tree)
        lineCollection = LineCollection(segments,
            linewidth=1.5, colors=treeColor, alpha=0.15, zorder=-10)
        ax.add_collection(lineCollection)

    ax.set_xlim(plottingBounds[0,:])
    ax.set_ylim(plottingBounds[1,:])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    #Set background color to transparent
    ax.set_alpha(0)

    if legendFlag:
        _makeLegend(ax)

    #Theta label (in degrees) add this to help with seeing the current orientation
    #x,y label added too to see drifts
    #Disable on belief and safety only and hardware
    if not groundTruthFlag: #Don't show this for the first case, since we aren't moving
        plt.text(.05,.95,f"x={np.round(physicalState[0],1)}, y={np.round(physicalState[2],1)}\nvx={np.round(physicalState[1],1)},vy={np.round(physicalState[3],1)}",
                 horizontalalignment='left',verticalalignment='center',transform = ax.transAxes)
        if rotationFlag:
            plt.text(.05,.88,fr"$\theta$={np.round(np.rad2deg(physicalState[4]),1)}°",horizontalalignment='left',verticalalignment='center',transform = ax.transAxes)
            plt.text(.05,.84,fr"$\omega$={np.round(np.rad2deg(physicalState[5]),1)}°/s",horizontalalignment='left',verticalalignment='center',transform = ax.transAxes)

    #Label sensors
    if groundTruthFlag:
        plt.text(.25,.75,"")

def _drawAndScaleSpacecraft(bodyScale,bodyXUpperRight, bodyYUpperRight,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,
                           failureProbabilities,action,futureAction,angle,ax,simple=False): # pylint: disable=invalid-name
    """
    Helper function that draws and positions the spacecraft as desired.
    """
    #bodyScale = scale
    artists = _drawSpacecraftComponents(bodyScale,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,failureProbabilities,action,futureAction,simple)

    #Apply rotation
    transform = Affine2D().translate(bodyXUpperRight, bodyYUpperRight)
    #Add rotations if we are considering them (if not, angle will be 0)

    #Theta is in radians, rotate around center. center seems to have been set to the bottom left corner of the s/c so moving back to center of mass
    transform = transform.rotate_around(bodyXUpperRight+bodyScale/2,bodyYUpperRight+bodyScale/2,angle)
    for artist in artists:
        artist.set_transform(transform + ax.transData)
    # add to ax
    for artist in artists:
        ax.add_artist(artist)

def _drawSpacecraftComponents(scale,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,failureProbabilities,action,futureAction=None,simple=False):
    """
    Sub method to draw each component of the spacecraft.
    """

    # body
    bodyLength = scale
    bodyColor="white"
    if simple:
        bodyAlpha = .5
    else:
        bodyAlpha = 1

    bodyCenterXY = (0,0)
    bodyRectangle = Rectangle(bodyCenterXY, bodyLength, bodyLength,
        facecolor=bodyColor, alpha=bodyAlpha, edgecolor=edgecolor, linewidth=linewidth)

    if simple:
        return [bodyRectangle]

    # actuators
    actuatorRectangles, actuationTriangles, futureActuationTriangles = _drawActuators(scale,bodyLength,numAct,failureBeliefColorMap,failureProbabilities,edgecolor,linewidth,action,futureAction)

    # sensors
    sensorCircles = _drawSensors(scale,bodyLength,numAct,numSen,failureBeliefColorMap,failureProbabilities,edgecolor,linewidth)

    #Show which way was originally +y
    upArrowXY = np.array([1/4* bodyLength, 15/16 *bodyLength])
    upArrow = FancyArrowPatch(posA=upArrowXY-np.array([0,1/4*bodyLength]),posB=upArrowXY,arrowstyle="->",mutation_scale=5)

    ##Hardware vis, want to point to -X now
    #upArrowXY = np.array([1/16 *bodyLength,1/4* bodyLength])
    #upArrow = FancyArrowPatch(posA=upArrowXY+np.array([1/4*bodyLength,0]),posB=upArrowXY,arrowstyle="->",mutation_scale=5)

    # #Apply rotation
    #transform = Affine2D().translate(bodyXUpperRight, bodyYUpperRight)
    ##Add rotations if we are considering them (if not, angle will be 0)
#
    ##Theta is in radians, rotate around center. center seems to have been set to the bottom left corner of the s/c so moving back to center of mass
    #transform = transform.rotate_around(bodyXUpperRight+bodyScale/2,bodyYUpperRight+bodyScale/2,angle)
    #for artist in artists:
    #    artist.set_transform(transform + ax.transData)


    if futureAction is not None:
        # make master artist list
        artists = [bodyRectangle, *actuatorRectangles, *sensorCircles, upArrow, *actuationTriangles, *futureActuationTriangles]
    else:
        # make master artist list
        artists = [bodyRectangle, *actuatorRectangles, *sensorCircles, upArrow, *actuationTriangles]
    return artists

def _drawActuators(scale,bodyLength,numAct,failureBeliefColorMap,failureProbabilities,edgecolor,linewidth,action,futureAction=None):
    """
    Sub method to draw actuators and actuator firings
    """
    actuatorScale = scale * 1/8
    actuatorLength = actuatorScale
    actuatorXYs = _getActuatorXYs(bodyLength,actuatorScale)
    actuatorRectangles = []
    for iActuator in range(numAct):
        actuatorColor = failureBeliefColorMap(1-failureProbabilities[iActuator])
        actuatorAlpha = 1.0
        actuatorRectangle = Rectangle(actuatorXYs[iActuator],
            actuatorLength, actuatorLength,
            facecolor=actuatorColor, alpha=actuatorAlpha,
            edgecolor=edgecolor, linewidth=linewidth)
        actuatorRectangles.append(actuatorRectangle)

    # actuation!
    magicNum = np.sqrt(2)/2
    actuationPoints = actuatorScale * np.array([
        [0,0],
        [1,0],
        [0.5, magicNum]])
    arcPoints = 2*actuatorScale * np.array([
        [-.25,-.7],
        [.75,-.7]])
    actuationRotations = [90, 90, 270, 270, 180, 180, 0, 0,90,270]

    actuationShifts,futureActuationShifts = _getActuationShifts(magicNum,actuatorScale)

    actuationTriangles = _drawActuation(numAct,action,actuationPoints,arcPoints,actuationRotations,actuatorXYs,actuationShifts)

    #Plot the action we will take (usually too much information)
    if futureAction is not None:
        futureActuationTriangles = _drawActuation(numAct,futureAction,actuationPoints,arcPoints,actuationRotations,actuatorXYs,futureActuationShifts)
    else:
        futureActuationTriangles = None

    return actuatorRectangles, actuationTriangles, futureActuationTriangles

def _actuationRotationMatrix(theta):
    """
    Computes a rotation matrix for the given theta.
    """
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

def _getActuatorXYs(bodyLength,actuatorScale):
    """
    Sub method to make defining the XYs cleaner to read
    """
    actuatorXYs = [ #All the positions to put the actuators as. Note if we have less actuators, the wheels just get left off!
        (bodyLength, 5/8 * bodyLength), #FX-MZ+
        (bodyLength, 1/4 * bodyLength), #FX-MZ-
        (-1 * actuatorScale, 1/4 * bodyLength), #FX+MZ+
        (-1 * actuatorScale, 5/8 * bodyLength), #FX+MZ-
        (1/4 * bodyLength, bodyLength), #FY-MZ+
        (5/8 * bodyLength, bodyLength),         #FY-MZ-
        (5/8 * bodyLength, -1 * actuatorScale), #FY+MZ+
        (1/4 * bodyLength, -1 * actuatorScale), #FY+MZ-
        (5/8 *bodyLength,7/16 * bodyLength), #Wheel locations
        (bodyLength/4,7/16 * bodyLength),
        ]
    return actuatorXYs

def _getActuationShifts(magicNum,actuatorScale):
    """
    Sub method to make defining the shifts cleaner to read
    """
    actuationShifts = [
        (magicNum * actuatorScale + actuatorScale, 0),
        (magicNum * actuatorScale + actuatorScale, 0),
        (-magicNum*actuatorScale, actuatorScale),
        (-magicNum*actuatorScale, actuatorScale),
        (actuatorScale, magicNum*actuatorScale+actuatorScale),
        (actuatorScale, magicNum*actuatorScale+actuatorScale),
        (0, -magicNum*actuatorScale),
        (0, -magicNum*actuatorScale),
        (magicNum * actuatorScale - actuatorScale, 0),#Wheel shifts
        (-magicNum*actuatorScale + 2*actuatorScale, actuatorScale),
    ]

    futureActuationShifts = [
        (.5*magicNum * actuatorScale + actuatorScale, 0),
        (.5*magicNum * actuatorScale + actuatorScale, 0),
        (-magicNum*.5*actuatorScale, actuatorScale),
        (-magicNum*.5*actuatorScale, actuatorScale),
        (actuatorScale, .5*magicNum*actuatorScale+actuatorScale),
        (actuatorScale, .5*magicNum*actuatorScale+actuatorScale),
        (0, -magicNum*.5*actuatorScale),
        (0, -magicNum*.5*actuatorScale),
        (magicNum  *.5* actuatorScale + actuatorScale, 0),#Wheel shifts
        (-magicNum *.5*actuatorScale, actuatorScale),
    ]
    return actuationShifts,futureActuationShifts

def _drawActuation(numAct,action,actuationPoints,arcPoints,actuationRotations,actuatorXYs,actuationShifts,futureAction=False):
    """
    Sub method to draw actuations
    """
    actuationTriangles = []
    for iActuator in range(numAct):
        #Add in arc!!
        if np.abs(action[iActuator]) > 1e-3: #Wheel actions can be negative
            if iActuator >= 8:
                points = arcPoints @ _actuationRotationMatrix(actuationRotations[iActuator]*np.pi/180) + actuatorXYs[iActuator]
                points += actuationShifts[iActuator]
                #If u negative reverse direction
                if action[iActuator] < 0:
                    #points = points[::-1]
                    style = "->"
                else:
                    style = "<-"
                #Positive increases to the wheel speed is negative torque
                actuationArrow = FancyArrowPatch(posA=points[0],posB=points[1],arrowstyle=style,connectionstyle=ConnectionStyle.Arc3(rad=.5),mutation_scale=5)
                actuationTriangles.append(actuationArrow)
            else:
                # if True:
                # print("ii_a")
                points = actuationPoints @ _actuationRotationMatrix(actuationRotations[iActuator]*np.pi/180) + actuatorXYs[iActuator]
                points += actuationShifts[iActuator]
                if futureAction:
                    actuationTri = Polygon(points, closed=True, fill=True, color="orange",alpha=.3)
                else:
                    actuationTri = Polygon(points, closed=True, fill=True, color="orange")
                actuationTriangles.append(actuationTri)
    return actuationTriangles


def _drawSensors(scale,bodyLength,numAct,numSen,failureBeliefColorMap,failureProbabilities,edgecolor,linewidth):
    """
    Sub method to draw sensors
    """
    sensorScale = scale * 1/8
    sensorRadius = sensorScale
    sensorXYs = [
        (0,0),
        (bodyLength,0),
        (bodyLength,bodyLength),
        (0,bodyLength),
        (bodyLength/2,bodyLength/2 + 2*sensorScale), #Rotation Sensors
        (bodyLength/2,bodyLength/2 - 2*sensorScale),
        ]
    #Use this to differentiate x vs y sensors
    #sensorHatches = ["x","x",None,None,None,None]
    sensorHatches = [None,None,None,None,None,None]
    sensorCircles = []
    for iSensor in range(numSen):
        sensorColor = failureBeliefColorMap(1-failureProbabilities[iSensor + numAct])
        sensorAlpha = 1.0
        sensorCircle = Circle(sensorXYs[iSensor],radius=sensorRadius,
            facecolor=sensorColor, alpha=sensorAlpha,
            edgecolor=edgecolor, linewidth=linewidth,hatch=sensorHatches[iSensor])
        sensorCircles.append(sensorCircle)
    return sensorCircles

#ax is widely used with matplotlib for axis, so will make an exception
def plotPositionBelief(ax,positionGaussianExtent,physicalState,beliefTuple,numMeshForPositionGaussian): # pylint: disable=invalid-name
    """
    Sub method for plotting the position belief distribution

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw s/c in
    positionGaussianExtent : float
        Interesting region of the Gaussian, set to make computing more efficient. Currently not very well motivated
    physicalState : array, shape(numState)
        Current physical state
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures,numAct+numSen)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state.
    numMeshForPositionGaussian : int
        How fine the Gaussian mesh is.
    """
    #First plot background of zeros (so everything looks the same, as no longer scaling up the gaussian)
    #plotX = np.linspace(plottingBounds[0,0], plottingBounds[0,1], numMeshForPositionGaussian)
    #plotY = np.linspace(plottingBounds[1,0], plottingBounds[1,1], numMeshForPositionGaussian)
    #plotMeshX, plotMeshY = np.meshgrid(plotX, plotY)

    #Trying to be more presentable
    #ax.contourf(plotMeshX, plotMeshY, 0*plotMeshX, zorder=-1,colors=['#481B6D'])
    #ax.contourf(plotMeshX, plotMeshY, 0*plotMeshX, zorder=-1,colors=['#FFFFFF'])

    #No longer scaling up
    positionX, positionY, pdfPositionBelief = evalMultimodalGaussian2(beliefTuple, positionGaussianExtent, physicalState, numMeshForPositionGaussian)
    #Setting background to white for cleaner plot
    beliefColorMap = get_cmap("viridis") #.copy()
    #beliefColorMap.set_under('w')
    #beliefColorMap.mappable.set_clim(vmin=.01)
    #Masking out small values of belief to make it so the color map fades to white.
    #pdfPositionBelief = pdfPositionBelief.at[np.where(pdfPositionBelief<.005)].set(-np.inf)
    pdfPositionBelief[np.where(pdfPositionBelief<.005)] = -np.inf
    beliefPlot = ax.contourf(positionX, positionY, pdfPositionBelief, cmap=beliefColorMap,zorder=-2)#,vmin=1000) #Should be under unsafe (-1)
    beliefPlot.cmap.set_under('w')
    #beliefPlot.cmap.set_under('w') #Looking at color.py, think I need to interrupt normalization somehow
    #beliefPlot.cmap.set_clim(vmin=.01)

    #beliefPlot = ax.contourf(positionX, positionY, pdfPositionBelief,zorder=0)
    #beliefColorMap = plt.colorbar(beliefPlot)
    #beliefColorMap.set_under('w')
    #beliefColorMap.mappable.set_clim(vmin=.01)

def _makeLegend(ax): # pylint: disable=invalid-name
    """
    Sub method to make legend for the plot
    """
    legendActuator, = ax.plot(np.nan, np.nan, color="white", marker="s", markeredgecolor="black", linestyle="None")
    legendSensor, = ax.plot(np.nan, np.nan, color="white", marker="o", markeredgecolor="black", linestyle="None")
    legendActuation, = ax.plot(np.nan, np.nan, color="orange", marker="^", linestyle="None")
    legendFail, = ax.plot(np.nan, np.nan, color="red", marker="o", linestyle="None")
    legendNominal, = ax.plot(np.nan, np.nan, color="green", marker="o", linestyle="None")
    ax.legend(
        [legendActuator, legendSensor, legendActuation, legendFail, legendNominal],
        ["Actuator", "Sensor", "Control Input", "Fault", "Nominal"]
        )

def translateArray(arrayToTranslate, shiftX, shiftY):
    """
    Translates the array by the specified amount
    """
    # create the translation matrix using tx and ty, it is a NumPy array
    width, height = arrayToTranslate.shape[0], arrayToTranslate.shape[1]
    translationMatrix = np.array([
        [1, 0, shiftX],
        [0, 1, shiftY]
        ], dtype=np.float32)
    arrayToTranslate = cv2.warpAffine(src=arrayToTranslate, M=translationMatrix, dsize=(width, height)) # pylint: disable=no-member
    return arrayToTranslate


def evalMultivariateGaussian(pos, mean, sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    dim = mean.shape[0]
    sigmaDeterminate = np.linalg.det(sigma)
    sigmaInverse = np.linalg.inv(sigma)
    normalization = np.sqrt((2*np.pi)**dim * sigmaDeterminate)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    intermediateExponentTerm = np.einsum('...k,kl,...l->...', pos-mean, sigmaInverse, pos-mean)
    return np.exp(-1* intermediateExponentTerm / 2) / normalization

def evalMultimodalGaussian2(beliefTuple, positionGaussianExtent, physicalState, numMeshForPositionGaussian):
    """
    Creates a multi modal Gaussian using each element of the belief tuple.
    evalRegion is the sub set of the plot region that we evaluate the gaussian over, as not all of this region is necessarily interesting

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures,numAct+numSen)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state.
    positionGaussianExtent : float
        Interesting region of the Gaussian, set to make computing more efficient. Currently not very well motivated
    plotRegion : array, shape(2,2)
        Bounds of the axis
    physicalState : array, shape(numState)
        Current physical state
    numMeshForPositionGaussian : int
        How fine the Gaussian mesh is.
    rotationFlag : boolean (default=False)
        Whether or not this is a 3DOF spacecraft
    """

    #We want to evaluate the gaussian near the peak of the belief, which we don't know directly
    #We do however, expect the peak to be near the actual physical state of the s/c, so we can use this as our center.
    physicalStateX = physicalState[0]
    physicalStateY = physicalState[2]

    # eval in "x" coordinates
    evalX = np.linspace(-positionGaussianExtent, positionGaussianExtent, numMeshForPositionGaussian)
    evalY = np.linspace(-positionGaussianExtent, positionGaussianExtent, numMeshForPositionGaussian)
    # Move eval region to area we expect the Gaussian to be significant over
    evalX += physicalStateX
    evalY += physicalStateY
    evalMeshX, evalMeshY = np.meshgrid(evalX, evalY)
    evalPos = np.empty(evalMeshX.shape + (2,))
    evalPos[:, :, 0] = evalMeshX
    evalPos[:, :, 1] = evalMeshY

    #plotX = np.linspace(plotRegion[0,0], plotRegion[0,1], numMeshForPositionGaussian)
    #plotY = np.linspace(plotRegion[1,0], plotRegion[1,1], numMeshForPositionGaussian)
    #plotMeshX, plotMeshY = np.meshgrid(plotX, plotY)

    beliefPdf = 0 * evalMeshX
    #Pick out variables to plot gaussian on, always x and y, so mask out the rest.
    gaussianMask = np.zeros((2,len(physicalState)))
    gaussianMask[0,0] = 1
    gaussianMask[1,2] = 1
    gaussianMask=gaussianMask.T
    #Old way, not robust
    #if not rotationFlag:
    #    gaussianMask = np.array([[1,0,0,0],[0,0,1,0]]).T
    #else:
    #    gaussianMask = np.array([[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]]).T
    for (weight, positionFilter) in zip(beliefTuple[0], beliefTuple[1]):
        physicalStateBeliefMean = positionFilter[0]
        physicalStateBeliefCovariance = positionFilter[1:1+len(physicalStateBeliefMean)]
        mean = physicalStateBeliefMean @ gaussianMask
        sigma = 1e-3 * np.eye(2) + gaussianMask.T @ physicalStateBeliefCovariance @ gaussianMask
        beliefPdf += weight * evalMultivariateGaussian(evalPos, mean, sigma)

    # compute "shift" coordinates in pixels of plotMeshX
    #plotDx = plotX[1] - plotX[0] # pixel per x
    #plotDy = plotY[1] - plotY[0]
    #beliefPdf = translateArray(beliefPdf, physicalStateX / plotDx, physicalStateY / plotDy)
    return evalX, evalY, np.array(beliefPdf)

def computeComponentFailureProbabilities(beliefTuple, possibleFailures, numActuators, numSensors):
    """
    Sums over the weighted possible failures to get the component wise possible failures
    """
    componentFailureProbabilities = np.zeros(numActuators+numSensors)
    for jFailure,possibleFailure in enumerate(possibleFailures):
        for iComponent,component in enumerate(possibleFailure):
            #Check if this component failed in this belief
            if component == 0:
                #Add to weighting

                componentFailureProbabilities[iComponent] += beliefTuple[0][jFailure]
    return componentFailureProbabilities

def computeMean(belief):
    """
    Computes the mean of the provided belief
    """
    mean = 0.0
    for (weight, kalmanFilter) in zip(belief[0], belief[1]):
        mean += weight * kalmanFilter[0] #First row is average
    return mean

def rootNodeToTree(rootNode):
    """
    Takes the root node of a tree data structure and converts into a list of each node and their parent
    """
    tree = []
    toAdd = [(rootNode,-1)]
    while len(toAdd) > 0:
        currentNode, parentIdx = toAdd.pop(0)
        #Added in v2 compatibility, will have belief tuple if v2
        if hasattr(currentNode, 'beliefTuple'):
            row = [currentNode.beliefTuple, parentIdx]
            tree.append(row)
            parentIdx = len(tree) - 1
            for decisionChild in currentNode.children:
                for beliefChild in decisionChild.children:
                    toAdd.append((beliefChild,parentIdx))
        #No longer supporting v1 trees
        else:
            incompatibleTree = "The rootNode is a type that is unsupported by the current version of failurePy."
            raise ValueError(incompatibleTree)
    return tree

def treeToSegments(tree):
    """
    Takes a tree structure and returns the segments that make up each branch
    """
    segments = []
    for row in tree:
        belief = row[0]
        parentIdx = row[1]
        mean = computeMean(belief)
        if parentIdx >= 0:
            parentMean = computeMean(tree[parentIdx][0])
            segments.append([[mean[0],mean[2]],[parentMean[0],parentMean[2]]])
    return segments

def plotUnsafeRegions(ax,safetyFunctionF,plottingBounds,resolution=200):
    """
    Function that overlays on the visualization any unsafe region.
    It does so by evaluating the safetyConstrainedRewardF at a grid of points using a fake belief

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw s/c in
    safetyFunctionF : function
        Constraints that physical states must satisfy to be safe. Returns 1 if safe, 0 if not.
    plottingBounds : array, shape(2,2)
        Bounds of the axis
    resolution : int (default=200)
        How high of a resolution the safe zone should be drawn in.
    """

    #Use custom color map to set obstacles (obstacleSafeZ=0) to red, and all other to transparent.
    #Each channel specified separately, each row is an anchor point, with before and after values specified
    colorMapDict = {'red':  [(0.0,  1.0, 1.0),
                             (1.0,  1.0, 1.0)],
                    'green':  [(0.0,  0.0, 0.0),
                             (1.0,  1.0, 1.0)],
                    'blue':  [(0.0,  0.0, 0.0),
                             (1.0,  1.0, 1.0)],
                    'alpha':  [(0.0,  0.5, 0.5),
                             (1.0,  0.0, 0.0)],}


    obstacleMaskColorMap = LinearSegmentedColormap('obstacleMask',colorMapDict)

    #Get points to evaluate over
    xMesh,yMesh = getMeshOverPlottingBounds(plottingBounds,resolution)

    ##Make "meshed" physical states
    #meshedPhysicalStates = np.array([xMesh,0*xMesh,yMesh,0*yMesh])

    obstacleSafeZ = np.zeros((resolution,resolution))
    #Split based on safetyFunctionF type
    if safetyFunctionF.__name__ == "worstCaseSafetyFunctionF":
        for iXCord in range(resolution):
            for jYCord in range(resolution):
                #Note that our thinking of cartesian 2d arrays is flipped! so jYCord gives the columns, while iXCord gives the rows!
                safetyReturn = safetyFunctionF(np.array([xMesh[0,iXCord],0,yMesh[jYCord,0],0]))
                if np.sign(safetyReturn) == -1:
                    obstacleSafeZ[jYCord,iXCord] = 1
                else:
                    obstacleSafeZ[jYCord,iXCord] = 0
    else:
        for iXCord in range(resolution):
            for jYCord in range(resolution):
                #Note that our thinking of cartesian 2d arrays is flipped! so jYCord gives the columns, while iXCord gives the rows!
                obstacleSafeZ[jYCord,iXCord] = safetyFunctionF(np.array([xMesh[0,iXCord],0,yMesh[jYCord,0],0]))

    ax.contourf(xMesh, yMesh, obstacleSafeZ, cmap=obstacleMaskColorMap,zorder=-1) #Should be over position belief (-2)

def getMeshOverPlottingBounds(plottingBounds,numMeshPointsPerAxis):
    """
    Function that returns a mesh given the plotting bounds and number of points

    Parameters
    ----------
    plottingBounds : array, shape(2,2)
        Bounds of the axis
    numMeshPointsPerAxis : int
        How fine the mesh is.

    Returns
    xMesh : array, shape(numMeshPointsPerAxis,numMeshPointsPerAxis)
        x-dimension of the mesh
    yMesh : array, shape(numMeshPointsPerAxis,numMeshPointsPerAxis)
        y-dimension of the mesh
    """

    plotX = np.linspace(plottingBounds[0,0], plottingBounds[0,1], numMeshPointsPerAxis)
    plotY = np.linspace(plottingBounds[1,0], plottingBounds[1,1], numMeshPointsPerAxis)
    xMesh, yMesh = np.meshgrid(plotX, plotY)
    return xMesh,yMesh


def getNumActNumSen(rotationFlag,action):
    """
    Helper method to set the numAct and numSen variables
    """
    #Number of sensor or actuators is pre set
    if rotationFlag:
        numSen = 6
    else:
        numSen = 6 #Add two rotation sensors

    numAct = len(action)

    return numAct, numSen

def saveFigs(fileName,savePngs=False):
    """
    Saves the generated figures as a pdf to given file name. Optionally save to pngs too
    """
    outputPdf = PdfPages(fileName)
    for iFigure in plt.get_fignums():
        figure = plt.figure(iFigure)
        outputPdf.savefig(figure,dpi=1000,facecolor=figure.get_facecolor(), edgecolor='none',transparent=True)
        #Save pngs if needed
        if savePngs:
            saveDirectory = os.path.join(os.path.dirname(fileName),"pngs")
            figure.savefig(os.path.join(saveDirectory,str(iFigure)+".png"))
        plt.close(figure)
    outputPdf.close()
