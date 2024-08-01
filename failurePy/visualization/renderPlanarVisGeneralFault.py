"""
Old render_2d_vis.py file from v1, converted to match v2 conventions for clarity.

Main method moved to wrapper

Some other minor changes to improve readability/remove unused code
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, ConnectionStyle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from failurePy.visualization.renderPlanarVis import plotPositionBelief,getNumActNumSen,rootNodeToTree,treeToSegments

plt.rcParams.update({'font.size': 10})

#ax is widely used with matplotlib for axis, so will make an exception. Similarly lots of plotting lines so we have an exception for number of statements.
def drawSpacecraft(physicalState, action, beliefTuple, possibleFailures, rootNode, ax, plottingBounds, legendFlag=False,rotationFlag=False): # pylint: disable=invalid-name,too-many-statements
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
    failureThreshold = .5 #This is the maximum failure chance that a component can have before it starts to show as red.
    numMeshForPositionGaussian = 200 #How many points in each axis will be sampled when making the position Gaussian
    positionGaussianExtent = 10 # what is interesting gaussian region?

    # make special colormap, using our threshold to set the linear scaling on the colors. Currently 0-.78 -> red - white. .78-1 -> white to green
    colorMapColors = ["green","white","red"] # Now 0 deg/bias is good!
    anchorValues = [0,1-failureThreshold,1.]
    colorMapAnchorPointList = list(zip(anchorValues,colorMapColors))
    failureBeliefColorMap = LinearSegmentedColormap.from_list('rg',colorMapAnchorPointList, N=256)

    # get failure probabilities:
    componentExpectedDegradation,componentExpectedBiases = computeComponentExpectedDegradationAndBiases(beliefTuple, possibleFailures, numAct, numSen)
    print("weights: ", beliefTuple[0],
          "\n expected degradation: ",componentExpectedDegradation,
          "\n expected biases", componentExpectedBiases,)

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
                           componentExpectedDegradation,componentExpectedBiases,action,angle,ax)

    #Draw actual size s/c, 1 meter size
    bodyXUpperRight = physicalState[0]-1/2
    bodyYUpperRight = physicalState[2]-1/2
    bodyScale = 1
    #Disable for belief and safety only and hardware vis of mini map
    _drawAndScaleSpacecraft(bodyScale,bodyXUpperRight, bodyYUpperRight,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,
                           componentExpectedDegradation,componentExpectedBiases,action,angle,ax,simple=True)

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
                           componentExpectedDegradation,componentExpectedBiases,action,angle,ax,simple=False): # pylint: disable=invalid-name
    """
    Helper function that draws and positions the spacecraft as desired.
    """
    #bodyScale = scale
    artists = _drawSpacecraftComponents(bodyScale,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,componentExpectedDegradation,componentExpectedBiases,action,simple)

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

def _drawSpacecraftComponents(scale,edgecolor,linewidth,numAct,numSen,failureBeliefColorMap,componentExpectedDegradation,componentExpectedBiases,action,simple=False):
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
    actuatorRectangles, actuationTriangles, actuationBiasTriangles = _drawActuators(scale,bodyLength,numAct,failureBeliefColorMap,componentExpectedDegradation,
                                                                                    componentExpectedBiases,edgecolor,linewidth,action)

    # sensors
    sensorCircles,sensorBiasArrows = _drawSensors(scale,bodyLength,numAct,numSen,failureBeliefColorMap,componentExpectedDegradation,componentExpectedBiases,edgecolor,linewidth)

    ##Show which way was originally +y
    #upArrowXY = np.array([1/4* bodyLength, 15/16 *bodyLength])
    #upArrow = FancyArrowPatch(posA=upArrowXY-np.array([0,1/4*bodyLength]),posB=upArrowXY,arrowstyle="->",mutation_scale=5)

    #Hardware vis, want to point to -X now
    upArrowXY = np.array([1/16 *bodyLength,1/4* bodyLength])
    upArrow = FancyArrowPatch(posA=upArrowXY+np.array([1/4*bodyLength,0]),posB=upArrowXY,arrowstyle="->",mutation_scale=5)

    # #Apply rotation
    #transform = Affine2D().translate(bodyXUpperRight, bodyYUpperRight)
    ##Add rotations if we are considering them (if not, angle will be 0)
#
    ##Theta is in radians, rotate around center. center seems to have been set to the bottom left corner of the s/c so moving back to center of mass
    #transform = transform.rotate_around(bodyXUpperRight+bodyScale/2,bodyYUpperRight+bodyScale/2,angle)
    #for artist in artists:
    #    artist.set_transform(transform + ax.transData)

    # make master artist list
    artists = [bodyRectangle, *actuatorRectangles, *sensorCircles,*sensorBiasArrows, upArrow, *actuationTriangles, *actuationBiasTriangles]
    return artists

def _drawActuators(scale,bodyLength,numAct,failureBeliefColorMap,componentExpectedDegradation,componentExpectedBiases,edgecolor,linewidth,action):
    """
    Sub method to draw actuators and actuator firings
    """
    actuatorScale = scale * 1/8
    actuatorLength = actuatorScale
    actuatorXYs = _getActuatorXYs(bodyLength,actuatorScale)
    actuatorRectangles = []
    for iActuator in range(numAct):
        #actuatorColor = failureBeliefColorMap(1-componentExpectedDegradation[iActuator])
        actuatorColor = failureBeliefColorMap(componentExpectedDegradation[iActuator])
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

    actuationShifts,biasActuationShifts = _getActuationShifts(magicNum,actuatorScale)

    actuationTriangles,actuationBiasTriangles = _drawActuation(numAct,action,actuationPoints,arcPoints,actuationRotations,actuatorXYs,actuationShifts,componentExpectedBiases,biasActuationShifts)



    return actuatorRectangles, actuationTriangles, actuationBiasTriangles

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

    biasActuationShifts = [
        (.5*magicNum * actuatorScale + actuatorScale, 0),
        (.5*magicNum * actuatorScale + actuatorScale, 0),
        (-magicNum*.5*actuatorScale, actuatorScale),
        (-magicNum*.5*actuatorScale, actuatorScale),
        (actuatorScale, .5*magicNum*actuatorScale+actuatorScale),
        (actuatorScale, .5*magicNum*actuatorScale+actuatorScale),
        (0, -magicNum*.5*actuatorScale),
        (0, -magicNum*.5*actuatorScale),
        (-magicNum  *.5* actuatorScale + actuatorScale, 0),#Wheel shifts
        (+magicNum *.5*actuatorScale, actuatorScale),
    ]
    return actuationShifts,biasActuationShifts

def _drawActuation(numAct,action,actuationPoints,arcPoints,actuationRotations,actuatorXYs,actuationShifts,componentExpectedBiases,biasActuationShifts):
    """
    Sub method to draw actuations
    """
    actuationTriangles = []
    actuationBiases = []
    for iActuator in range(numAct):
        biasAlpha = componentExpectedBiases[iActuator]
        if np.isnan(biasAlpha):
            biasAlpha=1
            print("nan biases detected")

        if iActuator >= 8:
            points = arcPoints @ _actuationRotationMatrix(actuationRotations[iActuator]*np.pi/180) + actuatorXYs[iActuator]
            biasPoints = points + biasActuationShifts[iActuator]
            if np.abs(action[iActuator]) > 1e-3: #Wheel actions can be negative
                wheelArrowPoints = points + actuationShifts[iActuator]
                #If u negative reverse direction
                if action[iActuator] < 0:
                    #points = points[::-1]
                    style = "->"
                else:
                    style = "<-"
                #Positive increases to the wheel speed is negative torque
                actuationArrow = FancyArrowPatch(posA=wheelArrowPoints[0],posB=wheelArrowPoints[1],arrowstyle=style,connectionstyle=ConnectionStyle.Arc3(rad=.5),mutation_scale=5)
                actuationTriangles.append(actuationArrow)

            #Bias arrow if one exits (always positive for now)
            actuationBiasArrow = FancyArrowPatch(posA=biasPoints[0],posB=biasPoints[1],arrowstyle="<-",connectionstyle=ConnectionStyle.Arc3(rad=.5),mutation_scale=5,
                                                 facecolor="r",edgecolor="r",alpha=biasAlpha)
            actuationBiases.append(actuationBiasArrow)
        else:
            # if True:
            # print("ii_a")
            points = actuationPoints @ _actuationRotationMatrix(actuationRotations[iActuator]*np.pi/180) + actuatorXYs[iActuator]
            biasPoints = points + biasActuationShifts[iActuator]
            if action[iActuator] > 1e-3:
                thrusterJetPoints = points + actuationShifts[iActuator]
                actuationTri = Polygon(thrusterJetPoints, closed=True, fill=True, color="orange")
                actuationTriangles.append(actuationTri)
            actuationBiasTri = Polygon(biasPoints, closed=True, fill=True, color="red",alpha=biasAlpha)
            actuationBiases.append(actuationBiasTri)
    return actuationTriangles,actuationBiases


def _drawSensors(scale,bodyLength,numAct,numSen,failureBeliefColorMap,componentExpectedDegradation,componentExpectedBiases,edgecolor,linewidth):
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
    biasArrows = []
    for iSensor in range(numSen):
        #sensorColor = failureBeliefColorMap(1-componentExpectedDegradation[iSensor + numAct])
        sensorColor = failureBeliefColorMap(componentExpectedDegradation[iSensor + numAct])
        sensorAlpha = 1.0
        sensorCircle = Circle(sensorXYs[iSensor],radius=sensorRadius,
            facecolor=sensorColor, alpha=sensorAlpha,
            edgecolor=edgecolor, linewidth=linewidth,hatch=sensorHatches[iSensor])
        sensorCircles.append(sensorCircle)
        biasAlpha = componentExpectedBiases[iSensor + numAct]
        if np.isnan(biasAlpha):
            biasAlpha=1
            print("nan biases detected")

        biasArrow = FancyArrowPatch(posA=sensorXYs[iSensor]+np.array([-1/3*sensorRadius,0]),posB=sensorXYs[iSensor]+np.array([sensorRadius,0]),
                                    arrowstyle="-|>",mutation_scale=5,facecolor="r",edgecolor="k",alpha=biasAlpha)
        biasArrows.append(biasArrow)
    return sensorCircles,biasArrows

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



def computeComponentExpectedDegradationAndBiases(beliefTuple, possibleFailures, numActuators, numSensors):
    """
    Sums over the weighted possible failures to get the component wise possible failures
    """
    componentExpectedDegradation = np.zeros(numActuators+numSensors)
    componentExpectedBiases = np.zeros(numActuators+numSensors)
    for jFailure,possibleFailure in enumerate(possibleFailures):
        for iComponent,component in enumerate(possibleFailure):
            #Weighted average of the component degradation and biases
            #Need to factor in that in that failure is currently actDeg, actBiases, senDeg, senBiases
            if iComponent < numActuators:
                componentExpectedDegradation[iComponent] += component*beliefTuple[0][jFailure]
            elif iComponent < 2*numActuators:
                componentExpectedBiases[iComponent-numActuators] += component*beliefTuple[0][jFailure]
            elif iComponent < 2*numActuators + numSensors:
                componentExpectedDegradation[iComponent-numActuators] += component*beliefTuple[0][jFailure]
            else:
                componentExpectedBiases[iComponent-numActuators-numSensors] += component*beliefTuple[0][jFailure]
    return componentExpectedDegradation,componentExpectedBiases
