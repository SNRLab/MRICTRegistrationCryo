import os
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from sys import platform
import logging
import time

#from slicer.util import VTKObservationMixin

class MRICTRegistrationCryo(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MRICTRegistrationCryo"  
        self.parent.categories = ["Registration"]  
        self.parent.dependencies = []  
        self.parent.contributors = ["Subhra Sundar Goswami, Junichi Takuda"]  
        self.parent.helpText = """
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
"""
        # Additional initialization step after application startup is complete
        
        moduleDir = os.path.dirname(self.parent.path)
        for iconExtension in ['.svg', '.png']:
          iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
          if os.path.isfile(iconPath):
            parent.icon = qt.QIcon(iconPath)
            break

class MRICTRegistrationCryoWidget(ScriptedLoadableModuleWidget):
    """
    """

    def __init__(self, parent=None):
        
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        
        ScriptedLoadableModuleWidget.setup(self)
        
        self.registrationInProgress = False

        # IO collapsible button
        IOCategory = qt.QWidget()
        self.layout.addWidget(IOCategory)
        IOLayout = qt.QFormLayout(IOCategory)
        self.logic = MRICTRegistrationCryoLogic()
        self.logic.logCallback = self.addLog
        
        inputCollapsibleButton = ctk.ctkCollapsibleButton()
        inputCollapsibleButton.text = "Input Volumes"
        inputCollapsibleButton.collapsed = 0
        self.layout.addWidget(inputCollapsibleButton)
        
        # Layout within the dummy collapsible button
        IOLayout = qt.QFormLayout(inputCollapsibleButton)
        
        self.inputFixedVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputFixedVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputFixedVolumeSelector.selectNodeUponCreation = False
        self.inputFixedVolumeSelector.noneEnabled = False
        self.inputFixedVolumeSelector.addEnabled = False
        self.inputFixedVolumeSelector.removeEnabled = True
        self.inputFixedVolumeSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Fixed Volume: ", self.inputFixedVolumeSelector)

        self.inputMovingVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputMovingVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputMovingVolumeSelector.selectNodeUponCreation = False
        self.inputMovingVolumeSelector.noneEnabled = False
        self.inputMovingVolumeSelector.addEnabled = False
        self.inputMovingVolumeSelector.removeEnabled = True
        self.inputMovingVolumeSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Moving Volume: ", self.inputMovingVolumeSelector)
        
        #
        # output volume selector
        #
        outputCollapsibleButton = ctk.ctkCollapsibleButton()
        outputCollapsibleButton.text = "Output Volume"
        outputCollapsibleButton.collapsed = 0
        self.layout.addWidget(outputCollapsibleButton)
        
        # Layout within the dummy collapsible button
        IOLayout = qt.QFormLayout(outputCollapsibleButton)
        
        self.outputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.outputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.outputVolumeSelector.selectNodeUponCreation = False
        self.outputVolumeSelector.addEnabled = True
        self.outputVolumeSelector.removeEnabled = True
        self.outputVolumeSelector.renameEnabled = True
        self.outputVolumeSelector.noneEnabled = False
        self.outputVolumeSelector.showHidden = False
        self.outputVolumeSelector.showChildNodeTypes = False
        self.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.outputVolumeSelector.setToolTip("Select output volume name.")
        IOLayout.addRow("Output volume:", self.outputVolumeSelector)

        
        #
        # Advanced Area
        #
        advancedCollapsibleButton = ctk.ctkCollapsibleButton()
        advancedCollapsibleButton.text = "Advanced"
        advancedCollapsibleButton.collapsed = 1
        self.layout.addWidget(advancedCollapsibleButton)

        # Layout within the dummy collapsible button
        advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)

            
        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Start registration."
        self.applyButton.enabled = True #Should be False but changed to True for testing
        self.layout.addWidget(self.applyButton)
        
        
        self.statusLabel = qt.QPlainTextEdit()
        self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.statusLabel.setCenterOnScroll(True)
        self.layout.addWidget(self.statusLabel)
        
        
        
        # uiWidget = slicer.util.loadUI(self.resourcePath('UI/CryoAblation.ui'))
        # uiWidget.setMRMLScene(slicer.mrmlScene)
        # self.layout.addWidget(uiWidget)
        # self.ui = slicer.util.childWidgetVariables(uiWidget)

        # These connections ensure that we update parameter node when scene is closed
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.inputFixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.inputMovingVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        # self.removeObservers()
        
    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        # self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        self.setParameterNode(self.logic.getParameterNode())

        
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        We will implement it later
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        # if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
        #    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        # if self._parameterNode is not None:
        #    self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

        
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.inputFixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputFixedVolume"))
        self.inputMovingVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputMovingVolume"))
        self.outputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputFixedVolume") and self._parameterNode.GetNodeReference("InputMovingVolume"):
         #and self._parameterNode.GetNodeReference("OutputVolume")"""
            self.applyButton.toolTip = "Compute output Volume"
            self.applyButton.enabled = True
        else:
            self.applyButton.toolTip = "Select input and output volume nodes"
            self.applyButton.enabled = True #Should be False but changed to True for testing

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = True
        
        # Add vertical spacer
        # self.layout.addStretch(2)
        
        
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputFixedVolume", self.inputFixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputMovingVolume", self.inputMovingVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.outputVolumeSelector.currentNodeID)
        self._parameterNode.EndModify(wasModified)

    
    def onSelect(self):

        #self.applyButton.enabled = self.inputFixedVolumeSelector.currentNode() and self.inputMovingVolumeSelector.currentNode() """and self.outputVolumeSelector.currentNode()"""

        if not self.registrationInProgress:
          self.applyButton.text = "Apply"
          return
        self.updateBrowsers()
            
        
    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        
        if self.registrationInProgress:
          self.registrationInProgress = False
          self.abortRequested = True
          raise ValueError("User requested cancel.")
          self.cliNode.Cancel() # not implemented
          self.applyButton.text = "Cancelling..."
          self.applyButton.enabled = True   #Should be False but changed to True for testing
          return

        self.registrationInProgress = True
        self.applyButton.text = "Cancel"
        self.statusLabel.plainText = ''
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        
        
        
        try:
            # Compute output
            # with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self.logic.process(self.inputFixedVolumeSelector.currentNode(),
                        self.inputMovingVolumeSelector.currentNode(), self.outputVolumeSelector.currentNode())

            # Compute Registration output
            
            # The transformed volume after registration is written there
            # self.logic.process(self.outputVolumeSelector.currentNode())
            time.sleep(3)
            
        except Exception as e:
          print(e)
          self.addLog("Error: {0}".format(str(e)))
          import traceback
          traceback.print_exc()
          
        finally:
          slicer.app.restoreOverrideCursor()
          self.registrationInProgress = False
          self.onSelect() # restores default Apply button state
            
            

    def addLog(self, text):
        """Append text to log window
        """
        self.statusLabel.appendPlainText(text)
        slicer.app.processEvents()  # force update



class MRICTRegistrationCryoLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual computation done by the module.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        A = 100

    def process(self, inputFixedVolume, inputMovingVolume, outputVolume):
        """
        Run the processing algorithm.
        """
        
        #outputVolume = inputFixedVolume #Temporary statement. Just to make the code run initially as now I am not writing the process to generate the output volume. Need to be removed otherwise it doesnt make sense
        
        if not inputFixedVolume or not inputMovingVolume or not outputVolume:
            raise ValueError("Input or output volume is missing or invalid")

        startTime = time.time()
        logging.info('Processing started')
        
        # Start execution in the background
        
        self.f_n4itkbiasfieldcorrection(inputMovingVolume, outputVolume)
        #self.f_segmentationMask(self, inputFixedVolume, inputMovingVolume)
        #self.f_registrationBrainsFit(self)
        
        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        
    def f_n4itkbiasfieldcorrection(self, inputVolumeNode, outputVolumeNode):
        
        # Set parameters
        parameters = {}
        parameters["inputImageName"] = inputVolumeNode
        parameters["outputImageName"] = outputVolumeNode
        N4BiasFilter = slicer.modules.n4itkbiasfieldcorrection
        cliNode = slicer.cli.runSync(N4BiasFilter, None, parameters)
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
          # success
          
    def f_segmentationMask(self, inputFixedVolume, inputMovingVolume):
        
        # Set parameters
        parameters = {}
        parameters["inputImageName"] = inputVolumeNode
        parameters["outputImageName"] = outputVolumeNode
        N4BiasFilter = slicer.modules.n4itkbiasfieldcorrection
        cliNode = slicer.cli.runSync(N4BiasFilter, None, parameters)
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
          # success
          
    def f_registrationBrainsFit(self, inputFixedVolume, inputMovingVolume):
        
        # Set parameters
        
        pNode = self.parameterNode()
        fixedVolumeID = pNode.GetParameter(inputFixedVolume.getID())
        movingVolumeID = pNode.GetParameter(inputMovingVolume.getID())
        self.__movingTransform = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(self.__movingTransform)

        parameters = {}
        parameters["fixedVolume"] = fixedVolumeID
        parameters["movingVolume"] = movingVolumeID
        parameters["initializeTransformMode"] = "useMomentsAlign"
        parameters["useRigid"] = True
        parameters["useScaleVersor3D"] = True
        parameters["useScaleSkewVersor3D"] = True
        parameters["useAffine"] = True
        parameters["linearTransform"] = self.__movingTransform.GetID()

        self.__cliNode = None
        self.__cliNode = slicer.cli.run(slicer.modules.brainsfit, self.__cliNode, parameters)

        self.__cliObserverTag = self.__cliNode.AddObserver('ModifiedEvent', self.processRegistrationCompletion)
        self.__registrationStatus.setText('Wait ...')
        self.__registrationButton.setEnabled(0)
        
        

class MRICTRegistrationCryoTest(ScriptedLoadableModuleTest):
    """
    This is the test case for the scripted module MRICTRegistrationCryo.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MRICTRegistration()

    def test_MRICTRegistration(self):
        """
        Should test the algorithm with test dataset
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        # import SampleData
        # registerSampleData()
        # inputVolume = SampleData.downloadSample('CryoAblation1')
        # self.delayDisplay('Loaded test data set')

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

        # Test the module logic
        # logic = MRICTRegistrationCryoLogic()

        # Test algorithm
        # logic.process(self, inputFixedVolume, inputMovingVolume, outputVolume)

        self.delayDisplay('Test passed')

