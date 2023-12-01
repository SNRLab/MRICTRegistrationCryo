"""
MRI-CT Registration for Cryoablation: Automated Registration and Analysis

This project is a comprehensive toolset for performing automated MRI-CT registration specifically tailored for Cryoablation procedures. The codebase includes a collection of modules, transformations, and utilities designed to streamline the registration process, facilitate analysis, and enhance the visualization of MRI and CT images in the context of Cryoablation procedures.

The code is organized into several main components:

1. Module Classes:
   - MRICTRegistrationCryo: This class represents the main module for MRI-CT registration specifically designed for Cryoablation procedures. It incorporates functionalities for handling MRI and CT data, performing registration, and integrating the registration process seamlessly into the Slicer environment.

   - MRICTRegistrationCryoWidget: A class representing the widget used in the Slicer interface for user interaction and configuration of the MRI-CT registration process. It sets up the graphical user interface elements, handles user inputs, and initiates the registration process.

   - PythonDependencyChecker: A utility class responsible for checking and installing Python package dependencies required for the functionality of the MRI-CT registration module. It performs checks on essential libraries like MONAI, ITK, PyTorch, and others.

   - SlicerLoadImage and Normalized: Transform classes utilized for adapting Slicer VolumeNodes to MONAI volumes and normalizing input volumes, respectively. These classes provide essential transformations necessary for processing and standardizing image data.

2. Functionality:
   - MRI-CT Registration: The core functionality revolves around performing registration between MRI and CT images. It enables alignment and fusion of MRI and CT images to facilitate Cryoablation planning and analysis.

   - Dependency Management: The project includes mechanisms for handling Python package dependencies required for the seamless execution of the registration module.

   - Preprocessing: The Normalized transformation normalizes input volumes within the specified keys, ensuring uniform intensity ranges across the data, a crucial preprocessing step in imaging tasks.

3. Usage:
   - The project is intended to be integrated into the 3D Slicer environment as a plugin/module, providing an intuitive and interactive interface for users to perform MRI-CT registration specifically tailored for Cryoablation procedures.

4. Collaboration and Acknowledgments:
   - Developed by Subhra Sundar Goswami, Junichi Takuda, and Nobuhiko Hata as part of a collaborative effort

5. License:None

"""

import os
import os.path
import unittest
import gc
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import slicer.modules
from sys import platform
import logging
import time
import numpy as np
import torch

import monai
from monai.inferers.utils import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets.unet import UNet
from monai.transforms import (AddChanneld, Compose, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord, Resized,
                              Resize, CropForegroundd, ScaleIntensityRange)
from monai.transforms.compose import MapTransform
from monai.transforms.post.array import AsDiscrete, KeepLargestConnectedComponent

# Logging is not working. Where to define this logging?
#logfilename = os.path.join(os.path.dirname(self.parent.path),"logfile.txt")
logging.basicConfig(filename="logfile.txt", encoding='utf-8', filemode='w', level=logging.DEBUG, format="%(name)s → %(levelname)s: %(message)s")


class Normalized(MapTransform):
  """
    Normalizes input volumes.

    This class serves as a transformation to normalize input volumes, ensuring consistency and uniformity
    in the intensity range of specified keys within the input data. It inherits from the MONAI `MapTransform`,
    enabling integration into various MONAI workflows for pre-processing and standardization of input volumes.

    Attributes:
    - 'keys': A list of keys identifying the volumes to be normalized within the input data.
    - 'meta_key_postfix': String postfix used to construct keys for metadata storage within the transformed data.

    Methods:
    - '__init__(self, keys, meta_key_postfix: str = "meta_dict") -> None': Initializes the normalization
      instance with the provided keys and a postfix for metadata keys.
    - '__call__(self, volume_node)': Executes the normalization process on the specified keys within
      the given volume node. It retrieves the data, iterates through the specified keys, and applies intensity
      range scaling using `ScaleIntensityRange` from MONAI, ensuring the intensity values fall within the
      desired range (0.0 to 1.0 by default). The normalized volumes are then returned as a dictionary,
      preserving the original keys and their normalized counterparts.
  """

  def __init__(self, keys, meta_key_postfix: str = "meta_dict") -> None:
    super().__init__(keys)
    self.meta_key_postfix = meta_key_postfix
    self.keys = keys

  def __call__(self, volume_node):
    # Normalizes input volumes within specified keys
    d = dict(volume_node)
    for key in self.keys:
      d[key] = ScaleIntensityRange(a_max=np.amax(d[key]), a_min=np.amin(d[key]), b_max=1.0, b_min=0.0, clip=True)(
        d[key])
    return d


class SlicerLoadImage(MapTransform):
  """
    Adapter from Slicer VolumeNode to MONAI volumes.

    This class acts as an adapter, converting Slicer VolumeNodes into MONAI volumes for seamless integration
    and compatibility within the MONAI framework. It extends the functionality of the MONAI `MapTransform`
    to handle the translation and extraction of essential information from Slicer VolumeNodes into a format
    suitable for MONAI workflows.

    Attributes:
    - 'keys': Keys used to extract and store data within the MONAI dictionary format.
    - 'meta_key_postfix': String postfix used to construct keys for metadata storage within the MONAI dictionary.

    Methods:
    - '__init__(self, keys, meta_key_postfix: str = "meta_dict") -> None': Initializes the adapter instance
      with provided keys for data storage and a postfix for metadata keys within the MONAI dictionary.
    - '__call__(self, volume_node)': Converts the given Slicer VolumeNode into a MONAI-compatible format.
      It extracts the volume data, swaps axes to align with MONAI conventions, gathers spatial information,
      and constructs metadata including spatial shape, affine transformation, and original spacing. Finally,
      it returns a dictionary containing the transformed volume data and associated metadata in the MONAI format.
  """

  def __init__(self, keys, meta_key_postfix: str = "meta_dict") -> None:
    super().__init__(keys)
    self.meta_key_postfix = meta_key_postfix

  def __call__(self, volume_node):
    # Converts Slicer VolumeNodes to MONAI volumes
    data = slicer.util.arrayFromVolume(volume_node)
    data = np.swapaxes(data, 0, 2)
    
    # Display volume information
    print("Load volume from Slicer : {}Mb\tshape {}\tdtype {}".format(data.nbytes * 0.000001, data.shape, data.dtype))
    
    # Extract spatial information
    spatial_shape = data.shape
    m = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m)
    affine = slicer.util.arrayFromVTKMatrix(m)
    
    # Gather metadata
    meta_data = {
        "affine": affine,
        "original_affine": affine,
        "spacial_shape": spatial_shape,
        "original_spacing": volume_node.GetSpacing()
    }

    return {
        self.keys[0]: data,
        '{}_{}'.format(self.keys[0], self.meta_key_postfix): meta_data
    }


class PythonDependencyChecker(object):
    """
    Class responsible for handling Python package dependencies.

    This class manages the verification and installation of required Python packages essential for the proper
    functionality of the associated module. It contains methods to check if the necessary dependencies are
    satisfied and, if needed, installs the missing packages.

    Methods:
    - 'areDependenciesSatisfied()': Verifies whether the required dependencies are already installed within
      the current environment. It checks for specific package versions critical for the module's functionality.
      Returns True if all dependencies are satisfied, otherwise returns False.
    - 'installDependenciesIfNeeded(progressDialog=None)': Installs required dependencies if they are not
      already satisfied in the current environment. It begins by checking the status of dependencies; if they
      are missing, it proceeds to install the necessary packages. The installation process involves handling
      PyTorch installation through the PyTorch Slicer extension or, if not available, via PIP. Additionally,
      it installs other essential packages such as ITK, NiBabel, scikit-image, gdown, and MONAI within the
      specified version range (0.6.0 < MONAI <= 0.9.0).
    """

    @classmethod
    def areDependenciesSatisfied(cls):
        """
        Checks if the required dependencies are satisfied.

        Returns:
            bool: True if dependencies are satisfied, False otherwise.
        """
        try:
            # Check for required package versions
            from packaging import version
            import monai
            import itk
            import torch
            import skimage
            import gdown
            import nibabel

            # Ensure MONAI version falls within the specified range
            return version.parse("0.6.0") < version.parse(monai.__version__) <= version.parse("0.9.0")
        except ImportError:
            return False

    @classmethod
    def installDependenciesIfNeeded(cls, progressDialog=None):
        """
        Installs required dependencies if not already satisfied.

        Args:
            progressDialog (object, optional): Progress dialog for installation. Defaults to None.
        """
        if cls.areDependenciesSatisfied():
            return

        progressDialog = progressDialog or slicer.util.createProgressDialog(maximum=0)
        progressDialog.labelText = "Installing PyTorch"

        try:
            # Try to install the best available pytorch version for the environment using the PyTorch Slicer extension
            import PyTorchUtils
            PyTorchUtils.PyTorchUtilsLogic().installTorch()
        except ImportError:
            # Fallback on default torch available on PIP
            slicer.util.pip_install("torch")
        
        # Install other required dependencies
        dependencies = ["itk", "nibabel", "scikit-image", "gdown", "monai>0.6.0,<=0.9.0"]
        for dep in dependencies:
            progressDialog.labelText = dep
            slicer.util.pip_install(dep)


class MRICTRegistrationCryo(ScriptedLoadableModule):
    """
    This class represents the MRI-CT Registration module for CryoAblation in 3D Slicer.

    It utilizes the base class ScriptedLoadableModule available in Slicer to create a module
    for MRI-CT registration within the CryoAblation context.

    Initialization:
    - Initializes the module with essential metadata, including title, category, contributors,
      help text, and acknowledgments.
    - Sets up module-specific icon(s) for visual representation in the Slicer user interface.
 
    Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        # Initialization using the parent class
        ScriptedLoadableModule.__init__(self, parent)
        
        # Module metadata and information
        self.parent.title = "MRICTRegistrationCryo"  
        self.parent.categories = ["Registration"]  
        self.parent.dependencies = []  
        self.parent.contributors = ["Subhra Sundar Goswami, Junichi Takuda", "Nobuhiko Hata"]
        self.parent.helpText = "MRI-CT registration for CryoAblation" # Module help text
        self.parent.helpText += self.getDefaultModuleDocumentationLink() # Need to add module documentation link
        self.parent.acknowledgementText = "" # Acknowledgement text

        # Additional initialization step after application startup is complete
        moduleDir = os.path.dirname(self.parent.path)
        for iconExtension in ['.svg', '.png']:
            iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
            if os.path.isfile(iconPath):
                parent.icon = qt.QIcon(iconPath) # Set module icon
                break # Use the first valid icon found
        

class MRICTRegistrationCryoWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Widget representing the graphical interface for the MRI-CT Registration module within the CryoAblation context.

    This class serves as the user interface for the MRI-CT Registration module in the context of CryoAblation. It provides
    functionalities for setting up input volumes, defining output volumes, managing device options, configuring ROI (Region
    of Interest) selectors for CT and MRI, and executing the registration process.

    The class inherits from ScriptedLoadableModuleWidget and VTKObservationMixin to manage the graphical user interface
    components and handle observations related to parameter nodes.

    Functionality:
    - Manages initialization of the widget and its attributes, including observations on the MRML scene and parameters.
    - Provides setup of the UI elements, including input and output volume selectors, device options, ROI selectors,
      Apply button, and a status label for displaying processing logs.
    - Maintains synchronization between the GUI and the associated parameter node, ensuring changes made in the GUI are
      reflected in the MRML scene, and vice versa.
    - Implements the registration process upon user interaction, handling exceptions, and logging error messages.
    - Ensures compatibility checks for Slicer versions and verifies dependencies needed by the module.
    - Facilitates downloads and installations of necessary dependencies (Slicer extensions and PIP packages) through
      the 'downloadDependenciesAndRestart()' method, including warnings for failed downloads and restart of Slicer
      upon successful installation.
    """
    
    enableReloadOnSceneClear = True
    
    def __init__(self, parent=None):
        """
        Initialization of the widget and necessary attributes.
        """
        # Initialization of the widget
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # Needed for parameter node observation
        self.addObserver(slicer.mrmlScene, None, None)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        
        # Widget attributes initialization
        self.addObserver(slicer.mrmlScene, None, None)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.device = None
        self.modality = None
        self.clippedMasterImageData = None
        self.lastRoiNodeId = ""
        self.lastRoiNodeModifiedTime = 0
        self.roiSelector = slicer.qMRMLNodeComboBox()
        
        
    @staticmethod
    def areDependenciesSatisfied():
        """
        Checks if required dependencies are satisfied.
        """
        try:
            import SegmentEditorLocalThresholdLib
        except ImportError:
            return False

        return PythonDependencyChecker.areDependenciesSatisfied()

    @staticmethod
    def downloadDependenciesAndRestart():
        """
        Downloads and installs necessary dependencies for the module.

        This function is responsible for downloading and installing Slicer extensions and PIP dependencies
        required by the module. It initiates a process that checks and installs the following Slicer
        extensions: SlicerVMTK, MarkupsToModel, SegmentEditorExtraEffects, and PyTorch.

        The method employs the Slicer Extensions Manager to retrieve metadata and download extensions
        based on the Slicer version. For Slicer versions prior to 5.0.3, extensions are downloaded using
        metadata, while for newer versions, direct extension download is supported.

        Additionally, it checks and installs PIP dependencies using the PythonDependencyChecker class,
        which ensures necessary Python packages are installed for the module's functionality.

        Upon completion of the download and installation process, the method closes the progress dialog and
        checks if any extensions failed to download. If any extensions failed, it warns the user about the
        failed downloads and suggests manual installation through Slicer's extension manager. If all
        dependencies are successfully installed, it triggers a restart of the Slicer application to apply
        the changes.
        """
        
        progressDialog = slicer.util.createProgressDialog(maximum=0)
        extensionManager = slicer.app.extensionsManagerModel()

        def downloadWithMetaData(extName):
            # Method for downloading extensions prior to Slicer 5.0.3
            meta_data = extensionManager.retrieveExtensionMetadataByName(extName)
            if meta_data:
                return extensionManager.downloadAndInstallExtension(meta_data["extension_id"])

        def downloadWithName(extName):
            # Direct extension download since Slicer 5.0.3
            return extensionManager.downloadAndInstallExtensionByName(extName)

        # Install Slicer extensions
        downloadF = downloadWithName if hasattr(extensionManager,
                                            "downloadAndInstallExtensionByName") else downloadWithMetaData

        slicerExtensions = ["SlicerVMTK", "MarkupsToModel", "SegmentEditorExtraEffects", "PyTorch"]
        for slicerExt in slicerExtensions:
            progressDialog.labelText = f"Installing the {slicerExt}\nSlicer extension"
            downloadF(slicerExt)

        # Install PIP dependencies
        PythonDependencyChecker.installDependenciesIfNeeded(progressDialog)
        progressDialog.close()

        # Restart if no extension failed to download. Otherwise warn the user about the failure.
        failedDownload = [slicerExt for slicerExt in slicerExtensions if
                      not extensionManager.isExtensionInstalled(slicerExt)]

        if failedDownload:
            failed_ext_list = "\n".join(failedDownload)
            warning_msg = f"The download process failed install the following extensions : {failed_ext_list}" \
                    f"\n\nPlease try to manually install them using Slicer's extension manager"
            qt.QMessageBox.warning(None, "Failed to download extensions", warning_msg)
        else:
            slicer.app.restart()


    def setup(self):
        """
        Sets up the user interface elements and configurations.

        The method initializes various components of the UI, handles Slicer version compatibility checks,
        verifies dependencies, and organizes the input and output volume selectors. It sets up collapsible
        buttons for input, output, and advanced settings sections.

        Additionally, it configures UI elements like volume selectors, device options, ROI selectors for
        both CT and MRI, Apply button, and a status label for displaying processing logs.

        Connections are established to ensure that changes made in the GUI are reflected in the MRML scene.
        For example, the changes made in the input and output volume selectors update the parameter node
        associated with the module.

        The Apply button's click event is connected to the 'onApplyButton' method to initiate the processing
        when triggered.

        Lastly, the method initializes the parameter node to ensure proper module functionality.
        """
        
        # Set up the UI by calling the superclass method
        ScriptedLoadableModuleWidget.setup(self)
        # Set initial registration status as False
        self.registrationInProgress = False
        
        # Observers to update parameter node when the scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Verify Slicer version compatibility
        if not (slicer.app.majorVersion, slicer.app.minorVersion, float(slicer.app.revision)) >= (4, 11, 29738):
            # Display error message if the Slicer version is incompatible
            error_msg = "The RVesselX plugin is only compatible from Slicer 4.11 2021.02.26 onwards.\n" \
                      "Please download the latest Slicer version to use this plugin."
            self.layout.addWidget(qt.QLabel(error_msg))
            self.layout.addStretch()
            slicer.util.errorDisplay(error_msg)
            return

        if not self.areDependenciesSatisfied():
            # Display error message if required dependencies are not satisfied
            error_msg = "Slicer VMTK, MarkupsToModel, SegmentEditorExtraEffects and MONAI are required by this plugin.\n" \
                      "Please click on the Download button to download and install these dependencies."
            self.layout.addWidget(qt.QLabel(error_msg))
            downloadDependenciesButton = createButton("Download dependencies and restart", self.downloadDependenciesAndRestart)
            self.layout.addWidget(downloadDependenciesButton)
            self.layout.addStretch()
            return
        
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
        
        # UI for input volume selection
        IOLayout = qt.QFormLayout(inputCollapsibleButton)
        # Selector for fixed input volume
        self.inputFixedVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputFixedVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputFixedVolumeSelector.selectNodeUponCreation = False
        self.inputFixedVolumeSelector.noneEnabled = False
        self.inputFixedVolumeSelector.addEnabled = False
        self.inputFixedVolumeSelector.removeEnabled = True
        self.inputFixedVolumeSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Fixed Volume: ", self.inputFixedVolumeSelector)
        
        # Selector for moving input volume
        self.inputMovingVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputMovingVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputMovingVolumeSelector.selectNodeUponCreation = False
        self.inputMovingVolumeSelector.noneEnabled = False
        self.inputMovingVolumeSelector.addEnabled = False
        self.inputMovingVolumeSelector.removeEnabled = True
        self.inputMovingVolumeSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Moving Volume: ", self.inputMovingVolumeSelector)
        
        # UI organization - Advanced collapsible button
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

        # Advanced Area
        advancedCollapsibleButton = ctk.ctkCollapsibleButton()
        advancedCollapsibleButton.text = "Advanced"
        advancedCollapsibleButton.collapsed = 0
        self.layout.addWidget(advancedCollapsibleButton)

        # UI for advanced settings (device selection, ROI options)
        advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
        
        self.deviceSelector = qt.QComboBox()
        self.deviceSelector.addItems(["cuda", "cpu"])
        advancedFormLayout.addRow("Device:", self.deviceSelector)
        
        ## Add ROI options for CT
        self.roiSelectorCT = slicer.qMRMLNodeComboBox()
        self.roiSelectorCT.nodeTypes = ['vtkMRMLMarkupsROINode']
        self.roiSelectorCT.noneEnabled = True
        self.roiSelectorCT.setMRMLScene(slicer.mrmlScene)
        advancedFormLayout.addRow("ROI CT: ", self.roiSelectorCT)
        
        ## Add ROI options for MRI
        self.roiSelectorMRI = slicer.qMRMLNodeComboBox()
        self.roiSelectorMRI.nodeTypes = ['vtkMRMLMarkupsROINode']
        self.roiSelectorMRI.noneEnabled = True
        self.roiSelectorMRI.setMRMLScene(slicer.mrmlScene)
        advancedFormLayout.addRow("ROI MRI: ", self.roiSelectorMRI)
        
        # Apply Button configuration
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Start registration."
        self.applyButton.enabled = True #Should be False but changed to True for testing
        self.layout.addWidget(self.applyButton)
        
        # Status label configuration for displaying processing logs
        self.statusLabel = qt.QPlainTextEdit()
        self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self.statusLabel.setCenterOnScroll(True)
        self.layout.addWidget(self.statusLabel)
        
        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.inputFixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.inputMovingVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.outputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.roiSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        
        # Connect Apply button click to perform the processing
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        
        # Initialize the parameter node. Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        # Removes observers and cleans up resources
        self.removeObservers()
        
    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Ensures parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)...Removes observer for parameter node changes
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

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
        # Sets and observes the parameter node
        self.setParameterNode(self.logic.getParameterNode())

        
    def setParameterNode(self, inputParameterNode):
        """
        Sets and observes the parameter node.
        Observation is essential as changes in the parameter node trigger immediate GUI updates.
        This method establishes the association between the provided parameter node and the widget's behavior.

        If an input parameter node is provided, it sets default parameters using the logic associated with the node.

        The function manages the observation of the parameter node to keep track of changes. It unobserves
        any previously selected parameter node to remove any existing observation and adds an observer
        to the newly selected node. This observation mechanism ensures that any alterations to parameters,
        either through scripts or other modules, are promptly reflected in the GUI.

        After setting up observation, it initiates an initial update of the GUI based on the parameter node
        by calling 'updateGUIFromParameterNode()'.
        """

        # Set default parameters if the input parameter node exists
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update based on the parameter node
        self.updateGUIFromParameterNode()

        
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        Updates the module's graphical user interface based on changes to the parameter node.

        Whenever the parameter node is modified, this method is triggered to synchronize the module's GUI
        with the current state of the parameter node.

        It starts by checking if the parameter node or GUI update flag is not set, allowing an early return
        if either condition is met to prevent unnecessary updates.

        Upon invocation, this method updates various GUI elements such as node selectors and buttons based
        on the parameter node's current state. It retrieves node references from the parameter node for
        input (fixed and moving) volumes and the output volume, then sets the corresponding current nodes
        in the GUI node selectors accordingly.

        The method further manages the states and tooltips of buttons based on the presence of selected
        volumes. When both fixed and moving input volumes are selected, it enables the 'Apply' button
        with a tooltip indicating the ability to compute the output volume. Conversely, if either input
        volume is not selected, it disables the 'Apply' button and provides a tooltip prompting users to
        select input and output volume nodes.

        """
        
        # Checking if the parameter node or GUI update flag is not set, then returns early
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (Setting the GUI update flag to prevent recursive calls)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders based on the current parameter node state
        self.inputFixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputFixedVolume"))
        self.inputMovingVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputMovingVolume"))
        self.outputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        self.roiSelectorCT.setCurrentNode(self._parameterNode.GetNodeReference("roiSelectorCT"))
        
        # Update buttons states and tooltips based on selected volumes
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
        
    # Similar update for GUI to Parameter node
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        Handles updates to the associated parameter node when the user interacts with the graphical user interface (GUI).
        
        This method captures any modifications made by the user within the GUI and saves these changes into
        the corresponding parameter node. By doing so, it ensures that alterations, such as selected input and
        output volumes, are retained and persist across sessions when the scene is saved and loaded again.

        The method begins by checking if the parameter node or GUI update flag is not set, allowing an early
        return to avoid unnecessary updates.

        Upon invocation, it initiates the modification of the parameter node to bundle all property changes
        into a single batch. Then, it updates the parameter node based on the user's selections in the GUI,
        specifically setting node references for the input fixed, input moving, and output volumes.

        Once the GUI changes are reflected in the parameter node, the method finalizes the modification,
        ensuring the changes are appropriately saved and handled for scene saving and loading.
        """
        # Checking if the parameter node or GUI update flag is not set, then returns early
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        
        # Start modifying the parameter node
        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # Update the parameter node based on GUI changes
        self._parameterNode.SetNodeReferenceID("InputFixedVolume", self.inputFixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputMovingVolume", self.inputMovingVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.outputVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("roiSelectorCT", self.roiSelectorCT.currentNodeID)
        # End modification of the parameter node
        self._parameterNode.EndModify(wasModified)
    
    def onSelect(self):
        """
        Executes actions when something is selected (might be incomplete or commented out).
        """
        #self.applyButton.enabled = self.inputFixedVolumeSelector.currentNode() and self.inputMovingVolumeSelector.currentNode() """and self.outputVolumeSelector.currentNode()"""

        if not self.registrationInProgress:
            self.applyButton.text = "Apply"
            return
        self.updateBrowsers()
        
    def onApplyButton(self):
        """
        Run processing when the user clicks the "Apply" button.
        If a registration process is in progress:
            - Cancels the ongoing process if requested by the user
        Otherwise:
            - Initiates the registration process using input and output volumes
        If the registration process encounters an exception:
            - Logs the error message in the application
            - Restores the default cursor state and button settings after handling the exception
        """
        
        if self.registrationInProgress:
            # Cancels the ongoing process if requested by the user
            self.registrationInProgress = False
            self.abortRequested = True
            raise ValueError("User requested cancel.")
            self.cliNode.Cancel() # not implemented
            self.applyButton.text = "Cancelling..."
            self.applyButton.enabled = False   #Should be False but changed to True for testing
            return

        self.registrationInProgress = True
        self.applyButton.text = "Cancel"
        self.statusLabel.plainText = ''
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        
        try:
            # Executes the registration process using input and output volumes
            self.logic.process(
                self.inputFixedVolumeSelector.currentNode(),
                self.inputMovingVolumeSelector.currentNode(),
                self.outputVolumeSelector.currentNode(),
                self.roiSelectorCT.currentNode())
            time.sleep(3) # Adds a delay for testing
            
        except Exception as e:
            # Handles exceptions during the registration process
            print(e)
            self.addLog("Error: {0}".format(str(e)))
            import traceback
            traceback.print_exc()
          
        finally:
            # Resets cursor and applies changes after process completion or cancellation
            slicer.app.restoreOverrideCursor()
            self.registrationInProgress = False
            self.onSelect() # restores default Apply button state
            
    def addLog(self, text):
        """
        Append text to the log window
    
        This function adds the given text to the log window within the application interface.
        It updates the log window to reflect the appended text immediately.
        """
        self.statusLabel.appendPlainText(text)
        slicer.app.processEvents()  # force update of the application events


class MRICTRegistrationCryoLogic(ScriptedLoadableModuleLogic, unittest.TestCase):
    """
    Handles the logic for MRI-CT registration within the CryoAblation module.
    
    This class encapsulates the main processing logic for MRI-CT registration. It includes methods
    for setting default parameters, performing image processing steps like bias field correction
    and segmentation, as well as running registration algorithms. Bias correction functions uses
    Slicer 3D module N4ITKBiasFieldCorrection described at
    http://viewvc.slicer.org/viewcvs.cgi/trunk/Applications/CLI/N4ITKBiasFieldCorrection. Additionally,
    it implements functions for creating and utilizing UNet models for liver segmentation based on the selected
    modality using Slicer 3D module R-Vessel-X described at https://github.com/R-Vessel-X/SlicerRVXLiverSegmentation.
    The registration functions are based on Slicer 3D module BrainsFit described at
    https://github.com/BRAINSia/BRAINSTools/tree/main/BRAINSFit
    
    Key Methods:
    - setDefaultParameters: Sets default parameters for processing.
    - process: Executes the processing algorithm for registration.
    - f_n4itkbiasfieldcorrection: Performs bias field correction using the N4 ITK module.
    - f_segmentationMask: Segments the liver using AI-based segmentation
    - f_registrationBrainsFit: Perform registration with or without masks
    - createUNetModel: Creates a UNet model based on the selected device.
    - getPreprocessingTransform & getPostProcessingTransform: Handles pre-processing and post-processing transformations.
    - launchLiverSegmentation: Initiates liver segmentation using a UNet model.
    """

    enableReloadOnSceneClear = True
    
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Set default parameters for the processing.
        """
        A = 100
    
    def process(self, inputFixedVolume, MovingVolume, outputVolume, roiCT):
        """
        Run the processing algorithm.
        """
        if not inputFixedVolume or not MovingVolume or not outputVolume:
            raise ValueError("Input or output volume is missing or invalid")
        
        startTime = time.time()
        logging.info('Processing started')
        
        # Pre-processing steps
                
        clippedCTImageData = None
        lastRoiNodeId = ""
        lastRoiNodeModifiedTime = 0
        
        # Segment the liver from CT using AI based segmentation module RVX
        inputFixedVolumeMask = slicer.vtkMRMLLabelMapVolumeNode()
        inputFixedVolumeMask.SetName('inputFixedVolumeMask')
        slicer.mrmlScene.AddNode(inputFixedVolumeMask)
        self.f_segmentationMask(inputFixedVolume, inputFixedVolumeMask, "cpu", "CT")
        
        #clone the original moving volume otherwise the transform will be automatically applied on it
        inputMovingVolume = slicer.modules.volumes.logic().CloneVolume(MovingVolume, "")
        
        #Correct bias using N4 filter #We don't need this step before segmentation. But at this point the segmentation algorithm is not working without the bias correction. It seems it is because of the data type of MRI volume which is "unsigned short" whereas the after correction it is int
        movingVolumeN4 = slicer.vtkMRMLScalarVolumeNode()
        movingVolumeN4.SetName('movingVolumeN4')
        slicer.mrmlScene.AddNode(movingVolumeN4)
        self.f_n4itkbiasfieldcorrection(inputMovingVolume, movingVolumeN4, None, [1,1,1])
        movingVolumeN4.SetAndObserveTransformNodeID(None)
        
        #Segment the liver from MRI using AI based segmentation module RVX
        inputMovingVolumeMask = slicer.vtkMRMLLabelMapVolumeNode()
        inputMovingVolumeMask.SetName('inputMovingVolumeMask')
        slicer.mrmlScene.AddNode(inputMovingVolumeMask)
        self.f_segmentationMask(movingVolumeN4, inputMovingVolumeMask, "cpu", "MRI")
        
        #inputFixedVolume = self.removeBedFromCT(inputFixedVolume, roiCT)
        
        regAffineVolume = slicer.vtkMRMLScalarVolumeNode()
        regAffineVolume.SetName('regAffineVolume')
        slicer.mrmlScene.AddNode(regAffineVolume)
        self.f_registrationAffine(inputFixedVolume, inputMovingVolume, regAffineVolume)
        inputMovingVolume.SetAndObserveTransformNodeID(None)
        
        maskProcessingMode = "ROI" #Specifies a mask to only consider a certain image region for the registration.  If ROIAUTO is chosen, then the mask is computed using Otsu thresholding and hole filling. If ROI is chosen then the mask has to be specified as in input.
        
        self.f_registrationBrainsFit(inputFixedVolume, movingVolumeN4, outputVolume, maskProcessingMode, inputFixedVolumeMask, inputMovingVolumeMask)
        movingVolumeN4.SetAndObserveTransformNodeID(None)
        
        print("returned to Process")
        
        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        
    
    def f_n4itkbiasfieldcorrection(self, inputVolumeNode, outputVolNode, inputMask, initMeshResolution):
        """
        Performs bias field correction using the N4 ITK module.
        Args:
        - inputVolumeNode: Input volume node to perform bias field correction.
        Returns:
        - outputVolumeNode: Output volume node after bias field correction.
        """
        
        # Set parameters for the N4 ITK bias field correction
        parameters = {
            "inputImageName": inputVolumeNode,
            "outputImageName": outputVolNode,
            "maskImageName": inputMask,
            "initialMeshResolution":initMeshResolution
        }
        
        # Run the N4 ITK bias field correction CLI
        N4BiasFilter = slicer.modules.n4itkbiasfieldcorrection
        cliNode = slicer.cli.runSync(N4BiasFilter, None, parameters)
        
        # Check CLI execution status
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error occurred during CLI executio
            errorText = cliNode.GetErrorText()
            #slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("N4BiasFilter CLI execution failed: " + errorText)
    
    
    def f_segmentationMask(self, inputVolumeNode, maskLabelMapNode, use_cudaOrCpu, modalityV):
        """
        Segments liver using AI-based segmentation based on the selected modality.

        Args:
        - inputVolumeNode: Input volume node for segmentation.
        - outputVolumeNode: Output volume node for the segmented mask.
        - use_cudaOrCpu: Selected device for segmentation ('cuda' or 'cpu').
        - modalityV: Modality type for segmentation ('CT' or 'MRI').
        """
        
        #Have to implement it to have the processing only on the ROI selected
        #slicer.vtkSlicerSegmentationsModuleLogic.CopyOrientedImageDataToVolumeNode(self.getClippedMasterImageData(), inputVolumeNode)
        
        try:
            outputSegmentationScalarVolumeNode = slicer.vtkMRMLScalarVolumeNode()
            slicer.mrmlScene.AddNode(outputSegmentationScalarVolumeNode)
            self.launchLiverSegmentation(inputVolumeNode, outputSegmentationScalarVolumeNode, use_cudaOrCpu, modalityV)
            
            # Have to convert outputSegmentationScalarVolumeNode to vtkMRMLLabelMapVolumeNode
            
            # Get the vtkMRMLScalarVolumeNode data that we want to convert to a labelmap:
            segmentationVolume_data = slicer.util.arrayFromVolume(outputSegmentationScalarVolumeNode)
            segmentationVolume_data = segmentationVolume_data.astype(np.uint8())
            
            # Use the maskLabelMapNode which is of type vtkMRMLLabelMapVolumeNode:
            slicer.util.updateVolumeFromArray(maskLabelMapNode, segmentationVolume_data)
        
            #The vtkMRMLLabelMapVolumeNode will probably have a different IJKToRAS matrix (and origin), so we need to update them with the ones from the original vtkMRMLScalarVolumeNode
            slicer.modules.volumes.logic().CreateLabelVolumeFromVolume(slicer.mrmlScene, maskLabelMapNode, outputSegmentationScalarVolumeNode)
            """
            volume_matrix = vtk.vtkMatrix4x4()
            outputSegmentationScalarVolumeNode.GetIJKToRASMatrix(volume_matrix)
            volume_origin = outputSegmentationScalarVolumeNode.GetOrigin()
            maskLabelMapNode.SetIJKToRASMatrix(volume_matrix)
            maskLabelMapNode.SetOrigin(volume_origin)
            """
            
        except Exception as e:
            qt.QApplication.restoreOverrideCursor()
            slicer.util.errorDisplay(str(e))

        finally:
            qt.QApplication.restoreOverrideCursor()
    
  
    def removeBedFromCT(self, inputFixedVolume, roiNode):
        """
        Crops the CT volume node if a ROI Node is selected in the parameter comboBox. Otherwise returns the full extent of the volume.
        """
        # Return CT volume unchanged if there is no ROI
        
        
        if roiNode is None:
            self.clippedCTImageData = None
            self.lastRoiNodeId = ""
            self.lastRoiNodeModifiedTime = 0
            return inputFixedVolume

        # Compute clipped CT image. Return last clipped image data if there was no change
        if (self.clippedCTImageData is not None and
                roiNode.GetID() == self.lastRoiNodeId and
                roiNode.GetMTime() == self.lastRoiNodeModifiedTime):
          
            # Use cached clipped CT image data
            return self.clippedCTImageData

        # Compute clipped CT image using the SegmentEditorLocalThresholdLib
        import SegmentEditorLocalThresholdLib
        self.clippedCTImageData = SegmentEditorLocalThresholdLib.SegmentEditorEffect.cropOrientedImage(ctImageData, roiNode)
        
        self.lastRoiNodeId = roiNode.GetID()
        self.lastRoiNodeModifiedTime = roiNode.GetMTime()
        return self.clippedCTImageData
        
    def f_registrationAffine(self, inputFixedVolume, inputMovingVolume, outputVolume):
        """
        Perform registration using BrainsFit
        """
        # Set parameters
        fixedVolumeID = inputFixedVolume.GetID()
        movingVolumeID = inputMovingVolume.GetID()
        outputVolumeID = outputVolume.GetID()
        
        self.affineTransform = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(self.affineTransform)
                
        parameters = {
            "fixedVolume": fixedVolumeID,
            "movingVolume": movingVolumeID,
            "outputVolume": outputVolumeID,
            "initializeTransformMode": "useMomentsAlign",
            #"useGeometryAlign", "useCenterOfROIAlign", "useMomentsAlign"
            "useRigid": True,
            "useScaleVersor3D": True,
            "useScaleSkewVersor3D": True,
            "useAffine": True,
            "linearTransform": self.affineTransform.GetID()
        }

        print("Affine registration started...")
        
        self.__cliNode = None
        cliNode = slicer.cli.runSync(slicer.modules.brainsfit, self.__cliNode, parameters)

   
   
   
    def f_registrationBrainsFit(self, inputFixedVolume, inputMovingVolume, outputVolume, maskProcessingMode, fixedBinaryVolume, movingBinaryVolume):
        """
        Perform registration using BrainsFit
        """
        # Set parameters
        fixedVolumeID = inputFixedVolume.GetID()
        movingVolumeID = inputMovingVolume.GetID()
        outputVolumeID = outputVolume.GetID()
        fixedBinaryVolumeID = fixedBinaryVolume.GetID()
        movingBinaryVolumeID = movingBinaryVolume.GetID()
        
        self.__movingTransform = slicer.vtkMRMLBSplineTransformNode()
        slicer.mrmlScene.AddNode(self.__movingTransform)
                
        parameters = {
            "fixedVolume": fixedVolumeID,
            "movingVolume": movingVolumeID,
            "outputVolume": outputVolumeID,
            "maskProcessingMode": maskProcessingMode,
            "fixedBinaryVolume": fixedBinaryVolumeID,
            "movingBinaryVolume": movingBinaryVolumeID,
            "initializeTransformMode": "useMomentsAlign",
            #"useGeometryAlign", "useCenterOfROIAlign", "useMomentsAlign"
            "useRigid": True,
            "useScaleVersor3D": True,
            "useScaleSkewVersor3D": True,
            "useAffine": True,
            "useBSpline": True,
            "bsplineTransform": self.__movingTransform.GetID()
        }

        print("Calling slicer.modules.brainsfit")
        
        self.__cliNode = None
        cliNode = slicer.cli.runSync(slicer.modules.brainsfit, self.__cliNode, parameters)
        
        #self.__cliObserverTag = self.__cliNode.AddObserver('ModifiedEvent', self.processRegistrationCompletion)
        
    
    @classmethod
    def createUNetModel(cls, device):
        """
        Create a UNet model based on device.
        """
        return UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                    num_res_units=2, norm=Norm.BATCH, ).to(device)

    @classmethod
    def getPreprocessingTransform(cls, modality):
        """
        Get Preprocessing transform which converts the input volume to MONAI format and resamples and normalizes its inputs. The values in this transform are the same as in the training transform preprocessing.
        """
        if modality == "CT":
            trans = [SlicerLoadImage(keys=["image"]), AddChanneld(keys=["image"]),      Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"), Orientationd(keys=["image"], axcodes="RAS"), ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True), AddChanneld(keys=["image"]), ToTensord(keys=["image"])]
            return Compose(trans)
        elif modality == "MRI":
            trans = [SlicerLoadImage(keys=["image"]), AddChanneld(keys=["image"]), Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"), Orientationd(keys=["image"], axcodes="LPS"), Normalized(keys=["image"]), AddChanneld(keys=["image"]), ToTensord(keys=["image"])]
            return Compose(trans)
    
    @classmethod
    def getPostProcessingTransform(cls, original_spacing, original_size, modality):
        """
        Simple post processing transform to convert the volume back to its original spacing.
        """
        return Compose([
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=original_spacing, mode="nearest"),
            Resized(keys=["image"], spatial_size=original_size)
        ])
        
    @classmethod
    def launchLiverSegmentation(cls, in_volume_node, out_volume_node, use_cuda, modality):
        """
        Launch liver segmentation using UNet model. on the input volume and returns the segmentation in the same volume.
        """
        device = torch.device("cpu") if not use_cuda or not torch.cuda.is_available() else torch.device("cuda:0")
        print("Start liver segmentation using device :", device)
        print(f"Using modality {modality}")
        try:
            with torch.no_grad():
                model_path = os.path.join(os.path.dirname(__file__),
                                      "liver_ct_model.pt" if modality == "CT" else "liver_mri_model.pt")
                print("Model path: ", model_path)
                model = cls.createUNetModel(device=device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model loaded .. ")
                transform_output = cls.getPreprocessingTransform(modality)(in_volume_node)
                print("Transform with MONAI applied .. ")
                model_input = transform_output["image"].to(device)

                roi_size = (160, 160, 160) if modality == "CT" else (240, 240, 96)

                model_output = sliding_window_inference(model_input, roi_size, 4, model, device="cpu", sw_device=device)

                print("Keep largest connected components and threshold UNet output")
                discrete_output = AsDiscrete(argmax=True)(model_output.reshape(model_output.shape[-4:]))
                post_processed = KeepLargestConnectedComponent(applied_labels=[1])(discrete_output)
                output_volume = post_processed.cpu().numpy()[0, :, :, :]
            
                print(transform_output["image"])
                print(transform_output["image"].max())
            
                del post_processed, discrete_output, model_output, model, model_input

                transform_output["image"] = output_volume
                original_spacing = (transform_output["image_meta_dict"]["original_spacing"])
                original_size = (transform_output["image_meta_dict"]["spacial_shape"])
                output_inverse_transform = cls.getPostProcessingTransform(original_spacing, original_size, modality)(transform_output)

                label_map_input = output_inverse_transform["image"][0, :, :, :]
                print("output label map shape is " + str(label_map_input.shape))
                output_affine_matrix = transform_output["image_meta_dict"]["affine"]

                out_volume_node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(output_affine_matrix))
                slicer.util.updateVolumeFromArray(out_volume_node, np.swapaxes(label_map_input, 0, 2))
                del transform_output

        finally:
            # Cleanup any remaining memory
            def del_local(v):
                if v in locals():
                    del locals()[v]

            for n in ["model_input", "model_output", "post_processed", "model", "transform_output"]:
                del_local(n)

            gc.collect()
            torch.cuda.empty_cache()
          
      
class MRICTRegistrationCryoTest(ScriptedLoadableModuleTest):
    """
    Test case for the scripted module MRICTRegistrationCryo.
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MRICTRegistration()

    def test_MRICTRegistration(self):
        """
        Test the algorithm with test dataset
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

