import os
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from sys import platform

import vtkSlicerShapeModuleMRMLPython as vtkSlicerShapeModuleMRML

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
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        
        ScriptedLoadableModuleWidget.setup(self)

        # IO collapsible button
        IOCategory = qt.QWidget()
        self.layout.addWidget(IOCategory)
        IOLayout = qt.QFormLayout(IOCategory)
        self.logic = MRICTRegistrationCryo()
        
        self.inputFixedImageSelector = slicer.qMRMLNodeComboBox()
        self.inputFixedImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputFixedImageSelector.selectNodeUponCreation = False
        self.inputFixedImageSelector.noneEnabled = False
        self.inputFixedImageSelector.addEnabled = False
        self.inputFixedImageSelector.removeEnabled = True
        self.inputFixedImageSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Fixed Image: ", self.inputFixedImageSelector)

        self.inputMovingImageSelector = slicer.qMRMLNodeComboBox()
        self.inputMovingImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputMovingImageSelector.selectNodeUponCreation = False
        self.inputMovingImageSelector.noneEnabled = False
        self.inputMovingImageSelector.addEnabled = False
        self.inputMovingImageSelector.removeEnabled = True
        self.inputMovingImageSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Input Moving Image: ", self.inputMovingImageSelector)

        self.outputImageSelector = slicer.qMRMLNodeComboBox()
        self.outputImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.outputImageSelector.selectNodeUponCreation = False
        self.outputImageSelector.noneEnabled = False
        self.outputImageSelector.addEnabled = False
        self.outputImageSelector.removeEnabled = True
        self.outputImageSelector.setMRMLScene(slicer.mrmlScene)
        IOLayout.addRow("Output Image: ", self.outputImageSelector)
        
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
