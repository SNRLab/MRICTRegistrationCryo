import os
from pyexpat import model
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import SimpleITK as sitk
import sitkUtils
from slicer.util import NodeModify
import numpy as np
from sys import platform

class PhaseImageUnwrap(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Phase Image Unwrap"
    self.parent.categories = ["Filtering"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King, Junichi Tokuda"]
    self.parent.helpText = """
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
"""
    # Set module icon from Resources/Icons/<ModuleName>.png
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

class PhaseImageUnwrapWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # IO collapsible button
    IOCategory = qt.QWidget()
    self.layout.addWidget(IOCategory)

    IOLayout = qt.QFormLayout(IOCategory)

    # Install Dependencies
    self.dependenciesButton = qt.QPushButton("Install Dependencies")
    IOLayout.addWidget(self.dependenciesButton)
    self.dependenciesButton.connect('clicked()', self.installDependencies)

    # Input scan plane transform
    self.phaseImageSelector = slicer.qMRMLNodeComboBox()
    self.phaseImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.phaseImageSelector.selectNodeUponCreation = False
    self.phaseImageSelector.noneEnabled = False
    self.phaseImageSelector.addEnabled = False
    self.phaseImageSelector.removeEnabled = True
    self.phaseImageSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Phase Image: ", self.phaseImageSelector)

    self.unwrapImageButton = qt.QPushButton("Unwrap Image")
    self.unwrapImageButton.toolTip = "Create phase unwrap and gradient images"
    IOLayout.addWidget(self.unwrapImageButton)
    self.unwrapImageButton.connect('clicked()', self.onUnwrapImage)

  def onUnwrapImage(self):
    phaseImageNode = self.phaseImageSelector.currentNode()
    self.phaseGradient(phaseImageNode)

  def phaseGradient(self, inode):
    image = sitk.Cast(sitkUtils.PullVolumeFromSlicer(inode), sitk.sitkFloat64)
    scalarType = inode.GetImageData().GetScalarTypeAsString()
    if scalarType == 'unsigned short':
        print('imageBaseline*numpy.pi/2048.0 - numpy.pi')
        imagePhase = image*np.pi/2048.0# - np.pi
    else:
        print('imageBaseline*numpy.pi/4096.0')
        imagePhase = image*np.pi/4096.0 + np.pi

    # Calculate gradient magnitude
    imagePhaseUW = self.phaseUnwrap(imagePhase)
    phaseUnwrapNodeName = inode.GetName() + '-phaseUnwrap'
    sitkUtils.PushVolumeToSlicer(imagePhaseUW, name=phaseUnwrapNodeName, className='vtkMRMLScalarVolumeNode')

    imagePhaseGrad = sitk.GradientMagnitude(imagePhaseUW)

    phaseGradNodeName = inode.GetName() + '-phaseGrad'
    sitkUtils.PushVolumeToSlicer(imagePhaseGrad, name=phaseGradNodeName, className='vtkMRMLScalarVolumeNode')
  
  def phaseUnwrap(self, imagePhase):
    from skimage.restoration import unwrap_phase
    imagePhaseNP = sitk.GetArrayFromImage(imagePhase)
    imageUnwrappedNP = unwrap_phase(imagePhaseNP)
    imageUnwrapped = sitk.GetImageFromArray(imageUnwrappedNP)
    imageUnwrapped.SetOrigin(imagePhase.GetOrigin())
    imageUnwrapped.SetSpacing(imagePhase.GetSpacing())
    imageUnwrapped.SetDirection(imagePhase.GetDirection())
    return imageUnwrapped
    
  def installDependencies(self):
    slicer.util.pip_install("scikit-image")