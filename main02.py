from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import sys
import cv2
from math import *
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import string
mpl.rcParams.update({'image.cmap': 'Accent',
                     'image.interpolation': 'none',
                     'xtick.major.width': 0,
                     'xtick.labelsize': 0,
                     'ytick.major.width': 0,
                     'ytick.labelsize': 0,
                     'axes.linewidth': 0})
a = np.array([[1.5, 0],
              [0, 1]])
b = 1.8*np.eye(2)
c = .5*np.eye(2)
d = np.array([[1, 0],
              [0, .5]])
x = np.array([[1, 0],
              [.5, 1]])
alpha = np.pi/4
y = np.array([[np.cos(alpha), -np.sin(alpha)],
              [np.sin(alpha), np.cos(alpha)]])
z = np.array([[np.cos(2*alpha), np.sin(2*alpha)],
              [np.sin(2*alpha), -np.cos(2*alpha)]])
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        uic.loadUi('Conv2D.ui', self)
        self.origin = None
        self.Gray = None
        self.szFilter = None
        self.actionOpen.triggered.connect(self.openFile)
        self.actionExit.triggered.connect(self.Exit)
        self.btnDilat18.clicked.connect(self.Dilating_18)
        self.btnScal15.clicked.connect(self.Scaling_15)
        self.btnDilat05.clicked.connect(self.Dilating_05)
        self.btnScal05.clicked.connect(self.Scaling_05)
        self.btnShear.clicked.connect(self.Shearing)
        self.btnRotation.clicked.connect(self.Rotation)
        self.btnReflexion.clicked.connect(self.Reflexion)
        self.show()

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*.png *.xpm *.jpg *.jpeg *.tif);;Python Files (*.py)",
                                                  options=options)

        self.image = cv2.imread(fileName)

        if (self.image.shape[1] > self.iframe_old.width() or self.image.shape[0] > self.iframe_old.height()):
            self.image = cv2.resize(self.image, (self.iframe_old.width(), self.iframe_old.height()))
        self.origin = self.image
        self.Gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.ShowImageGray(self.Gray, self.iframe_old)

    def Exit(self):
        quit(0)

    def ShowImageGray(self, image, label):
        label.clear()
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             QtGui.QImage.Format_Grayscale8).rgbSwapped()

        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def linear_transformation(self, image, a):
        newmean, newstd = image.shape
        points = np.mgrid[0:newstd, 0:newmean].reshape((2, newmean * newstd))
        new_points = np.linalg.inv(a).dot(points).round().astype(int)
        x, y = new_points.reshape((2, newmean, newstd), order='F')
        indices = x + newstd * y
        return np.take(image, indices, mode='wrap')

    def Scaling_15(self):
        self.labelAfter.setText('After Scaling the plane in the x - axis by a factor of 1.5')
        #self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        Scal15 = self.linear_transformation(img, a)
        self.ShowImageGray(Scal15, self.iframe_new)
    def Dilating_18(self):
        self.labelAfter.setText('After Dilating the plane by a factor of 1.8')
        img = copy.copy(self.Gray)
        Dilat18 = self.linear_transformation(img, b)
        self.ShowImageGray(Dilat18, self.iframe_new)
    def Dilating_05(self):
        self.labelAfter.setText('After Dilating the plane by a factor of 0.5')
        img = copy.copy(self.Gray)
        Dilat05 = self.linear_transformation(img, c)
        self.ShowImageGray(Dilat05, self.iframe_new)
    def Scaling_05(self):
        self.labelAfter.setText('After Scaling the plane in the y - axis by a factor of 0.5')
        img = copy.copy(self.Gray)
        Scal05 = self.linear_transformation(img, d)
        self.ShowImageGray(Scal05, self.iframe_new)
    def Shearing(self):
        self.labelAfter.setText('After Shearing about the y - axis with a vertical displacement of + x/2')
        img = copy.copy(self.Gray)
        Shear = self.linear_transformation(img, x)
        self.ShowImageGray(Shear, self.iframe_new)
    def Rotation(self):
        self.labelAfter.setText('After Rotation through 45 degree about the origin')
        img = copy.copy(self.Gray)
        Rotation = self.linear_transformation(img, y)
        self.ShowImageGray(Rotation, self.iframe_new)
    def Reflexion(self):
        self.labelAfter.setText('After Reflexion in a line with inclination of 45 degree through the origin')
        img = copy.copy(self.Gray)
        Reflexion = self.linear_transformation(img, z)
        self.ShowImageGray(Reflexion, self.iframe_new)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow()
    app.exec_()