import self as self
from PIL import ImageEnhance
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import sys
import cv2
import random
from math import *
from matplotlib import pyplot as plt
import numpy as np
import copy
from tkinter import *
import datetime


from skimage.util import random_noise

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import  mplot3d
from matplotlib.tri import Triangulation
import string

'''mpl.rcParams.update({'image.cmap': 'Accent',
                     'image.interpolation': 'none',
                     'xtick.major.width': 0,
                     'xtick.labelsize': 0,
                     'ytick.major.width': 0,
                     'ytick.labelsize': 0,
                     'axes.linewidth': 0})'''
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
        uic.loadUi('main.ui', self)
        self.origin = None
        self.Gray = None
        self.szFilter = None

        self.actionOpen.triggered.connect(self.openFile)
        self.actionExit.triggered.connect(self.Exit)
        self.btnMean.clicked.connect(self.Mean)
        self.btnBlur.clicked.connect(self.Blur)
        self.btnGaussBlur.clicked.connect(self.GaussBlur)
        self.btnMedian.clicked.connect(self.Median)
        self.btnGx.clicked.connect(self.Gx)
        self.btnGy.clicked.connect(self.Gy)
        self.btnGxAddGy.clicked.connect(self.GxAddGy)
        self.btnGxDivGy.clicked.connect(self.GxDivGy)
        self.btnDFT.clicked.connect(self.DFT)
        #self.btnLPF.clicked.connect(self.LPF)
        self.btnDirectional.clicked.connect(self.Directional)
        #self.btn3D.clicked.connect(self.Display)
        self.btnEqualize.clicked.connect(self.Equalize)
        self.btnEqualize_2.clicked.connect(self.equalizeHist)
        self.btnRefresh.clicked.connect(self.Refresh)
        self.btnSobelX.clicked.connect(self.SobelX)
        self.btnSobelY.clicked.connect(self.SobelY)
        self.btnPrewittX.clicked.connect(self.PrewittX)
        self.btnPrewittY.clicked.connect(self.PrewittY)
        self.btnCanny.clicked.connect(self.Canny)
        self.btnLaplacian.clicked.connect(self.Laplacian)
        self.btnTransform.clicked.connect(self.Transfrom)
        self.btnTozero.clicked.connect(self.Tozero)
        self.btnTrunC.clicked.connect(self.TrunC)
        self.btnThreshMeanC.clicked.connect(self.ThreshMeanC)
        self.btnThreshGaussC.clicked.connect(self.ThreshGaussC)
        self.btnThreshBinary.clicked.connect(self.ThreshBinary)
        self.btnKMean.clicked.connect(self.KMean)
        self.btnRotate.clicked.connect(self.Rotation)
        self.btnOtsu.clicked.connect(self.Otsu)
        self.btnScal05.clicked.connect(self.Scaling_15)
        self.btnShear.clicked.connect(self.Shearing)
        self.btnDilat05.clicked.connect(self.Dilating_05)
        self.btnReflexion.clicked.connect(self.Reflexion)
        self.btnCalcHist.clicked.connect(self.calcHist)
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

    def Refresh(self):
        self.labelAfter_2.clear()
        self.iframe_new_2.clear()
        self.labelAfter_3.clear()
        self.iframe_new_3.clear()
        self.labelAfter.clear()
        self.iframe_new.clear()


    def ShowImageGray(self, image, label):
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             QtGui.QImage.Format_Grayscale8).rgbSwapped()

        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def ShowImageGray2(self, image, label):
        label.clear()
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             QtGui.QImage.Format_Grayscale16).rgbSwapped()

        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def getGx(self):
        img = copy.copy(self.Gray)
        filterGx = np.array([
            [0,0,0],
            [-1,2,-1],
            [0,0,0]
        ])
        img_Gx = cv2.filter2D(img, -1, filterGx)
        return img_Gx
        #test = img_Gx + self.getGy()
        #self.ShowImageGray(img_Gx, self.iframe_new)

    def getGy(self):
        img = copy.copy(self.Gray)
        filterGy = np.array([
            [0,-1,0],
            [0,2,0],
            [0,-1,0]
        ])
        img_Gy = cv2.filter2D(img, -1, filterGy)
        return img_Gy

    def Gx(self):
        self.labelAfter.setText('Sau khi áp dụng GX')
        self.ShowImageGray(self.Gray-self.getGx(), self.iframe_new)

    def Gy(self):
        self.labelAfter.setText('Sau khi áp dụng GY')
        self.ShowImageGray(self.getGy(), self.iframe_new)

    def GxAddGy(self):
        self.labelAfter.setText('Sau khi áp dụng Gx + Gy')
        img = self.getGx() + self.getGy()
        self.ShowImageGray(img, self.iframe_new)

    def GxDivGy(self):
        self.labelAfter.setText('Sau khi áp dụng Gx/Gy')
        imgGx = self.getGx()
        imgGy = self.getGy()
        print(imgGx)
        print(imgGy)
        res = copy.copy(imgGx)
        for i in range(len(imgGx)):
            for j in range(len(imgGx[i])):
                if (imgGy[i][j] == 0):
                    continue
                res[i][j] = imgGx[i][j]/imgGy[i][j]*2
                #res[i][j]=atan(res[i][j])
        res = np.uint8(res)
        print(res)
        self.ShowImageGray(res, self.iframe_new)

    def Mean(self):
        self.labelAfter.setText('Sau khi áp dụng Mean Filter')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        img_filter = np.ones(shape=(self.szFilter, self.szFilter))
        img_filter = img_filter / sum(img_filter)
        img_mean = cv2.filter2D(img, -1, img_filter)
        self.ShowImageGray(img_mean, self.iframe_new)

    def Blur(self):
        self.labelAfter.setText('Sau khi áp dụng Blur Filter')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        blur = cv2.blur(img, (self.szFilter, self.szFilter))
        self.ShowImageGray(blur, self.iframe_new)


    def GaussBlur(self):
        self.labelAfter.setText('Sau khi áp dụng Gauss Filter')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        gauss = cv2.GaussianBlur(img, (self.szFilter, self.szFilter),0)
        self.ShowImageGray(gauss, self.iframe_new)


    def Median(self):
        self.labelAfter.setText('Sau khi áp dụng Median Filter')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        median = cv2.medianBlur(img, self.szFilter)
        self.ShowImageGray(median, self.iframe_new)


    def DFT(self):
        self.labelAfter.setText('Sau khi dùng DFT')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        img_float32 = np.float32(img)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        #self.ShowImageGray(magnitude_spectrum, self.iframe_new)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    def LPF(self):
        self.labelAfter.setText('Sau khi áp dụng LPF')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)

        img_float32 = np.float32(img)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = img.shape
        crow, ccol = rows / 2, cols / 2


        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_back, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    '''def Display(self):
        self.labelAfter.setText('AFTER APPLY FILTER')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = np.arange(-5, 5, 0.25)
        y = np.arange(-5, 5, 0.25)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x ** 2 + y ** 2)
        z = np.sin(r)

        surf = ax.plot_wireframe(x, y, z, color='blue', linewidth=0.8)

        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('$.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()'''

    def Directional(self):
        self.labelAfter.setText('Image 1')
        self.labelAfter_2.setText('Image 2')
        self.labelAfter_3.setText('Image 3')
        self.szFilter = self.boxFilter.value()
        #img1 = copy.copy(self.Gray)

        img = cv2.imread("image/Tower_of_Pisa.jpg")
        img = cv2.resize(img, (200, 200))
        a = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) / 3
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3
        c = np.rot90(a)
        d = np.rot90(b)

        dst = np.empty_like(img)
        noise = cv2.randn(dst, (0, 0, 0), (30, 30, 30))
        noise_image = cv2.addWeighted(img, 0.5, noise, 0.5, 30)

        img1 = cv2.filter2D(noise_image, -1, a)
        img2 = cv2.filter2D(noise_image, -1, b)
        img3 = cv2.filter2D(noise_image, -1, c)
        img4 = cv2.filter2D(noise_image, -1, d)

        plt.figure(figsize=(5, 11))
        plt.subplot(3, 2, 1), plt.imshow(img), plt.title('Original Image')
        plt.subplot(3, 2, 2), plt.imshow(noise_image), plt.title('Noise Image')
        plt.subplot(3, 2, 3), plt.imshow(img1), plt.title('Image with Filter 1')
        plt.subplot(3, 2, 4), plt.imshow(img2), plt.title('Image with Filter 2')
        plt.subplot(3, 2, 5), plt.imshow(img3), plt.title('Image with Filter 3')
        plt.subplot(3, 2, 6), plt.imshow(img4), plt.title('Image with Filter 4')
        plt.show()

        #self.ShowImageGray(img4, self.iframe_new_4)

        #self.ShowImageGray(img2, self.iframe_new_2)


    def Equalize(self):
        self.labelAfter.setText('Sau khi dùng Gaussian Filter')
        self.labelAfter_2.setText('Sau khi dùng Blur Filter')
        self.labelAfter_3.setText('Sau khi dùng Box Filter')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)

        box = cv2.boxFilter(img, -1, (53, 53))
        blur = cv2.blur(img, (13, 13))
        gaussian = cv2.GaussianBlur(img, (37, 37), 0)

        self.ShowImageGray(gaussian, self.iframe_new)
        self.ShowImageGray(blur, self.iframe_new_2)
        self.ShowImageGray(box, self.iframe_new_3)

    def SobelX(self):
        kernel_Sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])
        self.labelAfter.setText('Sau khi dùng Sobel X')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        output_1 = cv2.filter2D(img, -1, kernel_Sobel_x)
        # output_1 = cv2.resize(output_1, (800, 600))

        self.ShowImageGray(output_1, self.iframe_new)

    def SobelY(self):
        kernel_Sobel_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]])
        self.labelAfter.setText('Sau khi dùng Sobel Y')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        output_1 = cv2.filter2D(img, -1, kernel_Sobel_y)
        # output_1 = cv2.resize(output_1, (800, 600))

        self.ShowImageGray(output_1, self.iframe_new)

    def PrewittX(self):
        kernel_Prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]])
        self.labelAfter.setText('Sau khi dùng Prewitt X')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        output_1 = cv2.filter2D(img, -1, kernel_Prewitt_x)
        # output_1 = cv2.resize(output_1, (800, 600))

        self.ShowImageGray(output_1, self.iframe_new)

    def PrewittY(self):
        kernel_Prewitt_x = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]])
        self.labelAfter.setText('Sau khi dùng Prewitt Y')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        output_1 = cv2.filter2D(img, -1, kernel_Prewitt_x)
        # output_1 = cv2.resize(output_1, (800, 600))

        self.ShowImageGray(output_1, self.iframe_new)

    def Laplacian(self):
        self.labelAfter.setText('Sau khi dùng Laplacian')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        imgGauss = cv2.GaussianBlur(img, (3, 3), 0)

        # laplacian = cv2.Laplacian(imgGauss, cv2.CV_64F)
        laplacian = cv2.Laplacian(imgGauss, cv2.CV_16S, ksize=3)
        self.ShowImageGray2(laplacian, self.iframe_new)
        '''plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.show()'''

    def Canny(self):
        self.labelAfter.setText('Sau khi dùng Canny')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        edges = cv2.Canny(img, 100, 200)
        self.ShowImageGray(edges, self.iframe_new)

    #----------------
    def noise(image, probability):
        out = np.zeros(image.shape, np.uint8)
        thres = 1 - probability

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rnd = random.random()
                if rnd < probability:
                    out[i][j] = 0
                elif rnd > probability:
                    out[i][j] = 255
                else:
                    out[i][j] = image[i][j]
        return out




    # --------GAUSSIAN-------------
    def corr(img, mask):
        row, col = img.shape
        m, n = mask.shape
        new = np.zeros((row + m - 1, col + n - 1))
        n = n // 2
        m = m // 2
        filtering_img = np.zeros(img.shape)
        new[m:new.shape[0] - m, n:new.shape[1] - n] = img
        for i in range(m, new.shape[0] - m):
            for j in range(n, new.shape[1] - n):
                temp = new[i - m:i + m + 1, j - m: j + m + 1]
                # print(temp)
                result = temp * mask
                filtering_img[i - m, j - n] - result.sum()
        return filtering_img

    def gaussian(m, n, sigma):
        gaussian = np.zeros((m, n))
        m = m // 2
        n = n // 2
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = sigma * (2 * np.pi) ** 2
                x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gaussian[x + m, y + n] = (1 / x1) * x2
        return gaussian

    def Gauss(self):
        self.labelAfter.setText('Sau khi dùng MedianThreshold')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        g = self.gaussian(5, 5, 2)
        Ig1 = self.corr(img, g)
        g = self.gaussian(5, 5, 5)
        Ig2 = self.corr(img, g)
        edg = Ig1 - Ig2
        alpha = 30
        sharped = img + edg * alpha
        # plt.figure()
        #
        # plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Original")
        # plt.subplot(1, 2, 2), plt.imshow(sharped, cmap='gray'), plt.title("Sharpen")
        # plt.show()
        #self.ShowImageGray(sharped, self.iframe_new)
        cv2.imshow('sharped', sharped)
        cv2.imwrite('image/sample.png', sharped)
        cv2.waitKey(0)



    def ham(self):
        self.labelAfter.setText('Sau khi dùng AdaptiveThreshold')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh6 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        titles = ['BINARY', 'MEAN C', 'TRUN C', 'GAUSSIAN C', 'TOZERO', 'OTSU']
        images = [ th1, th2, thresh4, th3, thresh5, thresh6]

        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    def ThreshBinary(self):
        self.labelAfter.setText('Sau khi dùng Thres Binary')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.ShowImageGray(th1, self.iframe_new)

    def ThreshMeanC(self):
        self.labelAfter.setText('Sau khi dùng Thres Mean C')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        self.ShowImageGray(th2, self.iframe_new)

    def ThreshGaussC(self):
        self.labelAfter.setText('Sau khi dùng Thres Gauss C')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        self.ShowImageGray(th2, self.iframe_new)

    def Tozero(self):
        self.labelAfter.setText('Sau khi dùng To ZERO')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        self.ShowImageGray(th2, self.iframe_new)

    def Otsu(self):
        self.labelAfter.setText('Sau khi dùng Otsu')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ret, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.ShowImageGray(th2, self.iframe_new)

    def TrunC(self):
        self.labelAfter.setText('Sau khi dùng Trun C')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        self.ShowImageGray(th2, self.iframe_new)

    def KMean(self):
        self.labelAfter.setText('Sau khi dùng K Mean')
        #self.labelAfter_2.setText('Sau khi dùng K Mean + Binary')
        self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        #img = cv2.imread("image/Tower_of_Pisa.jpg")
        #cv2.imshow("Original", img)
        #ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        #cv2.imshow("BINARY", th1)
        z = img.reshape(-1, 3)
        z = np.float32(z)

        critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret, label, center = cv2.kmeans(z, K, None, critera, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)
        ret, th2 = cv2.threshold(res2, 100, 255, cv2.THRESH_BINARY)
        '''cv2.imshow("K Mean",res2)
        cv2.imshow("K Mean", th2)
        cv2.waitKey()
        cv2.destroyAllWindows()'''
        cv2.imshow("K Mean + BINARY" , res2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #self.ShowImageGray(th2, self.iframe_new)
        #self.ShowImageGray(th2, self.iframe_new_2)

    def linear_transformation(self, image, a):
        newmean, newstd = image.shape
        points = np.mgrid[0:newstd, 0:newmean].reshape((2, newmean * newstd))
        new_points = np.linalg.inv(a).dot(points).round().astype(int)
        x, y = new_points.reshape((2, newmean, newstd), order='F')
        indices = x + newstd * y
        return np.take(image, indices, mode='wrap')

    def Scaling_15(self):
        self.labelAfter.setText('After Scaling')
        # self.szFilter = self.boxFilter.value()
        img = copy.copy(self.Gray)
        Scal15 = self.linear_transformation(img, a)
        self.ShowImageGray(Scal15, self.iframe_new)

    def Dilating_18(self):
        self.labelAfter.setText('After Dilating ')
        img = copy.copy(self.Gray)
        Dilat18 = self.linear_transformation(img, b)
        self.ShowImageGray(Dilat18, self.iframe_new)

    def Dilating_05(self):
        self.labelAfter.setText('After Dilating')
        img = copy.copy(self.Gray)
        Dilat05 = self.linear_transformation(img, c)
        self.ShowImageGray(Dilat05, self.iframe_new)

    def Scaling_05(self):
        self.labelAfter.setText('After Scaling')
        img = copy.copy(self.Gray)
        Scal05 = self.linear_transformation(img, d)
        self.ShowImageGray(Scal05, self.iframe_new)

    def Shearing(self):
        self.labelAfter.setText('After Shearing')
        img = copy.copy(self.Gray)
        Shear = self.linear_transformation(img, x)
        self.ShowImageGray(Shear, self.iframe_new)

    def Rotation(self):
        self.labelAfter.setText('After Rotation')
        img = copy.copy(self.Gray)
        Rotation = self.linear_transformation(img, y)
        self.ShowImageGray(Rotation, self.iframe_new)

    def Reflexion(self):
        self.labelAfter.setText('After Reflexion')
        img = copy.copy(self.Gray)
        Reflexion = self.linear_transformation(img, z)
        self.ShowImageGray(Reflexion, self.iframe_new)

    def Transfrom(self):
        self.labelAfter.setText('After Transform')
        image = copy.copy(self.Gray)
        #image = Image.open('img/Tower_of_Pisa.jpg')
        row, col, ch = np.shape(image)
        # changing image to bytes so as to get pixel intesities
        image_to_float = image.tobytes()
        pixel_intensities = [image_to_float[i] for i in range(len(image_to_float))]

        img = np.array(pixel_intensities).reshape((row, col, ch))

        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        # cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = ((cdf_m - cdf_m.min()) * 255) / (cdf_m.max() - cdf_m.min())
        cdf_scaled = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf_scaled[img]
        self.ShowImageGray(img2, self.iframe_new)
        '''plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.title("Given Image cumulative distribution function")
        plt.plot(cdf_normalized, color='b')
        plt.hist(img.flatten(), 256, [0, 256], color='r')

        plt.subplot(2, 2, 2)
        plt.title("After Global Histogram Equalization")
        plt.plot(cdf_normalized, color='b')
        plt.hist(img2.flatten(), 256, [0, 256], color='r')

        plt.subplot(2, 2, 3)
        plt.title("Original Image")
        plt.axis('off')
        plt.imshow(img.astype('uint8'), cmap='gray')

        plt.subplot(2, 2, 4)
        plt.title("Transformed Image")
        plt.axis('off')
        plt.imshow(img2.astype('uint8'), cmap='gray')

        plt.show()'''

    def calcHist(self):
        self.labelAfter.setText('After CalcHist')
        #img = copy.copy(self.Gray)
        img = cv2.imread('image/Tower_of_Pisa.jpg')
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
        cv2.imshow("Original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def equalizeHist(self):
        self.labelAfter.setText("EqualizeHist")
        img = copy.copy(self.Gray)
        equ = cv2.equalizeHist(img)
        self.ShowImageGray(equ, self.iframe_new)













if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow()
    app.exec_()


Ui_MainWindow()

