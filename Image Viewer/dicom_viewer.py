from turtle import color
from matplotlib.dates import MINUTES_PER_DAY
import pydicom as dicom
import matplotlib.pyplot as plot
from PyQt5 import QtWidgets, uic
import sys
import os
from PIL import Image
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import magic
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import math
from matplotlib.widgets import Cursor
from mplwidget import MplWidget
import random
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        #Load the UI Page 
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('GUI.ui', self)
        
        # Connect ui elements
        self.action_Open_Image.triggered.connect(self.open)
        self.zoomButton.clicked.connect(self.zoom)
        self.rotateNearestButton.clicked.connect(self.rotate_nearest_neighbour)
        self.rotateBilinearButton.clicked.connect(self.rotate_bilinear)
        self.shearButton.clicked.connect(self.shear)
        self.equalizeButton.clicked.connect(self.equalize)
        self.filterButton.clicked.connect(self.filter)
        self.filterButton_median.clicked.connect(self.median_filter)
        self.add_noise_button.clicked.connect(self.add_noise)
        self.transform_button.clicked.connect(self.fourier_open)
        self.filterButton_fourier.clicked.connect(self.frequency_filter)
        self.difference_button.clicked.connect(self.filter_difference)
        self.remove_pattern_button.clicked.connect(self.denoising)
        self.filtered_image.canvas.ax.axis('off')
        self.image.canvas.ax.axis('off')

        # create origial t image
        t_array = np.zeros((128, 128))
        t_array[29:49,29:99]=255
        t_array[49:99,54:74]=255
        self.t_image= Image.fromarray(np.uint8(t_array))
        self.img_np = np.asarray(self.t_image)
        # view t image in tab 3 
        scene = QtWidgets.QGraphicsScene(self)
        self.scene = scene
        figure = Figure()
        axes = figure.gca() 
        axes.imshow(t_array, cmap=plot.cm.gray)
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)
        self.original_T.setScene(scene)
        
    # Browse files
    def open(self):
        try:
            # open any png/bmp/dcm/jpg file 
            files_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image ', os.getenv('HOME'), "Images (*.png *.bmp *.dcm *.jpg *.jpeg)")
            self.path = files_name[0]
            #Check the file format if dicom or tiff(for CT image case)
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                self.dicom(self.path)
            else: 
                self.other_images(self.path) #for non dicom images
        except IOError:                 #in case of not selecting a file to view
            pass  
    #viewing jpg and bmp images with details 
    def other_images(self, path):
        try:                       #check for image validity
            img = Image.open(self.path)
            self.img_np = np.asarray(img) #put image pixel values in a numpy array
            # print image colour
            if(img.mode)=='RGB':
                self.colour.setText('Color(RGB)')
            elif (img.mode)=='L':
                self.colour.setText('Gray')
            elif (img.mode)=='1':
                self.colour.setText('Binary')
            else:
                self.colour.setText(img.mode)
            # viewing image
            pix = QPixmap(path)
            item = QtWidgets.QGraphicsPixmapItem(pix)
            scene = QtWidgets.QGraphicsScene(self)
            scene.addItem(item)
            self.graphicsView.setScene(scene)
            self.origina_image_graphicsView.setScene(scene)
            

            qqimg=(Image.open(self.path).convert('L')).toqpixmap()
            self.noisy_image = Image.open(self.path).convert('L')
            self.image.canvas.ax.clear()
            self.image.canvas.ax.imshow(self.img_np, interpolation = "None", cmap="gray")
            self.image.canvas.ax.axis('off')
            self.image.canvas.draw()

            self.magnitude.canvas.ax.clear()
            self.phase.canvas.ax.clear()
            self.log_magnitude.canvas.ax.clear()
            self.log_phase.canvas.ax.clear() 

            # get number of channels
            if (len(self.img_np.shape)==2):
                self.channels=1
            elif (len(self.img_np.shape)==3):
                self.channels=self.img_np.shape[2]

            # get width, height, and size
            self.width.setText(f'{img.width}')
            self.height.setText(f'{img.height}')

            # getting bit depth in bits/pixel
            min=int(np.amin(img))
            max=int(np.amax(img))
            self.bit_depth=(np.ceil(np.log2(max-min+1)))*self.channels
            self.bit_depth_label.setText(f'{self.bit_depth}')

            # getting size
            self.total_size.setText(f'{img.width* img.height* self.bit_depth}') 
            # mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
            # self.bit_depth.setText(f'{mode_to_bpp[img.mode]}')
            # clear dicom info boxes
            self.modality.clear() 
            self.name.clear()
            self.age.clear()
            self.body_part.clear()
        except IOError:                 #pop error message in case of corrupted files
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Corrupted Image! Please choose a valid image file')
            msg.setWindowTitle("Error")
            msg.exec_()

    def dicom(self, path):
        # read header info
        header=dicom.dcmread(path)

        # get image width, height, and size from header, if available 
        if hasattr(header, 'Rows'):
            self.width.setText(f'{header.Rows}')
        else:
            self.width.setText('Not Found')
        
        if hasattr(header, 'Columns'):
            self.height.setText(f'{header.Columns}')
        else:
            self.height.setText('Not Found')
        
        

        # getting bit depth in bits/pixel from header, if available 
        if hasattr(header, 'BitsAllocated'):
            self.bit_depth = header.BitsAllocated
            self.bit_depth_label.setText(f'{header.BitsAllocated}')
        else:
            self.bit_depth_label.setText('Not Found')
        
        self.total_size.setText(f'{(header.BitsAllocated *header.Columns * header.Rows )}')

        # getting image colour, modality, body part examined, and patients' info from header, if available 
        if hasattr(header, 'PhotometricInterpretation'):
             self.colour.setText(header.PhotometricInterpretation)
        else:
            self.colour.setText('Not Found') 
       
        if hasattr(header, 'Modality'):
             self.modality.setText(header.Modality)
        else:
            self.modality.setText('Not Found') 
        
        if hasattr(header, 'PatientName'):
             self.name.setText(f'{header.PatientName}')
        else:
            self.name.setText('Not Found') 
        
        if hasattr(header, 'PatientAge'):
             self.age.setText(f'{header.PatientAge}')
        else:
            self.age.setText('Not Found') 
       
        if hasattr(header, 'StudyDescription'):
             self.body_part.setText(header.StudyDescription)
        else:
            self.body_part.setText('Not Found') 

        #viewing dicom image    
        scene = QtWidgets.QGraphicsScene(self)
        self.scene = scene
        figure = Figure()
        axes = figure.gca()

        # hide axes
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        axes.imshow(header.pixel_array, cmap=plot.cm.bone)
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)
        self.graphicsView.setScene(scene)
        self.origina_image_graphicsView.setScene(scene)

        # view image in filtering tab
        ds = dicom.dcmread(self.path)
        new= ds.pixel_array.astype(float)
        scaled_image=(np.maximum(new, 0)/new.max())*255.0
        
        scaled_image=np.uint8(scaled_image)
        image_array_1d=np.asarray(scaled_image)
        width= image_array_1d.shape[1]
        height= image_array_1d.shape[0]
        img=Image.fromarray(scaled_image)

        qqimg=(img.toqpixmap())
        self.noisy_image = qqimg
        self.unfilteredImage.clear()
        self.filteredImage.clear()
        self.unfilteredImage.setPixmap(qqimg) 

        self.image.canvas.ax.clear()
        self.image.canvas.ax.imshow(scaled_image, interpolation = "None", cmap="gray")
        self.image.canvas.ax.axis('off')
        self.image.canvas.draw()
        
    def nearest_neighbour_interpolation(self, path, factor):
        try:
            # for dicom files  
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                ds = dicom.dcmread(path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                scaled_image=np.uint8(scaled_image)
                image=np.asarray(scaled_image)
                width= image.shape[1]
                height= image.shape[0]
                image=Image.fromarray(scaled_image)

            #  if the image is gray no need to convert it it already has 1 channel
            elif self.channels==1 and Image.open(path).mode=='L':
                image=Image.open(path)
            else:
                image=Image.open(path).convert('L') 
            width=image.width
            height=image.height
            # create an array with new dimensions
            image=np.asarray(image)
            new=np.arange(0,int(np.ceil(factor*height))*int(np.ceil(factor*width)))
            newimgarr=new.reshape(int(np.ceil(factor*height)),int(np.ceil(factor*width)))
            
            # Loop over the whole image oixels replacing current pixel of new array with the nearest pixel from origonal image
            for i in range(0,int(np.ceil(factor*height))):
                for j in range(0,int(np.ceil(factor*width))):
                    if i/factor >height-1 or j/factor>width-1:
                        # print('int')
                        inew= int(i/factor)
                        jnew= int(j/factor)
                    else:
                        # print('round')
                        inew= round(i/factor)
                        jnew= round(j/factor)
                    newimgarr[i,j]=image[inew,jnew]
            # create a image from new array
            newimg= Image.fromarray(np.uint8(newimgarr))
            # viewing interpolated image
            qqimg=newimg.toqpixmap()
            self.nearest.clear()
            self.nearest.setPixmap(qqimg)  
            self.old_dim.setText(f'{width}'+'x'+ f'{height}')
            self.new_dim.setText(f'{int(np.ceil(factor*width))}' +'x'+ f'{int(np.ceil(factor*height))}')
        except IOError:                 #pop error message in case of error while interpolating
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
   
    def bilinear_interpolation(self, path, factor):
        try:
            # for dicom images
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                ds = dicom.dcmread(path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                scaled_image=np.uint8(scaled_image)
                image=np.asarray(scaled_image)
                old_w= image.shape[0]
                old_h= image.shape[1]
                
                 #  if the image is gray no need to convert it it already has 1 channel
            elif self.channels==1 and Image.open(path).mode=='L':
                image=Image.open(path)
                old_w=image.width
                old_h=image.height
                image=np.asarray(image)
            else:
                image=Image.open(path).convert('L') 
                old_w=image.width
                old_h=image.height
                image=np.asarray(image)
            
            # getting new dimensions
            new_w=int(np.ceil(old_w*factor))
            new_h=int(np.ceil(old_h*factor))
            image=np.asarray(image)
            
            # craete array of zeroes with new dimensions
            new_arr = np.zeros((new_h, new_w))

            #get 1/scale factor to use it to get coordinated from original image
            w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
            h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0

            # loop over the whole image
            for i in range(new_h):
                for j in range(new_w):
                    #map the coordinates back to the original image
                    x = i * h_scale_factor
                    y = j * w_scale_factor
                    #get coordinates of 4 surrounding pixels
                    x_floor = math.floor(x)
                    # height and width minus one to avoid index error
                    x_ceil = min( old_h - 1, math.ceil(x))
                    y_floor = math.floor(y)
                    y_ceil = min(old_w - 1, math.ceil(y))

                    # get pixel mvalue from original image as is
                    if (x_ceil == x_floor) and (y_ceil == y_floor):
                        q = image[int(x), int(y)]
                    elif (x_ceil == x_floor):
                        q1 = image[int(x), int(y_floor)]
                        q2 = image[int(x), int(y_ceil)]
                        q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                    elif (y_ceil == y_floor):
                        q1 = image[int(x_floor), int(y)]
                        q2 = image[int(x_ceil), int(y)]
                        q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
                    # for pixels in the middle
                    else:
                        v1 = image[x_floor, y_floor]
                        v2 = image[x_ceil, y_floor]
                        v3 = image[x_floor, y_ceil]
                        v4 = image[x_ceil, y_ceil]

                        # multiply pixel value with distance of other pixels and vice versa
                        q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                        q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                        q = q1 * (y_ceil - y) + q2 * (y - y_floor)

                    new_arr[i,j] = q

            # create new image from new array
            newimgarr=new_arr.astype(np.uint8)
            newimg= Image.fromarray(np.uint8(newimgarr))       
            # viewing image
            qqimg=newimg.toqpixmap()
            self.bilinear.clear()
            self.bilinear.setPixmap(qqimg)  #.scaled(int(factor*width), int(factor*height))
        except IOError:                 #pop error message in case of error in interpolation
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
    def zoom(self):
        # get scale factor from user
        self.factor = self.doubleSpinBox.value()
        # handling tha case of requesting zero factor
        if self.factor<=0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Image Viewer")
            msg.setInformativeText('Choose an image and enter a non-zero factor!')
            msg.setWindowTitle("Zoom")
            msg.exec_()
        else:
            self.nearest_neighbour_interpolation(self.path, self.factor)
            self.bilinear_interpolation(self.path, self.factor)
            

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Image is zoomed")
            msg.setInformativeText('Open tab 2 to view the image')
            msg.setWindowTitle("Zoom")
            msg.exec_()
    
    def rotate_nearest_neighbour(self):
        # get rotation angle from the user
        self.angle_direction.clear()
        degree= self.doubleSpinBox_2.value()
        # convert angle to radians
        rads = math.radians(degree)
        # print angle of rotation and direction
        if degree> 0 and degree<360:
            deg =degree
            angle=' Anticlockwise'
        elif degree>-360 and degree<0:
            deg =-degree
            angle= ' Clockwise'
        elif degree%360==0:
            deg =degree
            angle= ' No rotation'
        self.angle_direction.setText('Angle: '+ f'{deg}'+ angle)
        rot_img = np.uint8(np.zeros(self.img_np.shape))
        # getting the centre point of image
        height = rot_img.shape[0]
        width  = rot_img.shape[1]
        midx,midy = (width//2, height//2)

        # loop over the array of image to replace with new rotated coordinates
        for i in range(rot_img.shape[0]):
            for j in range(rot_img.shape[1]):
                # apply fixed point rotation by subtracting the centre then multiply the rotation matrix the add the centre again
                x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads) 
                y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads) 
                
                x = x +midx
                y = y +midy
                # interpolate with nearest neighour to assign new coordiates with pixel values
                if x >height-1 or y>width-1:
                    x= int(x) 
                    y= int(y)
                else:
                    x= round(x) 
                    y= round(y) 
                    # to only assign pixel values to coordinates within range of old dim (128*128) (shown cropped)
                if (x>=0 and y>=0 and x<self.img_np.shape[0] and  y<self.img_np.shape[1]):
                    rot_img[i,j] = self.img_np[x,y]

       
        rot_imgg = Image.fromarray(np.uint8(rot_img))

        # view rotated t image in tab 3 
        scene = QtWidgets.QGraphicsScene(self)
        self.scene = scene
        figure = Figure()
        axes = figure.gca() 
        axes.imshow(rot_imgg, cmap=plot.cm.gray)
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)
        self.rotated_T.setScene(scene)

    def shear(self):
        self.angle_direction.clear()
        rot_img = np.uint8(np.zeros(self.img_np.shape))
        # Finding the center point of rotated (or original) image.
        height = rot_img.shape[0]
        width  = rot_img.shape[1]
        # getting the centre point of image
        midx,midy = (width//2, height//2)
        
        # loop over the array of image to replace with new sh5eared coordinates
        for i in range(rot_img.shape[0]):
            for j in range(rot_img.shape[1]):
                # apply fixed point shearing by subtracting the centre then multiply the shearing matrix then add the centre again

                x= (i-midx) 
                y= (j-midy)
                y = x+y
                x = x +midx
                y = y +midy
                # interpolate with nearest neighour to assign new coordiates with pixel values
                if x >height-1 or y>width-1:
                    x= int(x) 
                    y= int(y)
                else:
                    x= round(x) 
                    y= round(y) 
                # to only assign pixel values to coordinates within range of old dim (128*128) (shown cropped)
                if (x>=0 and y>=0 and x<self.img_np.shape[0] and  y<self.img_np.shape[1]):
                    rot_img[i,j] = self.img_np[x,y]

       
        rot_imgg = Image.fromarray(np.uint8(rot_img))

        # view rotated t image in tab 3 
        scene = QtWidgets.QGraphicsScene(self)
        self.scene = scene
        figure = Figure()
        axes = figure.gca() 
        axes.imshow(rot_imgg, cmap=plot.cm.gray)
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)
        self.rotated_T.setScene(scene)
    def rotate_bilinear(self): 
        self.angle_direction.clear()
        # get rotation angle from the user
        degree= self.doubleSpinBox_2.value()
        # convert angle to radians
        rads = math.radians(degree)
        # print angle of rotation and direction
        if degree> 0 and degree<360:
            deg =degree
            angle=' Anticlockwise'
        elif degree>-360 and degree<0:
            deg =-degree
            angle= ' Clockwise'
        elif degree%360==0:
            deg =degree
            angle= ' No rotation'
        self.angle_direction.setText('Angle: '+ f'{deg}'+ angle)
        rot_img = np.uint8(np.zeros(self.img_np.shape))
        # getting the centre point of image
        height = rot_img.shape[0]
        width  = rot_img.shape[1]
        midx,midy = (width//2, height//2)
        # loop over the array of image to replace with new rotated coordinates
        for i in range(rot_img.shape[0]):
            for j in range(rot_img.shape[1]):
                # apply fixed point rotation by subtracting the centre then multiply the rotation matrix the add the centre again
                x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads) 
                y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads) 
                
                x = x +midx
                y = y +midy

                # interpolate with bilinear to assign new coordiates with pixel values

                

                #get coordinates of 4 surrounding pixels
                x_floor = math.floor(x) 
                # height and width minus one to avoid index error
                x_ceil = min( height - 1, math.ceil(x)) 
                y_floor = math.floor(y) 
                y_ceil = min(width - 1, math.ceil(y)) 
                if y_floor > 0 and y_floor < width-1 and x_floor > 0 and x_floor < height-1: # to only assign pixel values to coordinates within range of old dim (128*128) (shown cropped)
                    if (x_ceil == x_floor) and (y_ceil == y_floor):
                        q = self.img_np[int(x), int(y)]
                        rot_img[i,j] = q
                    elif (x_ceil == x_floor):
                        q1 = self.img_np[int(x), int(y_floor)]
                        q2 = self.img_np[int(x), int(y_ceil)]
                        q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                        rot_img[i,j] = q
                    elif (y_ceil == y_floor):
                        q1 = self.img_np[int(x_floor), int(y)]
                        q2 = self.img_np[int(x_ceil), int(y)]
                        q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
                        rot_img[i,j] = q 
                    else:
                    # if (y_floor > 0 and y_floor < width-1 and x_floor > 0 and x_floor < height-1):
                        v1 = self.img_np[x_floor, y_floor]
                        v2 = self.img_np[x_ceil, y_floor]
                        v3 = self.img_np[x_floor, y_ceil]
                        v4 = self.img_np[x_ceil, y_ceil]

                        # multiply pixel value with distance of other pixels and vice versa
                        q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                        q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                        q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            
                        rot_img[i,j] = q
                        
        rot_imgg = Image.fromarray(np.uint8(rot_img))
        # view rotated t image in tab 3 
        scene = QtWidgets.QGraphicsScene(self)
        self.scene = scene
        figure = Figure()
        axes = figure.gca() 
        axes.imshow(rot_imgg, cmap=plot.cm.gray)
        canvas = FigureCanvas(figure)
        canvas.setGeometry(0, 0, 500, 500)
        scene.addWidget(canvas)
        self.rotated_T.setScene(scene)

    def histogram(self, image_array): #this function takes a 1D image array as an input and returns its histogram
        try:
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                normalized_histogram = np.zeros(self.max_bit_depth)
                pixel_val=np.arange(self.max_bit_depth)
                for i in range(image_array.shape[0]):
                    for j in range(image_array.shape[1]):
                        normalized_histogram[int(image_array[i, j])] += 1
                        
                normalized_histogram /= (image_array.shape[0] * image_array.shape[1])
            else:
                # create an array of zeros to store frequencies
                histogram = np.zeros(self.max_bit_depth) 
                # create an array of zeros to store  normalized frequencies
                normalized_histogram = np.zeros(self.max_bit_depth) 
                # create an array to stor ordered pixel values
                pixel_val=np.arange(self.max_bit_depth)
                # count the numer of occurence of each pixel value in the image and add it to the array of frequencies
                for i in range(len(image_array)):
                    histogram[image_array[i]] +=1
                # normalize the frequencies by dividing each count by size of image
                for i in range(len(histogram)):
                    normalized_histogram[i] = histogram[i]/len(image_array)
            res = [pixel_val, normalized_histogram]
                # return both arrays to be plotted on axes
            return res
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def equalize(self):
        try:
            # self.max_bit_depth = int(2**self.bit_depth)
            # print(self.max_bit_depth)
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                    ds = dicom.dcmread(self.path)
                    new= ds.pixel_array.astype(float)
                    scaled_image=(np.maximum(new, 0)/new.max())*255.0
                    # print(new.max())
                    scaled_image=np.uint8(scaled_image)
                    image_array_1d=np.asarray(scaled_image)
                    width= image_array_1d.shape[1]
                    height= image_array_1d.shape[0]
                    img=Image.fromarray(scaled_image)

                    max = np.amax(scaled_image)
                    depth = math.ceil(math.log((int(max) + 1), 2))
                    self.max_bit_depth = 2 ** depth
                    self.img_np = np.asarray(img) #put image pixel values in a numpy array
                    # viewing image
                    qqimg=img.toqpixmap()
                    self.originalImage.clear()
                    self.originalImage.setPixmap(qqimg)

                    # calculate histogram of the original image and plot it
                    output = self.histogram(scaled_image)
                    normalized_histogram = output[1]
                
                    self.originalHistogram.canvas.ax.clear()
                    self.originalHistogram.canvas.ax.bar(output[0], output[1])
                    self.originalHistogram.canvas.draw()

                    output = self.histogram(image_array_1d)
                    normalized_histogram = output[1]
                    eq_histo = np.zeros_like(normalized_histogram)
                    equalized_image = np.zeros_like(scaled_image)

                    for i in range(len(normalized_histogram)):
                        eq_histo[i] = round((self.max_bit_depth - 1) * np.sum(normalized_histogram[0:i]))

                    for i in range(scaled_image.shape[0]):
                        for j in range(scaled_image.shape[1]):
                            pixel_val = int(scaled_image[i, j])
                            equalized_image[i, j] = eq_histo[pixel_val]

                    self.equalizedHistogram.canvas.ax.clear()
                    # calculate equalized histogram and plot it
                    equalized_output = self.histogram(equalized_image)
                    self.equalizedHistogram.canvas.ax.bar(equalized_output[0], equalized_output[1])
                    self.equalizedHistogram.canvas.draw()
                
                    newimg= Image.fromarray(np.uint8(equalized_image)) 
                    # newimg.show()

                    # viewing equalized image
                    qqimg=newimg.toqpixmap()
                    self.equalizedImage.clear()
                    self.equalizedImage.setPixmap(qqimg) 

            else:
                # convert image to grayscale
                img = Image.open(self.path).convert('L') 
                # img.show()
                width= self.img_np.shape[1]
                height= self.img_np.shape[0]
                # create a 1D array of the imagee
                image_array_1d = self.img_np.reshape(-1)
                max=int(np.amax(img))
                self.max_bit_depth= int(2 ** (np.ceil(np.log2(max+1))))


                self.img_np = np.asarray(img) #put image pixel values in a numpy array
                # viewing image
                qqimg=img.toqpixmap()
                self.originalImage.clear()
                self.originalImage.setPixmap(qqimg)
            
            
       
                # print(self.max_bit_depth)
                # print(len(image_array_1d))
                # calculate histogram of the original image and plot it
                output = self.histogram(image_array_1d)
                normalized_histogram = output[1]
            
                self.originalHistogram.canvas.ax.clear()
                self.originalHistogram.canvas.ax.bar(output[0], output[1])
                self.originalHistogram.canvas.draw()
                
            

                # create an array of zeros to store cumilative fr3equency values
                cdf = np.zeros(self.max_bit_depth) 
                # create an array of zeros to store new equalized intensities
                sk = np.zeros(self.max_bit_depth) 
                # create an array of the same image size for the new equalized image
                equalized_image = np.zeros((int(height), int(width)))
                # cdf
                cdf[0] = normalized_histogram[0]
                for i in range(len(cdf)):
                    cdf[i] = normalized_histogram[i] + cdf[i-1]
                # get sk
                for i in range(len(cdf)):
                    sk[i] = round(cdf[i] * (self.max_bit_depth - 1))


                # get equalized image
                for i in range(height):
                    for j in range(width):
                        equalized_image[i,j] = sk[int(self.img_np[i,j])]
                equalized_1d = equalized_image.reshape(-1)
                equalized_int = np.zeros(len(equalized_1d), dtype=np.uint)
                for i in range(len(equalized_1d)):
                    equalized_int[i] = equalized_1d[i]

                self.equalizedHistogram.canvas.ax.clear()
                # calculate equalized histogram and plot it
                equalized_output = self.histogram(equalized_int)
                self.equalizedHistogram.canvas.ax.bar(equalized_output[0], equalized_output[1])
                self.equalizedHistogram.canvas.draw()
            
                newimg= Image.fromarray(np.uint8(equalized_image)) 
                # newimg.show()

                # viewing equalized image
                qqimg=newimg.toqpixmap()
                self.equalizedImage.clear()
                self.equalizedImage.setPixmap(qqimg)  


            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Image is equalized")
            msg.setInformativeText('Open Equaliation tab to view the image.')
            msg.setWindowTitle("Equalize")
            msg.exec_()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def filter(self, path):
        try:
            # create kernel
            kernel_size= self.kernelSize.value()
            kFactor= self.kFactor.value()
            kernel_value= 1/(kernel_size*kernel_size) 
            kernel = np.full((kernel_size, kernel_size), kernel_value)

            # for dicom images
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                    
                ds = dicom.dcmread(self.path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                
                scaled_image=np.uint8(scaled_image)
                original_image_array=np.asarray(scaled_image)
                # image original size
                original_image_width= original_image_array.shape[1]
                original_image_height= original_image_array.shape[0]
                img=Image.fromarray(scaled_image)

                
            else:

                # image original size
                image=Image.open(self.path).convert('L') 
                original_image_width=image.width
                original_image_height=image.height
                original_image_array=np.asarray(image)

            # image padded size
            padded_image_width=original_image_width + kernel_size -1
            padded_image_height=original_image_height + kernel_size -1

            # padded image
            padded_image_array = np.zeros((padded_image_height, padded_image_width))
            for i in range(kernel_size//2, original_image_height + kernel_size//2):
                for j in range(kernel_size//2, original_image_width + kernel_size//2):
                    padded_image_array[i][j] = original_image_array[i - kernel_size//2][j - kernel_size//2]

            # create array for new filtered padded image
            filtered_image_array = np.zeros((padded_image_height, padded_image_width))

            # apply filter
            # looping over the original image with the kernel size replacing the centre of the box each time
            for i in range(original_image_height):
                for j in range(original_image_width):
                    sum = 0
                    for k in range(kernel_size):
                        for l in range(kernel_size):
                            # multiply the filter by the image pixel values for each box and add them 
                            sum += padded_image_array[k + i,l + j] * kernel[k,l]
                    # replace the box centre by the summation
                    filtered_image_array[i + kernel_size//2,j + kernel_size//2] = sum
            
            # obtain the edges by subtracting the blurred filtered image from the original image
            subtracted_image_array = padded_image_array - filtered_image_array 
            
            # amplify the edges using the kfactor input from the user
            multiplied_image_array = subtracted_image_array * kFactor

            # add the enhanced edges to the original image to obtain the final high boosted image
            added_image_array = multiplied_image_array + padded_image_array
            
            # scale the values of the new image to be in range 0-255
            if (self.scaleRadioButton.isChecked()):
                a = added_image_array - np.min(added_image_array)
                maxa = np.max(a)
                mina = np.min(a)
                final_image_array = (a/maxa) *255
            # clip the values of the new image, for negatives to be zero and >255 to be 255
            elif (self.clipRadioButton.isChecked()):
                final_image_array = self.clip(added_image_array)
            unclipped_image = final_image_array[kernel_size//2:padded_image_height - kernel_size//2,kernel_size//2:padded_image_width - kernel_size//2]
            filtered_image= Image.fromarray(np.uint8(unclipped_image))
            
            # view new image
            qqimg=filtered_image.toqpixmap()
            self.filteredImage.clear()
            self.filteredImage.setPixmap(qqimg) 

            # warn the user in case of using an even filter that the image will be slightly shifted 
            if kernel_size%2 ==0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Warning")
                msg.setInformativeText('Using an even kernel size may result in a little shift in the image, we recommend using an odd kernel size.')
                msg.setWindowTitle("Filter")
                msg.exec_()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def clip(self, image_array):
        try:
            # scale image values to be in range 0-255 by replacing negatives with 0 and values >255 with 255
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    if image_array[i][j] < 0:
                        image_array[i][j] = 0
                    elif image_array[i][j] > 255:
                        image_array[i][j] = 255
            return image_array
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
    def median_filter(self):
        try:
            # create kernel
            kernel_size= self.kernelSize_median.value()
            temp = []

            # image original size
            image=self.noisy_image
            original_image_width=image.width
            original_image_height=image.height
            original_image_array=np.asarray(image)
            # image padded size
            padded_image_width=original_image_width + kernel_size -1
            padded_image_height=original_image_height + kernel_size -1

            # padded image
            padded_image_array = np.zeros((padded_image_height, padded_image_width))
            for i in range(kernel_size//2, original_image_height + kernel_size//2):
                for j in range(kernel_size//2, original_image_width + kernel_size//2):
                    padded_image_array[i][j] = original_image_array[i - kernel_size//2][j - kernel_size//2]
            
            # create new array for filtered padded image
            filtered_image_array = np.zeros((padded_image_height, padded_image_width))
            for i in range(original_image_height):
                for j in range(original_image_width):
                    for k in range(kernel_size):
                        for l in range(kernel_size):
                            # add elements of first box to a temp array to sort them
                            temp.append(padded_image_array[k + i,l + j])
                    mergeSort(temp)
                    # replace centre of the box with the median of the box pixels
                    filtered_image_array[i + kernel_size//2,j + kernel_size//2] = temp[len(temp) // 2]
                    temp = []
            unclipped_image_array = filtered_image_array[kernel_size//2:padded_image_height - kernel_size//2,kernel_size//2:padded_image_width - kernel_size//2]
            unclipped_image= Image.fromarray(np.uint8(unclipped_image_array))
            
            # show new image
            qqimg=unclipped_image.toqpixmap()
            self.filteredImage.clear()
            self.filteredImage.setPixmap(qqimg) 

            # warn the user in case of using an even filter that the image will be slightly shifted
            if kernel_size%2 ==0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Warning")
                msg.setInformativeText('Using an even kernel size may result in a little shift in the image, we recommend using an odd kernel size.')
                msg.setWindowTitle("Filter")
                msg.exec_()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
    def add_noise(self):
        try:

            # for dicom images
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                    
                ds = dicom.dcmread(self.path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                
                scaled_image=np.uint8(scaled_image)
                original_image_array=np.asarray(scaled_image)
                original_image_width= original_image_array.shape[1]
                original_image_height= original_image_array.shape[0]
                img=Image.fromarray(scaled_image)
            else:
                # get original size
                image=Image.open(self.path).convert('L') 
                original_image_width=image.width
                original_image_height=image.height
                original_image_array=np.asarray(image)

            # get percent of noise as a user input
            percent = self.noise_percent.value()
            # create an array for new noisy image
            noisy_image_array = np.zeros((original_image_height, original_image_width))

            # number of pixel is a variable of both salt and pepper noise percent in ther image so multiply percent by image size
            number_of_pixels = int((percent/200)*original_image_width*original_image_height)
            
            # make new image equal to original image as a first step
            for i in range(original_image_height):
                for j in range(original_image_width):
                    noisy_image_array[i][j] = original_image_array[i][j]
            
            # then add noise randomly across the image
            for i in range(number_of_pixels):
                # get random coordinats inside the image
                y_coord=random.randint(0, original_image_height - 1) 
                x_coord=random.randint(0, original_image_width - 1)

                # replace random coordinates with white pixels, AKA salt
                noisy_image_array[y_coord][x_coord] = 255
            for i in range(number_of_pixels):
                # get random coordinats inside the image
                y_coord=random.randint(0, original_image_height - 1)
                x_coord=random.randint(0, original_image_width - 1)

                # replace random coordinates with black pixels, AKA pepper
                noisy_image_array[y_coord][x_coord] = 0

            # view noisy image    
            self.noisy_image= Image.fromarray(np.uint8(noisy_image_array))
            qqimg=self.noisy_image.toqpixmap()
            self.unfilteredImage.clear()
            self.unfilteredImage.setPixmap(qqimg) 
            return noisy_image_array
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
    
    def zero_pad_kernel(self, kernel, image_array): 
        try:
            # image dimensions
            img_height = image_array.shape[0]
            img_width = image_array.shape[1]

            # kernel dimensions
            kernel_height = kernel.shape[0]
            kernel_width = kernel.shape[1]

            # padding dimensions
            pad_width = int((img_width - kernel_width) / 2 )
            pad_height = int((img_height - kernel_height) / 2)

            # created new padded kernel array
            padded_kernel = np.zeros((img_height, img_width)) 
            for i in range(pad_height, kernel_height + pad_height):
                for j in range(pad_width, kernel_width + pad_width):
                    padded_kernel[i][j] = kernel[i - pad_height][j - pad_width] # inserting image data within the frame of the padding
            return padded_kernel
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def fourier_open(self):
        try:
            # to open dicom files
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                        
                ds = dicom.dcmread(self.path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                
                scaled_image=np.uint8(scaled_image)
                original_image_array=np.asarray(scaled_image)

            # open a gray scale image
            elif self.channels==1 and Image.open(self.path).mode=='L':
                image=Image.open(self.path)
                original_image_array=np.asarray(image)
            else:
                image=Image.open(self.path).convert('L')
                original_image_array=np.asarray(image)
            fourier_shift = self.fourier_transform(original_image_array)

            # get real and imaginary components
            real_component = fourier_shift.real
            imaginary_component = fourier_shift.imag

            # calculate magnitude and phase
            magnitude = np.sqrt((real_component ** 2) + (imaginary_component ** 2))
            phase = np.arctan2(imaginary_component, real_component)

            # apply log to the magnitude, while adding 1 to scale
            log_magnitude = np.log(magnitude + 1)

            # apply log to the phase, while adding 2pi to add another cycle making the range 0-2pi instead of -pi-pi, as no negative is allowed inside the log
            log_phase = np.log(phase + 2*math.pi)

            # plot magnitude
            self.magnitude.canvas.ax.clear()
            self.magnitude.canvas.ax.imshow(magnitude, interpolation = "None", cmap="gray")
            self.magnitude.canvas.draw()

            # plot phase
            self.phase.canvas.ax.clear()
            self.phase.canvas.ax.imshow(phase, interpolation = "None", cmap="gray")
            self.phase.canvas.draw()

            # plot magnitude after log
            self.log_magnitude.canvas.ax.clear()
            self.log_magnitude.canvas.ax.imshow(log_magnitude, interpolation = "None", cmap="gray")
            self.log_magnitude.canvas.draw()

            # plot phase after log
            self.log_phase.canvas.ax.clear()
            self.log_phase.canvas.ax.imshow(log_phase, interpolation = "None", cmap="gray")
            self.log_phase.canvas.draw()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def fourier_transform(self, array):
        try:
            # apply fourier transform and fourier shift to the image array
            fourier = np.fft.fft2(array)
            fourier_shift = np.fft.fftshift(fourier)
            
            return fourier_shift
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
        
    def frequency_filter(self): 
        try:
            # to open dicom files
            if magic.from_file(self.path) == 'DICOM medical imaging data' or magic.from_file(self.path)=='TIFF image data, little-endian':
                        
                ds = dicom.dcmread(self.path)
                new= ds.pixel_array.astype(float)
                scaled_image=(np.maximum(new, 0)/new.max())*255.0
                
                scaled_image=np.uint8(scaled_image)
                original_image_array=np.asarray(scaled_image)

            # open a gray scale image
            elif self.channels==1 and Image.open(self.path).mode=='L':
                image=Image.open(self.path)
                original_image_array=np.asarray(image)
            else:
                image=Image.open(self.path).convert('L')
                original_image_array=np.asarray(image)

            # create kernel
            kernel_size= self.kernelSize_fourier.value()
            kernel_value= 1/(kernel_size*kernel_size) 
            kernel = np.full((kernel_size, kernel_size), kernel_value)  

            # image original size
            original_image_width=original_image_array.shape[1]
            original_image_height=original_image_array.shape[0]

            # padded kernel
            padded_kernel = self.zero_pad_kernel(kernel, original_image_array)
            
            # apply fourier transform to both kernel and image
            padded_kernel_fourier_shift = self.fourier_transform(padded_kernel)
            image_fourier_shift = self.fourier_transform(original_image_array)
            
            # multiply kernel and image in frequency domain
            output = padded_kernel_fourier_shift * image_fourier_shift
            
            # obtain the inverse fourier of the filtered image to restore it
            output_1 = np.fft.ifftshift(output)
            inverse_fourier_output = (np.fft.fftshift(np.fft.ifft2(output_1))).real
            
            # spatial domain filtering
            # image padded size
            padded_image_width=original_image_width + kernel_size -1
            padded_image_height=original_image_height + kernel_size -1

            # padded image
            padded_image_array = np.zeros((padded_image_height, padded_image_width))
            for i in range(kernel_size//2, original_image_height + kernel_size//2):
                for j in range(kernel_size//2, original_image_width + kernel_size//2):
                    padded_image_array[i][j] = original_image_array[i - kernel_size//2][j - kernel_size//2]

            # create array for new filtered padded image
            filtered_image_array = np.zeros((padded_image_height, padded_image_width))

            # apply filter
            # looping over the original image with the kernel size replacing the centre of the box each time
            for i in range(original_image_height):
                for j in range(original_image_width):
                    sum = 0
                    for k in range(kernel_size):
                        for l in range(kernel_size):
                            # multiply the filter by the image pixel values for each box and add them 
                            sum += padded_image_array[k + i,l + j] * kernel[k,l]
                    # replace the box centre by the summation
                    filtered_image_array[i + kernel_size//2,j + kernel_size//2] = sum
            
            spatial_filtered_image_array = filtered_image_array[kernel_size//2:padded_image_height - kernel_size//2,kernel_size//2:padded_image_width - kernel_size//2]
        
            # get difference between spatially filtered image and frequency filtered image
            # clip the difference to eliminate the negative values
            self.difference_array = self.clip(inverse_fourier_output - spatial_filtered_image_array)
            
            # show the desired filtered image upon the radio button selection
            if (self.frequency_domain_RadioButton.isChecked()):
                # view new image
                self.filtered_image.canvas.ax.clear()
                self.filtered_image.canvas.ax.imshow(self.clip(inverse_fourier_output), interpolation = "None", cmap="gray")
                self.filtered_image.canvas.ax.axis('off')
                self.filtered_image.canvas.draw() 

            elif(self.spatial_domain_RadioButton.isChecked()):
                # view new image
                self.filtered_image.canvas.ax.clear()
                self.filtered_image.canvas.ax.imshow(self.clip(spatial_filtered_image_array), interpolation = "None", cmap="gray")
                self.filtered_image.canvas.ax.axis('off')
                self.filtered_image.canvas.draw()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def filter_difference(self):
        try:
            # plot the difference image
            self.filtered_image.canvas.ax.clear()
            self.filtered_image.canvas.ax.imshow(self.difference_array, interpolation = "None", cmap="gray")
            self.filtered_image.canvas.ax.axis('off')
            self.filtered_image.canvas.draw()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()

    def denoising(self):
        try:
            # image original size
            image=Image.open(self.path).convert('L') 
            original_image_width=image.width
            original_image_height=image.height
            original_image_array=np.asarray(image)
            mask_array = np.full((original_image_height, original_image_width), 1)

            # create the mask filter with black boxes covering the known noise coordinates in the fourier transform of the image with pattern
            mask_array[474:500,316:335] =0
            mask_array[479:490,374:391] =0
            mask_array[564:575,319:337] =0
            mask_array[556:575,376:398] =0
            mask_array[443:452,323:330] =0
            mask_array[440:447,378:388] =0
            mask_array[607:614,326:332] =0
            mask_array[604:610,381:387] =0

            image_fourier_transform = self.fourier_transform(original_image_array)
            real_component =  image_fourier_transform.real
            imaginary_component =  image_fourier_transform.imag

            # multiply the image and the mask in the frequency domain
            denoised_image_array_fourier_transform = image_fourier_transform * mask_array

            
            # restore the image with inverse fourier
            denoised_image_array = ((np.fft.ifft2(np.fft.ifftshift(denoised_image_array_fourier_transform))))

            # plot the denoised image
            real = denoised_image_array.real
            imaginary = denoised_image_array.imag

            denoised_image_magnitude = np.sqrt((real ** 2) + (imaginary ** 2))

            self.filtered_image.canvas.ax.clear()
            self.filtered_image.canvas.ax.imshow(denoised_image_magnitude, interpolation = "None", cmap="gray" )
            self.filtered_image.canvas.ax.axis('off')
            self.filtered_image.canvas.draw()
        except :                 #pop error message in case of error 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('An error has occured!')
            msg.setWindowTitle("Error")
            msg.exec_()
        


        

def mergeSort(arr):
	if len(arr) > 1:

		# Finding the mid of the array
		mid = len(arr)//2

		# Dividing the array elements into 2 halves
		L = arr[:mid]
		R = arr[mid:]

		# Sorting the first half
		mergeSort(L)

		# Sorting the second half
		mergeSort(R)

		i = j = k = 0

		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1







app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
