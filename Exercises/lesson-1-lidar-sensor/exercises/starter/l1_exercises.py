# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    ri[ri<0]=0.0
    # map value range to 8bit
    ri_scaled = np.amax(ri[:,:,1])/2 * ri[:,:,1] * 255 / (np.amax(ri[:,:,1]) -  np.amin(ri[:,:,1]))
    img_intensity = ri_scaled.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(ri.shape[1]/8)      # 45deg corresponds to 1/8th of a circle
    center = int(ri.shape[1]/2)
    img_intensity = img_intensity[:, center-deg45: center+deg45]

    cv2.imshow('intensity image', img_intensity)
    cv2.waitKey(0)


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    # compute vertical field-of-view from lidar calibration 
    vfov_rad = calib_lidar.beam_inclination_max - calib_lidar.beam_inclination_min
    # compute pitch resolution and convert it to angular minutes
    pitch_resolution = vfov_rad * 180/ np.pi /64 /60 # divided by 64 led. multiplied by 60 minutes in 1 degree
    print('The vertical pitch resolution is:' + str(pitch_resolution))

# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")    

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = 0
    for obj in frame.laser_labels:
        if obj.type == obj.TYPE_VEHICLE:
            num_vehicles += 1
    print("number of labeled vehicles in current frame = " + str(num_vehicles))