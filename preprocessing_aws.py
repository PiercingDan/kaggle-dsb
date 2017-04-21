########################################################################################################################
# UTDST
# Data Science Bowl Competition
# Preprocessing Functions
# Cem Anil (2017)
# Edited by Danny Luo (2017) for AWS
# Functions are heavily inspired by the Full Preprocessing Tutorial on Kaggle Kernels by Guido Zuidhof
# Also used the visualization function from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
########################################################################################################################
# Imports

import os
import numpy as np
import pandas as pd
#import tensorflow as tf #tensorflow makes import slow.... not necessary here
import scipy.ndimage
import matplotlib.pyplot as plt
import skimage
import skimage.measure
from skimage import measure
from io  import BytesIO
import dicom
import time
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
#import cPickle as pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

########################################################################################################################
# Paths (change root path before using)
root_path = '/Users/cemanil/Projects/UTDST_DataScienceBowl'
raw_data_path = os.path.join(root_path, 'data', 'raw_data')
preprocessed_data_path = os.path.join(root_path, 'data', 'preprocessed_data')
results_path = os.path.join(root_path, 'results')

########################################################################################################################
# Functions for Preprocessing Data

def preprocess(path):
    """
    Return the slices given the path to a scan (in dicom format)
    :param path: (string) path to the dicom scan
        e.g. '/root_path/sample_images/0a0c32c9e08cc2ea76a71649de56be6d'
    :return: preprocessed 3D image
    """
    # Load scans
    scan = load_scan(patient_path)

    # Obtain the 3D image
    image = get_3d_image(scan)

    # Rescale the image
    image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])

    # Normalize the image
    image_normalized = normalize_3d_image(image_rescaled)

    # Zero center the image
    image_zero_centered = zero_center_3d_image(image_normalized)

    return image_zero_centered

def load_scan(path_or_objs, aws=False):
    #path or AWS s3 object summaries retrieved by resource.bucket.objects
    if aws: 
        slices = [dicom.read_file(BytesIO(obj.get()['Body'].read())) for obj in path_or_objs]
    else:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices



def get_3d_image(slices):
    # Stack the slices to get a 3D image (in numpy array form)
    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    # Return the image
    return (np.array(image, dtype=np.int16))


def resample_3d_image(image, scan=None, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    # Determine how much to scale
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # Resample
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def normalize_3d_image(image):
    # Normalize the image
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image


def zero_center_3d_image(image,
                         scan_dict=None, use_scan_dict=False, add_to_scan_dict=False):
    # Zero center the image
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image


def preprocess(path_or_objs, aws=False):
    """
    Return the slices given the path to a scan (in dicom format)
    :param path: (string) path to the dicom scan
        e.g. '/root_path/sample_images/0a0c32c9e08cc2ea76a71649de56be6d'
    :return: preprocessed 3D image
    """
    # Load scans
    scan = load_scan(path_or_objs, aws)

    # Obtain the 3D image
    image = get_3d_image(scan)

    # Rescale the image
    image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])

    # Normalize the image
    image_normalized = normalize_3d_image(image_rescaled)

    # Zero center the image
    image_zero_centered = zero_center_3d_image(image_normalized)

    return image_zero_centered

########################################################################################################################
# if __name__ == "__main__":
#
#     # Test the load_scan(path) function
#     test_load_scan = False
#     if test_load_scan:
#         print 'TESTING LOADING SCAN'
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#         scan = load_scan(patient_path)
#
#     # Test the get_3d_image(...) function
#     test_get_3d_image = False
#     if test_get_3d_image:
#         print 'TESTING GETTING 3D IMAGES'
#         # Get patient paths
#         patient_path_1 = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#         patient_path_2 = os.path.join(raw_data_path, 'sample_images', '0a38e7597ca26f9374f8ea2770ba870d')
#
#         # Load scans
#         scan_1 = load_scan(patient_path_1)
#         scan_2 = load_scan(patient_path_2)
#
#         # Test the function
#         image_1 = get_3d_image(scan_1)
#         print image_1.shape
#
#     # Test the resample 3d image function
#     test_resample_3d = False
#     if test_resample_3d:
#         print 'TESTING RESAMPLING'
#         # Get patient path
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#
#         # Load scans
#         scan = load_scan(patient_path)
#
#         # Obtain the 3D image
#         image = get_3d_image(scan)
#
#         # Rescale the image
#         image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])
#         print 'previous shape: ', image.shape
#         print 'current shape: ', image_rescaled.shape
#
#     # Test the normalize_3d_image function
#     test_normalize_3d_image = False
#     if test_normalize_3d_image:
#         print 'TESTING NORMALIZATION'
#         # Get patient path
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#
#         # Load scans
#         scan = load_scan(patient_path)
#
#         # Obtain the 3D image
#         image = get_3d_image(scan)
#
#         # Rescale the image
#         image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])
#
#         # Normalize the image
#         image_normalized = normalize_3d_image(image_rescaled)
#         print 'previous max: ', np.max(image_rescaled)
#         print 'new max: ', np.max(image_normalized)
#
#     # Test the zero_center_normalize_3d_image = True
#     test_zero_center = False
#     if test_zero_center:
#         print 'TESTING ZERO CENTERING'
#         # Get patient path
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#
#         # Load scans
#         scan = load_scan(patient_path)
#
#         # Obtain the 3D image
#         image = get_3d_image(scan)
#
#         # Rescale the image
#         image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])
#
#         # Normalize the image
#         image_normalized = normalize_3d_image(image_rescaled)
#
#         # Zero center the image
#         image_zero_centered = zero_center_3d_image(image_normalized)
#         print 'previous mean: ', np.mean(image_normalized)
#         print 'current mean: ', np.mean(image_zero_centered)
#
#     # Test the batch_from_images
#     test_batch_from_images = False
#     if test_batch_from_images:
#         # Get patient paths
#         patient_path_1 = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#         patient_path_2 = os.path.join(raw_data_path, 'sample_images', '0a38e7597ca26f9374f8ea2770ba870d')
#
#         # Load scans
#         scan_1 = load_scan(patient_path_1)
#         scan_2 = load_scan(patient_path_2)
#
#         # Get the images
#         image_1 = get_3d_image(scan_1)
#         image_2 = get_3d_image(scan_2)
#
#         # Resample the images
#         image_rescaled_1, new_spacing = resample_3d_image(image_1, scan=scan_1, new_spacing=[1, 1, 1])
#         image_rescaled_2, new_spacing = resample_3d_image(image_2, scan=scan_2, new_spacing=[1, 1, 1])
#
#         # Create a batch
#         image_list = [image_rescaled_1, image_rescaled_2]
#         batch = batch_from_images(images=image_list, target_dims=[350, 350, 350])
#         print batch.shape
#
#         # Test how long it takes to process a patient
#
#     # Test how long it takes per patient
#     test_time_per_patient = False
#     if test_time_per_patient:
#         print 'TESTING HOW LONG IT TAKES TO PREPROCESS PER PATIENT'
#         # Get patient path
#         time0 = time.time()
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#
#         # Load scans
#         scan = load_scan(patient_path)
#
#         # Obtain the 3D image
#         image = get_3d_image(scan)
#
#         # Rescale the image
#         image_rescaled, new_spacing = resample_3d_image(image, scan=scan, new_spacing=[1, 1, 1])
#
#         # Normalize the image
#         image_normalized = normalize_3d_image(image_rescaled)
#
#         # Zero center the image
#         image_zero_centered = zero_center_3d_image(image_normalized)
#         time1 = time.time()
#         print 'time elapsed: ', time1 - time0
#
#     # Test full preprocessing function
#     test_full_preprocessing = True
#     if test_full_preprocessing:
#         # Get patient path
#         patient_path = os.path.join(raw_data_path, 'sample_images', '0a0c32c9e08cc2ea76a71649de56be6d')
#
#         # Do preprocessing
#         print 'preprocessing data...'
#         processed_image = preprocess(patient_path)
#
#         # Save the processed image
#         # file_name = os.path.join(preprocessed_data_path, 'sample_images_preprocessed',
#         #                          '0a0c32c9e08cc2ea76a71649de56be6d.pkl')
#         # pickle.dump(processed_image, open(file_name, 'wb'))
#
#         # Get histogram of pixels
#         # plt.hist(processed_image.flatten(), bins=80, color='c')
#         # plt.xlabel("Hounsfield Units (HU)")
#         # plt.ylabel("Frequency")
#         # plt.show()
#
#         # Inspect the image
#         print 'visualizing data...'
#         plot_3d(processed_image, threshold=0.2)
#         print 'done'


