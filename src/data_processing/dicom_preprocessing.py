from glob import glob

import numpy as np
import pydicom


def remove_air(pixels):
    pixels[pixels <= -1000] = 0
    return pixels


def to_hu(files: list[pydicom.FileDataset]):
    pixels = np.stack([file.pixel_array.astype(np.int16).copy() for file in files])
    pixels = remove_air(pixels)

    for i in range(len(files)):

        if files[i].RescaleSlope != 1:
            pixels[i] *= np.int16(files[i].RescaleSlope)
        pixels[i] += np.int16(files[i].RescaleIntercept)

    return pixels


def load_dicom(path) -> list[pydicom.FileDataset]:

    files_path = glob(f"{path}/*.dcm")
    files_path.sort()
    files = []
    files.append([pydicom.dcmread(f) for f in files_path])
    return files


def preprocess(path):

    files = load_dicom(path)
    pixels = to_hu(files)
    return files, pixels
