import pydicom
import matplotlib.pyplot as plt
from glob import glob

def extract_dicom_info(data: pydicom.FileDataset):
    meta_data = data
    res = {
        "PatientName": meta_data.PatientName,
        "PatientID": meta_data.PatientID,
        "PatientAge": meta_data.PatientAge,
        "PatientSex": meta_data.PatientSex,
        "Modality": meta_data.Modality
    }

    
    if 'PixelData' in meta_data:
        rows = int(meta_data.Rows)
        cols = int(meta_data.Columns)
        res['image_size'] = (rows, cols)

        if 'PixelSpacing' in meta_data:
            res['PixelSpacing'] = meta_data.PixelSpacing

    return res

def visualize_dicom(data):
    if 'PixelData' not in data:
        print("No pixel data found in DICOM file.")
        return
    plt.imshow(data.pixel_array,cmap=plt.cm.bone)
    plt.show()

i = 0 
print("Visualizing DICOM files:")
x = glob('./data/raw/*.dcm')
print(f"Found {len(x)} DICOM files.")
for file in x:
    print(f"Processing file: {file}")
    if i == 5:
        break
    i += 1
    dicom_data = pydicom.dcmread(file)
    dicom_info = extract_dicom_info(dicom_data)
    [print(f"{key}: {value}") for key, value in dicom_info.items()]
    print()
    visualize_dicom(dicom_data)