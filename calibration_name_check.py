import zipfile
with zipfile.ZipFile('CalibrationPhantomData.zip') as z:
    names = [n for n in z.namelist() if 'Water_Phantom' in n]
    z.extractall('data/', members=names)