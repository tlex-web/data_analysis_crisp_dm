import glob
import shutil

files = [f for f in glob.glob('*.ipynb')]

with open('./crisp_dm_final.ipynb', 'wb') as output_file:
    for file in files:
        with open(file, 'rb') as f:
            shutil.copyfileobj(f, output_file)
