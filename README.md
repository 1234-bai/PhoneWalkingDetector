# PhoneWalkingDetector

## Install

1. Create a conda virtual environment.

   ```shell
   conda create -n pwd python=3.8
   conda activate pwd
   ```

2. Install dependencies

   ```shell
   cd Phone_Walking_Detector
   pip install -r requirements
   ```

3. Install cython_bbox

   ```shell
   pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
   ```

   Then directory  `/src` come in to being. Change 12 and 13 lines of  `src\cython-bbox\src\cython_bbox.pyx`  to

   ```cython
   DTYPE = np.float64	# DTYPE = np.float
   ctypedef np.float64_t DTYPE_t # ctypedef np.float_t DTYPE_t
   ```

   Then run commands:

   ```shell
   cd src/cython-bbox
   python setup.py build develop
   ```

## Run

1. detect some images and view results

   ```shell
   cd Phone_Walking_Detector
   python detect.py --nosave --view-img --source {dir to test images}
   ```

2. UI

   ```shell
   cd Phone_Walking_Detector
   python main.py
   ```

   The detect results will be saved to `runs/result/`

