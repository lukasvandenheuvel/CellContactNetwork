## Cellpose installation on Windows

Follow these instructions to install [Cellpose](https://www.cellpose.org/) on your computer, together with other Packages you need for running the image processing in parallel. 

1. Install an [Anaconda](https://www.anaconda.com/products/individual) distribution. Choose Windows and python 3.7.
2. Clone or download this repository.
3. Open the Anaconda Prompt application.
4. Navigate to the folder where the ```environment.yml``` file is stored with ```cd path/to/folder```. To move to the M-drive, press ```M:``` and enter.
5. Run ```conda env create -f environment.yml```.
6. When all packages are installed, activate the environment with ```conda activate cellpose```.
7. You should see (```cellpose```) on the left side of the terminal line.
8. Run ```python -m cellpose``` to start up the Cellpose GUI. If it works, you're all set.

NOTE: if you are working on a Dimphna Meijer lab computer, and the above installation did not work for you (for whatever reason), there is the possibility to use a pre-installed environment on the TU Delft M-drive. *Keep in mind that using it is not recommended, because Python scripts will run significantly slower.* If you have no choice, you can activate this environment with ```conda activate M:\tnw\bn\dm\Shared\cellpose```.
