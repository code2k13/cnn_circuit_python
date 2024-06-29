# Running Convolutional Neural Networks on CircuitPython

This repository contains code for running Convolutional Neural Networks (CNNs) on CircuitPython.  It contains code to train models using Tensorflow on computers and convert them to CircuitPython.
For more information, please visit: https://ashishware.com/2024/06/29/pipico_digit_classification_cnn/

Here's a table detailing the files and their functions:

| **File Name**         | **Description**                                                                                                            |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------|
| `code.py`             | Main program that runs on your CircuitPython board. Copy this file to the root directory of the CircuitPython board.        |
| `mnist_clf.mpy`       | Module that contains the converted model. Copy this to the `/lib` folder of the CircuitPython board.                        |
| `mnist_clf.py`        | Plain CircuitPython implementation of the above file. Does not load when imported in CircuitPython due to memory limitations on the Raspberry Pi Pico (not tested with other boards). |
| `mnist_clf_pc.py`     | A version of the converted model that can be tested and debugged on a normal PC. Uses a tiny subset of `numpy` features.     |
| `mnist_clf_template.py` | Template file used by `training_and_export.ipynb` notebook. Utilized during the generation or updating of `mnist_clf.mpy`.  |
| `libs_list.txt`       | Contains a list of all libraries present in the `/lib` folder of the Raspberry Pi Pico during testing.                       |
| `training_and_export.ipynb`    | Jupyter notebook which contains code to train and export a CNN model to CircuitPython. A live version of the notebook can be found here: https://www.kaggle.com/code/finalepoch/image-classification-using-cnns-on-circuitpython                       |
