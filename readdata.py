import os
from skimage import io
from PIL import Image
import numpy as np

__globalDirectory = "/home/dewitt/Downloads/"
#__globalDirectory = "/home/ubuntu/"
__folder = "Data/SummerCampData/"
#__folder = "SummerCampData/"

__taskName = "D2/SCT1"
__labelsFile = "labels.csv"

__currentTaskFolder = __globalDirectory + __folder + __taskName


def make_dir(training, difficulty, task_id, global_directory):
    if training:
        dir = 'TrainingData'
    else:
        dir = 'TestingData'
    dir = global_directory + __folder + dir +'/' + 'D' + str(difficulty) + '/' + 'SCT' + str(task_id)
    return dir


def read_data(training=True, difficulty=1, task_id=1, global_directory=__globalDirectory):
    task_folder = make_dir(training, difficulty, task_id, global_directory)
    os.chdir(task_folder)
    labels = np.array(list(map(lambda line: np.array(list(map(int, line.split(',')))), open(__labelsFile, 'r').readlines())))
    #labels = labels[:20]
    N = len(labels)
    images = []

    for i in range(N):
        images.append(io.imread(task_folder +'/' + ('%.7d' % i) + '.bmp'))
    images = np.array(images)
    """for i in range(N):
        images[i] = images[i].T
        print(images[i].shape)"""
    return images, labels