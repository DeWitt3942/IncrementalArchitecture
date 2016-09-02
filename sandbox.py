from readdata import read_data
from keras.models import model_from_json
import numpy as np
import utils as u
import network as nn
# import network_trellis as nn
# import network_basic as nn
import label_classifier as _classifier
from functools import reduce

np.random.seed(1)


def number_of_outputs(filtered_labels):
    return max(map(lambda label: (label == 1).sum(), filtered_labels))


def classify_task(task_labels):
    type = reduce(
        lambda x, y: x * y
        , map(lambda label: (label == 1).sum(), task_labels)
        , 1

    )
    # print(type)
    if type == 1:
        return 'single_classification'
    return 'multiclassification'


def transform_labels(task_labels, task_class):
    if (task_class == 'single_classification'):
        epitome = task_labels[0]
        classes_ids = []
        for i in len(epitome):
            if (epitome[i] != 2):
                classes_ids.append(i)
        classes_ids.append(-1)
        labels = []
        for label_id in len(task_labels):
            for parameter in task_labels[label_id]:
                if (parameter == 1):
                    labels.append(classes_ids.index(parameter))
            if (len(labels) == label_id):
                labels.append(-1)
    return labels


def transform_data(raw_images, raw_labels, task_id=None):
    labels = _classifier.labels_remove_twos(raw_labels)
    labels, representation = _classifier.transform_labels_with_representation(labels)
    return raw_images, labels, representation


model = None
history = None
task_dims = []


def generate_trainingset(labels_count, true_labels):
    data = [np.zeros((labels_count, dim)) for dim in task_dims]
    data.append(true_labels)
    return data


def train_network(X, Y, epochs=1, train=True, filename=None, load_model=False, finetune=False):
    global model, history
    if load_model:
        print('Loading model')
        input_model = open(filename[0], 'r')
        model = model_from_json(input_model.read())

        input_model.close()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.load_weights(filename[1])
        print('Loaded(probably)')
        if finetune:
            lastLayer = None
            print(model.layers[0].__dict__)
            for layer in model.layers:
                if (layer.name == 'dense_1'):
                    nn.conv = layer
                lastLayer = layer
            import h5py

    if not load_model or finetune:
        X, Y, _ = transform_data(X, Y)
        print('Labels shape: ' + str(Y.shape))
        print('Labels look like this : ')
        print(Y[0])
        if not finetune:
            nn.construct_model(X[0].shape)

        nn.add_new_task(len(Y[0]))

        model = nn.get_model()

    if train:

        print('Training started')
        input_model = open(filename[0], 'w')
        input_model.write(model.to_json())
        input_model.close()
        # model.summary()
        print('Fitting')
        history = model.fit(X, generate_trainingset(X.shape[0], Y), nb_epoch=epochs)
        task_dims.append(Y.shape[1])

        print('Training end')
        if filename is not None:
            input_model = open(filename[0], 'w')
            input_model.write(model.to_json())
            input_model.close()

            model.save_weights(filename[1], overwrite=True)
        return history


tasks = {}


def train_network_ui(task_id, difficulty, epochs=3):
    tasks[(task_id, difficulty)] = len(tasks.keys())
    X, Y = read_data(task_id=task_id, difficulty=difficulty)
    return train_network(X, Y, epochs=epochs, train=True, load_model=False, filename=['model.txt', 'weights.hdf5'])


def accuracy(predicted, original, raw, errors):
    acc = 0
    for id in range(predicted.shape[0]):
        if all(predicted[id] == original[id]):
            acc += 1
        else:
            if (errors):
                print(str(predicted[id]) + " " + str(original[id]) + " raw: " + str(list(map(u.round3, raw[id]))))
    return acc * 1.0 / predicted.shape[0]


def evaluate_accuracy(task_id, difficulty, errors=False, outputFile='result.txt'):
    print('reading and processing testing data')
    X, Y = read_data(training=False, task_id=task_id, difficulty=difficulty)
    print("Getting representation")

    print('read')
    Y = _classifier.labels_remove_twos(Y)
    print('predicting..')
    representation = _classifier.find_representation(Y)

    if (task_id, difficulty) in tasks.keys():
        model = nn.get_model(tasks[(task_id, difficulty)])
    else:
        raise Exception('Task unknown')
        return

    raw_predicted = model.predict(X, verbose=1)
    
    predicted = _classifier.get_normal_output(raw_predicted, representation)
    acc = accuracy(predicted, Y, raw_predicted, errors)
    print('Accuracy %.4f' % acc)
    return acc


def show_model():
    # from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot
    #return SVG(model_to_dot(model).create(prog='dot', format='svg'))


if False:
    while True:
        print("task id:")
        id = int(input())
        print("difficulty")
        diff = int(input())
        print('train or test?(1/2)')
        if input() == '1':
            print('epochs')
            epochs = int(input())
            train_network_ui(id, diff, epochs)
        else:
            evaluate_accuracy(id, diff)
        print('If you want to exit -- enter 0 ; 1 -- to test last trained task')
        inp = input()
        if inp == '0':
            from keras.utils.visualize_util import plot
            import os

            os.chdir('/media/dewitt/FAB907B81E039285/GoodAI/')
            plot(model, to_file='model.png')
            raise SystemExit()
        elif inp == '1':
            evaluate_accuracy(id, diff)
