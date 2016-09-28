from readdata import read_data
from keras.models import model_from_json
import numpy as np
import utils as u
# import network as nn
import network_optimized as nn
#import  network_res as nn
## import network_basic as nn
# import network_trellis as nn

import label_classifier as _classifier
from functools import reduce

np.random.seed(1)


def number_of_outputs(filtered_labels):
    return max(map(lambda label: (label == 1).sum(), filtered_labels))


def transform_data(raw_images, raw_labels, task_id=None):
    labels = _classifier.labels_remove_twos(raw_labels)
    labels, representation = _classifier.transform_labels_with_representation(labels)
    return raw_images, labels, representation


model = None
history = None
task_dims = []
tasks_encoded = {}


def generate_trainingset(labels_count, true_labels):
    data = [np.zeros((labels_count, dim)) for dim in task_dims]
    data.append(true_labels)
    return data


def train_network(X, Y, epochs=1, train=True, filename=None, load_model=False, finetune=False, task_id=None, independent=False):
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
        X, Y, _ = transform_data(X, Y, task_id)
        print('Labels shape: ' + str(Y.shape))
        print('Labels look like this : ')
        print(Y[0])
        if not finetune:
            nn.construct_model(X[0].shape)

        nn.add_new_task(len(Y[0]), independent)
        model = nn.get_model().model

    if train:

        print('Training started')
        input_model = open(filename[0], 'w')
        input_model.write(model.to_json())
        input_model.close()
        # model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='model.png')
        print('Fitting')
        history = nn.get_model().fit(X, Y, epoch=epochs)
        task_dims.append(Y.shape[1])

        print('Training end')
        if filename is not None:
            input_model = open(filename[0], 'w')
            input_model.write(model.to_json())
            input_model.close()

            model.save_weights(filename[1], overwrite=True)


tasks = {}


def train_network_ui(task_id, difficulty, epochs=3):
    # tasks[(task_id, difficulty)] = len(tasks.keys())
    X, Y = read_data(task_id=task_id, difficulty=difficulty)
    train_network(X, Y, epochs=epochs, train=True, load_model=False, filename=['model.txt', 'weights.hdf5'],
                  task_id=task_id)
    tasks_encoded[(task_id, difficulty)] = len(nn.tasks) - 1
    old_accuracy = evaluate_accuracy(task_id, difficulty, silent=True)
    print('Old accuracy: ', old_accuracy)
    if len(nn.tasks)>1:
        buffered = nn.tasks[-1]
        nn.tasks.pop()
        train_network(X, Y, epochs=epochs, train=True, load_model=False, filename=['model.txt', 'weights.hdf5'],
                      task_id=task_id, independent=True)
        new_accuracy = evaluate_accuracy(task_id, difficulty, silent=True)
        print('New accuracy: ', new_accuracy)
        if new_accuracy<old_accuracy:
            nn.tasks[-1] = buffered
        old_accuracy = new_accuracy
    if old_accuracy<0.9:
        nn.tasks[-1].kill()

        
def get_original_output(task_id, difficulty):
    X, Y = read_data(training=False, task_id=task_id, difficulty=difficulty)
    Y = _classifier.labels_remove_twos(Y)
    representation = _classifier.find_representation(Y)
    raw_predicted = nn.get_model(tasks_encoded[(task_id, difficulty)]).predict(X)
    predicted = _classifier.get_normal_output(raw_predicted, representation)

    def transform_single_label(label, how):
        i = 0
        new = []
        for id in range(len(how)):
            if how[id]==2:
                new.append(0)
            else:
                new.append(label[i])
                i  += 1
        return how
    return list(map(lambda label: transform_single_label(label, Y[0]), predicted))


def accuracy(predicted, original, raw, errors=False):
    acc = 0
    for id in range(predicted.shape[0]):
        if all(predicted[id] == original[id]):
            acc += 1
        else:
            if errors:
                print(str(predicted[id]) + " " + str(original[id]) + " raw: " + str(list(map(u.round3, raw[id]))))
    return acc * 1.0 / predicted.shape[0]


def evaluate_accuracy(task_id, dificulty, errors=False, outputFile='result.txt', silent=False):
    if not silent:
        print('reading and processing testing data')
    X, Y = read_data(training=False, task_id=task_id, difficulty=dificulty)
    if not silent:
        print("Getting representation")
    Y = _classifier.labels_remove_twos(Y)
    representation = _classifier.find_representation(Y)

    if not silent:
        print('read')
        print('predicting..')

    raw_predicted = nn.get_model(tasks_encoded[(task_id, dificulty)]).predict(X)
    
    predicted = _classifier.get_normal_output(raw_predicted, representation)
    acc = accuracy(predicted, Y, raw_predicted, errors)
    print('Accuracy %.4f' % acc)
    return acc


def model_statistics_on_task(task_id, difficulty, epochs):
    # train for 1
    X_, Y = read_data(task_id=task_id, difficulty=difficulty)
    X_, Y, _ = transform_data(X_, Y, task_id)
    nn.construct_model(X_[0].shape)
    nn.add_new_task(len(Y[0]))

    model = nn.get_model().model
    X = nn.get_model().make_input(X_)

    tasks_encoded[(task_id, difficulty)] = len(nn.tasks) - 1
    acc_history = []
    for epoch in range(epochs + 1):
        print('Epoch #', epoch)
        if epoch > 0:
            model.fit(X, Y, nb_epoch=1)
        acc_history.append(evaluate_accuracy(task_id, difficulty, silent=True))
    return acc_history


def reset():
    global tasks_encoded
    tasks_encoded = {}
    nn.clear()


while False:
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
