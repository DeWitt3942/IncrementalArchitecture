from readdata import read_data
import numpy as np
import utils as u

a = [1,2,3,5]

import itertools

#_, labels = read_data()




#label_dim = labels.shape[1]
#suppose class-types are solid

#type representation  <=> list of pairs (Li, Ki)

def labels_remove_twos(labels):
    return np.array(list(map(lambda label: np.array(list(filter(lambda x: x != 2, label))), labels)))

def print_labels(labels, representation = None):
    print(labels)


def count_max_number_of_ones_in_matrix(arr):
    return max(map(lambda label: (label == 1).sum(), arr))


def count_min_number_of_ones_in_matrix(arr):
    return min(map(lambda label: (label == 1).sum(), arr))


def default_representation(labels):
    return [(labels.shape[1], count_max_number_of_ones_in_matrix(labels))]

def non_default_representation(labels):
    return [(1,1)] * labels.shape[1]

def transform_labels_with_representation(labels):

    representation = find_representation(labels)
    labels_count = labels.shape[0]
    new_labels = np.zeros((labels_count, 1))
    labels_sums = np.array(list(map(lambda x: list(itertools.accumulate(x)), labels))).T
    starting_from = 0

    for class_size, class_k in representation:
        sums = labels_sums[starting_from + class_size - 1]
        if starting_from != 0:
            sums -= labels_sums[starting_from - 1]
        #print(sums)
        extra = [[(class_k - sums[label_id])*1.0/class_k for val_id in range(class_k)] for label_id in range(labels_count)]
        extra = np.array(extra)
        #print(new_labels.shape)
        #print(labels.shape)
        new_labels = np.append(new_labels, labels[:, starting_from: (starting_from + class_size)], axis=1)
        new_labels = np.append(new_labels, extra, axis=1)
        starting_from += class_size
    K = count_max_number_of_ones_in_matrix(new_labels)
    for label_id in range(labels_count):
        for char_id in range(len(new_labels[label_id])):
            new_labels[label_id, char_id]*=1.0/K
    print('Representation : ', representation)
    return new_labels[:, 1:], representation


def get_normal_output(raw_predicted, representation):
    predicted = np.zeros((raw_predicted.shape[0], sum(map(lambda rule: rule[0], representation))), dtype='int8')
    starting_from = 0
    starting_from_new = 0
    for class_size, class_k in representation:
        current_class = raw_predicted[:, starting_from:(starting_from+class_size+class_k)]
        indecies = np.array(list(map(lambda x: u.k_most(x, class_k), current_class)))
        for label_id in range(raw_predicted.shape[0]):
            for index in indecies[label_id]:
                if index < class_size:
                    predicted[label_id][starting_from_new + index] = 1

        starting_from += class_size + class_k
        starting_from_new += class_size
    return predicted



def partial_sums(labels):
    labels_sums = np.array(list(map(lambda x: list(itertools.accumulate(x)), labels))).T
    return labels_sums


def find_representation_old(labels):
    partials = partial_sums(labels)
    label_size = labels.shape[1]
    representation = []
    #print_labels(labels[:5])
    #print_labels(partials.T[:5])
    def get_representation(start):
        if start>=label_size:
            return
        if start==label_size-1:
            representation.append((1, max(partials[label_size-1] - partials[label_size-2] )))
        for i in range(label_size-1, start, -1):
            sums = partials[i]
            if start>0:
                sums-= partials[start-1]
            #print ('From ', start, ' to ', i-1, ' is ', max(sums))#
            max_s = max(sums)
            if max_s==1:
                representation.append((i-start+1, max_s))
                get_representation(i+1)
                return

        representation.append((label_size-start, count_max_number_of_ones_in_matrix(labels[start:])))
        return
    get_representation(0)
    return representation


def find_representation(labels):
    partials = partial_sums(labels)
    s = list(map(max, partials))

    label_size = labels.shape[1]
    representation = []
    increasing = False
    seps = [0]
    print(s)
    for i in range(2, label_size):
        if (s[i]> s[i-1]) and (s[i-1]==s[i-2]):
            seps.append(i)
    seps.append(label_size)
    
    for i in range(1, len(seps)):
        from_ = seps[i-1]-1
        to_ = seps[i]-1
        k = s[to_]
        if from_>0:
            k -= s[from_]
        representation.append((to_-from_, k))
    return representation


def check():

    #raw_labels = np.array([[0.0, 0.142, 0.542, 0.001, 0.0, 0.13, 0.124, 0.0, 0.0, 0.061000001]])
    #r#epresentation = [(8,1)]
    #Y, _ = tra#nsform_labels_with_representation(labels_remove_twos(Y), 4)
    _, Y = read_data(task_id=2, difficulty=2)
    Y = labels_remove_twos(Y)
    rep = find_representation(Y)
    print(rep)
    #Y = transform_labels_with_representation(Y)
    #normal = get_normal_output(raw_labels, representation)
    #print_labels(normal)


check()










