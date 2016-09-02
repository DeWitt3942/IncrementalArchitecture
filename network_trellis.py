from  keras.models import Sequential, Model
from keras.layers import Convolution2D, Dense, Layer, Input, MaxPooling2D, merge, UpSampling2D, Activation, Flatten
import numpy as np
layers = 5
scales = 2
channels = 3

trellis = None
trellis_end = None
task_outputs = []

def valid(l, s, c):
    return 0 <= l < layers and 0 <= s < scales and 0 <= c < channels


def get_previous_nodes(l, s, c):
    inputs = []
    for ds in range(-1, 2):
        for dc in range(-1, 2):
            if valid(l - 1, s + ds, c + dc):
                inputs.append((l-1, s+ds, c+dc))
    if (l==0) or (l==layers-1):
        for ds in range(-1, 1):
            for dc in range(-1, 1):
                if valid(l, s+ds, c+dc) and not(ds == dc == 0):
                    inputs.append((l, s+ds, c+dc))
    return inputs


def get_transformed_node(from_shape, to_shape):
    l0, s0, c0 = from_shape
    l, s, c = to_shape
    node = trellis[l0, s0, c0]
    if s0>s:
        node =  UpSampling2D()(node)
    if s0<s:
        node = MaxPooling2D()(node)
    return node

def construct_layer(l, s, c):
    layer_input = None
    prev = get_previous_nodes(l, s, c)
    """for l0, s0, c0 in prev:
        prev_node = None
        if s0>s:
            prev_node = UpSampling2D()(trellis[l0, s0, c0])
        elif s0<s:
            prev_node = MaxPooling2D()(trellis[l0, s0, c0])
        else:
            prev_node = trellis[l0, s0, c0]
        if layer_input is None:
            layer_input = prev_node
        else:
            layer_input = merge([layer_input, prev_node], mode='concat', concat_axis=1
    """
    print('Getting inputs for ',l, s, c)
    inputs = list(map(lambda node: get_transformed_node(node, (l, s, c)), prev))
    if len(inputs)==1:
        layer_input = inputs[0]
    else:
        layer_input = merge(list(map(lambda node: get_transformed_node(node, (l, s, c)), prev)) , mode='concat', concat_axis=1)
    node = Convolution2D(c+1, 3, 3, border_mode='same')(layer_input)
    return Activation('relu', name='layer-'+str(l)+'-'+str(s)+'-'+str(c))(node)


def construct_model(input_shape, L=layers, S=scales, C=channels):
    global trellis, layers, scales, channels, trellis_end
    layers = L
    scales = S
    channels = C

    trellis = np.empty((layers, scales, channels), dtype=object)

    trellis[0, 0, 0] = Input(input_shape, name='trellis_input')
    for l in range(layers):
        for s in range(scales):
            for c in range(channels):
                if l+s+c>0:
                    trellis[l, s, c] = construct_layer(l, s, c)
    trellis_end = Flatten()(trellis[L-1, S-1, C-1])


def add_new_task(output_shape):
    task_output = Dense(output_shape*2)(trellis_end)

    task_output = Activation('relu')(task_output)
    task_output = Dense(output_shape)(task_output)
    task_outputs.append(task_output)

    return task_output


def get_model():
    model = Model(input=[trellis[0, 0, 0]], output=task_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    #model.summary()
    return model

