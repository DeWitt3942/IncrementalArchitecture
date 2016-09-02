import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.models import Model, Sequential
from keras.layers import Layer, Input, Embedding, Dense, merge, Convolution2D, Activation, Dropout, MaxPooling2D, Flatten, UpSampling2D
kernels = []
filters = [64]

g_input = None
conv = None
outputs = []

columns = []
def freeze_a_layer(layer):
    layer.updates = []
    layer.params = []
    return layer


"""def construct_convolutional_column(input_size, old_columns = None, task_id=0):
    column = []
    for single_layer_id in range(len(structure.layers)):
        input_for_layer = structure.layer[single_layer_id-1]
        if (old_columns):
            input_for_layer = merge([old_columns.layers])
"""


def tfh(f = True):
    if f:
        return 'tf'
    else:
        return 'th'


pool_count = 0
layers_pool = []

def resident_block(input_block=None, pooling=0, dropout=False, f=False, last_activation=None):
    global pool_count
    input = input_block
    if last_activation is not None:
        print(last_activation)
        down_scale = pool_count - last_activation[1]

        last_activation = last_activation[0]
        for i in range(down_scale):
            _last_activation = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')
            layers_pool.append(last_activation)
            last_activation = _last_activation(last_activation)

        input = merge([input, last_activation])
    block = Convolution2D(64, 3, 3, border_mode='same', dim_ordering=tfh(f))#(input_block)
    layers_pool.append(block)
    block = block(input_block)
    block_output = [block, pool_count]
    block = Activation('relu')(block)
    if dropout:
        block = Dropout(0.3)(block)
    for i in range(pooling):
        _block = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')#(block)
        layers_pool.append(block)
        block = _block(block)

    pool_count+=pooling
    return block, block_output


LAYERS_COUNT = 5


def construct_layer_by_ccn_id(id, layer_input, residental_input = None):
    if id==0:
        return resident_block(layer_input, 1, last_activation=residental_input)
    if id==1:
        return resident_block(layer_input, 0, True, last_activation=residental_input)
    if id == 2:
        return resident_block(layer_input, 0, True, last_activation=residental_input)
    if id == 3:
        return resident_block(layer_input, last_activation=residental_input)
    if id == (LAYERS_COUNT - 1):
        return Flatten()(layer_input), None
    raise Exception('OMG WTF')



previous = None


def construct_model(input_size):
    global conv, g_input, previous, layers_pool
    for i in range(len(layers_pool)):
        print('Freezing', layers_pool[i])
        layers_pool[i].trainable = False
    layers_pool = []
    if g_input is None:
        g_input = Input(input_size, name='image_input')
    conv = MaxPooling2D((2,2), dim_ordering='tf')(MaxPooling2D(pool_size=(1,2))(g_input))
    resid = None
    new_previous = []
    for i in range(LAYERS_COUNT):
        print(i)
        layer_input = conv
        if previous is not None:
            layer_input = merge([freeze_a_layer(previous[i]), conv], mode='concat')
        new_previous.append(layer_input)
        conv, resid = construct_layer_by_ccn_id(i, layer_input)
    previous = new_previous

def add_new_task(output_size):
    #print(conv, ' is conv')
    #if conv:
        #print(conv.__dict__)
    task_output = Dense(256)(conv)
    task_output = Activation('tanh')(task_output)
    task_output = Dropout(0.5)(task_output)
    task_output = Dense(output_size, activation='softmax')(task_output)

    global outputs
    outputs.append(task_output)


def get_model():
    print(g_input)
    model = Model(input=[g_input], output=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Trainable layers:')
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            if (layer.trainable):
                print(layer.name)
    model.summary()
    return model


def pop_layer(model):
    if not model.outputs:
        raise Exception('Empty model')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False



