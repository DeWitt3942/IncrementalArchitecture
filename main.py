def f1():
    s = __import__('sandbox_new')
    res = [
    #s.model_statistics_on_task(1, 1, 15),
    s.model_statistics_on_task(1, 2, 20),
    """s.model_statistics_on_task(2, 1, 30),
    s.model_statistics_on_task(2, 2, 40),
    s.model_statistics_on_task(3, 1, 20),
    s.model_statistics_on_task(3, 2, 30),
    s.model_statistics_on_task(4, 1, 20),
    s.model_statistics_on_task(4, 2, 30),
    s.model_statistics_on_task(5, 1, 20),
    s.model_statistics_on_task(5, 2, 30),
    s.model_statistics_on_task(6, 1, 30),
    s.model_statistics_on_task(6, 2, 40)]"""
    ]
    print(res)
    out = open("output.txt","w")
    out.write(str(res))
    out.close()

    #s.train_network_ui(1, 2, 10)
f1()
## 2.3818
#f1()
def g():
    import network_scaling as nn
    nn.INPUT_SHAPE = (64,64,3)
    nn.add_new_task(11, False)
    nn.add_new_task(11, False)
    print(len(nn.tasks[0].outputs))
    print(len(nn.tasks[1].inputs))
#f1()


def p():

    i1 = Input((1,))
    i2 = Input((1,))
    p2 = ScalingLayer()(i2)
    m = merge([i1,p2])
    X1, X2 = np.array([i for i in range(10)]), np.array([i for i in range(10)])
    Y = np.array([0 for i in range(10)])
    model = Model(input = [i1, i2], output = [m])
    model.compile(optimizer = 'adam', loss = 'mse')
    model.fit([X1, X2], Y, nb_epoch=10000)
    for layer in model.layers:
        print(layer.get_weights())

def f3():
    import numpy as np
    import keras.backend as K
    from keras.models import Model, Sequential
    from keras.layers import Layer, Input, Embedding, Dense, merge, Convolution2D, Activation, Dropout, MaxPooling2D, \
        Flatten, UpSampling2D

    class ScalingLayer(Layer):
        def __init__(self, **kwargs):
            super(ScalingLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.dim = input_shape
            # initial_weight_value = np.random.random((input_dim, input_dim))
            # self.output_dim = input_dim
            self.W = K.variable(np.random.random())
            self.trainable_weights = [self.W]

        def call(self, x, mask=None):
            return self.W * x  #

        def get_output_shape_for(self, input_shape):
            return input_shape

    inp = Input((64, 16, 1))
    def resident_block(input_block=None, pooling=0, dropout=False, f=False, last_activation=None):
        input = input_block
        if last_activation is not None:
            print('Residual..')
            down_scale = pool_count - last_activation[1]

            last_activation = last_activation[0]
            for i in range(down_scale):
                last_activation = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(last_activation)
            input = merge([input, last_activation], method='concat')
        print(input)
        block = Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th')(input)
        #block_output = [block, pool_count]
        block = Activation('relu')(block)
        if dropout:
            block = Dropout(0.3)(block)
        """for i in range(pooling):
            block = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(block)"""
        #pool_count += pooling
        return block

    def construct_layer_by_ccn_id(id, layer_input, residental_input=None):
        if id == 0:
            return resident_block(layer_input, 1, last_activation=residental_input)
        if id == 1:
            return resident_block(layer_input, 0, True, last_activation=residental_input)
        if id == 2:
            return resident_block(layer_input, 0, True, last_activation=residental_input)
        if id == 3:
            return resident_block(layer_input, last_activation=residental_input)
        if id == (10 - 1):
            return Flatten()(layer_input), None
        raise Exception('OMG WTF')

    conv = construct_layer_by_ccn_id(0, inp)
    input_2 = Input((64,16,1))
    input_2_ = ScalingLayer()(input_2)
    m = merge([input_2, conv])
    inp = m
    conv = construct_layer_by_ccn_id(1, inp)
#f1()

def f2():
    import numpy as np
    import keras.backend as K
    from keras.models import Model, Sequential
    from keras.layers import Layer, Input, Embedding, Dense, merge, Convolution2D, Activation, Dropout, MaxPooling2D, \
        Flatten, UpSampling2D


    class ScalingLayer(Layer):
        def __init__(self, **kwargs):

            super(ScalingLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.dim = input_shape
            #initial_weight_value = np.random.random((input_dim, input_dim))
            #self.output_dim = input_dim
            self.W = K.variable(np.random.random())
            self.trainable_weights = [self.W]

        def call(self, x, mask=None):
            return self.W * x#

        def get_output_shape_for(self, input_shape):
            return input_shape



    inp = Input((64,16,3))
    inp_ = ScalingLayer()(inp)
    #inp_ = inp
    inp2 = Input((64,16,3), name='Input 2')
    inp2_ = ScalingLayer(name='Scaled input 2')(inp2)

    m = merge([inp_, inp2_])
    #print(m._keras_shape)
    #inp_ = Flatten()(inp_)
    #m = Dense(5)(inp_)
    m = Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th')(m)
    model = Model(input = [inp, inp2], output = [m])


    class ScalingLayer(Layer):
        def __init__(self, output_dim , **kwargs):
            self.output_dim = output_dim
            super(ScalingLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            input_dim = input_shape[1]
            #initial_weight_value = np.random.random((input_dim, input_dim))
            self.output_dim = input_dim
            self.W = K.variable(np.random.random())
            self.trainable_weights = [self.W]

        def call(self, x, mask=None):
            return self.W * x#K.dot(x, self.W)

        def get_output_shape_for(self, input_shape):
            return (input_shape[0], self.output_dim)

    inp = Input((5,))
    out = ScalingLayer((5,))(inp)