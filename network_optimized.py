import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.models import Model, Sequential
from keras.layers import Layer, Input, Embedding, Dense, merge, Convolution2D, Activation, Dropout, MaxPooling2D, \
    Flatten, UpSampling2D

tasks = []
tasks_alive = 0
LAYERS_COUNT = 5
INPUT_SHAPE = None


class SingleTaskModel:
    def __init__(self, input_shape, output_shape, independent = False):
        
        self.first = (tasks_alive == 0)
        self.task_id = len(tasks)

        self.independent = independent or (self.task_id == 0)
        self.outputs = []
        self.input = None
        self.inputs = []
        self.pool_count = 0
        self.model = None
        self.alive = True

        self.conv = None
        self.output = None
        self.construct_model(input_shape)
        self.add_dense_layers(output_shape)
        self.compile()

    def resident_block(self, input_block=None, pooling=0, dropout=False, f=False, last_activation=None):
        input = input_block
        if last_activation is not None:
            print(last_activation)
            down_scale = self.pool_count - last_activation[1]

            last_activation = last_activation[0]
            for i in range(down_scale):
                last_activation = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(last_activation)
            input = merge([input, last_activation], method='concat')
        #print(input)
        block = Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th')(input)
        # block = merge([block, input])
        block_output = [block, self.pool_count]
        block = Activation('relu')(block)
        # block = merge([block, input])
        if dropout:
            block = Dropout(0.3)(block)
        for i in range(pooling):
            block = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')(block)
        self.pool_count += pooling
        return block, block_output

    def construct_layer_by_ccn_id(self, id, layer_input, residental_input=None):
        if id == 0:
            return self.resident_block(layer_input, 1, last_activation=residental_input)
        if id == 1:
            return self.resident_block(layer_input, 0, True, last_activation=residental_input)
        if id == 2:
            return self.resident_block(layer_input, 0, True, last_activation=residental_input)
        if id == 3:
            return self.resident_block(layer_input, last_activation=residental_input)
        if id == (LAYERS_COUNT - 1):
            return Flatten()(layer_input), None
        raise Exception('OMG WTF')

    def get_own_layer(self, layer_id):
        if layer_id < len(self.outputs):
            return self.outputs[layer_id]
        return []

    def get_all_outputs(self):
        return self.outputs

    @staticmethod
    def get_output_shape(layer_id):
        # print("Coming hear'")
        shapes = [(32, 16, 1), (64, 16, 1), (64, 16, 1), (64, 16, 1)]
        return shapes[layer_id]

    @staticmethod
    def add_inputs(old, new):
        ans = []
        for id in range(len(old)):
            u = old[id] if isinstance(old[id], list) else [old[id]]
            v = new[id] if isinstance(new[id], list) else [new[id]]
            ans.append(u + v)
        return ans

    @staticmethod
    def add_arrays(arr1, arr2):
        return [(arr1[i] + arr2[i]) for i in range(min(len(arr1), len(arr2)))]

    def construct_model(self, input_size):
        self.g_input = Input(input_size, name='image_input')
        self.inputs = [self.g_input]
        self.conv = MaxPooling2D((2, 2), dim_ordering='tf')(MaxPooling2D(pool_size=(1, 2))(self.g_input))
        for i in range(LAYERS_COUNT):
            print(i, self.conv)
            layer_input = self.conv
            if not self.first and i > 0 and not self.independent:
                inp = Input(tuple([i for i in SingleTaskModel.get_output_shape(i - 1)]))
                self.inputs.append(inp)
                layer_input = merge([inp] + [self.conv], mode='sum')
            self.conv, _ = self.construct_layer_by_ccn_id(i, layer_input)
            self.outputs.append(self.conv)

    def add_dense_layers(self, output_size):
        self.task_output = Dense(256)(self.conv)
        self.task_output = Activation('tanh')(self.task_output)
        self.task_output = Dropout(0.5)(self.task_output)
        self.output = Dense(output_size, activation='softmax')(self.task_output)

    def compile(self):
        #print(self.inputs)
        #print(self.output)
        self.model = Model(input=self.inputs, output=[self.output])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        #self.model.summary()

    def make_input(self, X):

        if self.first or self.independent:
            return X
        input_ = []
        for task_id in range(self.task_id):
            if tasks[task_id].alive:
                print('Its alive!')
                model = Model(input=tasks[task_id].inputs, output=tasks[task_id].outputs)
                if not tasks[task_id].independent:
                    vals = model.predict([X] + input_)[:-1]
                else:
                    vals = model.predict([X])[:-1]
                if input_ == []:
                    input_ = vals
                else:
                    input_ = vals
                    # input_ = SingleTaskModel.add_arrays(input_, vals)
        return [X] + input_

    def fit(self, x, Y, epoch, safe=True):
        if not safe:
            return self.model.fit(self.make_input(x), Y, nb_epoch=epoch)
        else:
            return self.model.fit(self.make_input(x), Y, nb_epoch=epoch, callbacks=[stoppingCallBack])

    def predict(self, x):
        return self.model.predict(self.make_input(x), verbose=1)

    def summary(self):
        return self.model.summary()

    def kill(self):
        global tasks_alive
        tasks_alive-=1
        self.alive = False



def construct_model(inp_shape):
    global INPUT_SHAPE
    INPUT_SHAPE = inp_shape


def add_new_task(output_shape, independent=False):
    global tasks_alive
    new_task = SingleTaskModel(INPUT_SHAPE, output_shape, independent)
    tasks.append(new_task)

    tasks_alive += 1


def get_model(task_id=None):
    if task_id is None:
        return tasks[-1]
    return tasks[task_id]


def clear():
    global tasks
    tasks = []


from keras.callbacks import EarlyStopping

stoppingCallBack = EarlyStopping(monitor='loss', patience=3)