from keras.callbacks import EarlyStopping, History, ModelCheckpoint

from keras.layers import Input, Embedding, Dense, MaxPooling1D, Conv1D, Dropout
from keras.layers import Flatten, SpatialDropout1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence

import json
with open("encoded_labels.json", "rb") as infile:
    label_info = json.load(infile)


with open("5_ed.json", "rb") as infile:
    encoding_dict = json.load(infile)
    max_features = max(encoding_dict.values())+1

max_length = 550
embedding_dims = 256



x = Input(shape=[max_length, ])

nn = Embedding(max_features, embedding_dims,
                       input_length=max_length, dropout=0.2)(x)


nn = Conv1D(32, 128, activation="relu", padding="valid")(nn)
nn = MaxPooling1D(pool_length=64, stride=32)(nn)
nn = Flatten()(nn)
nn = Dense(1024, activation="relu")(nn)
nn = Dropout(0.2)(nn)
outputs = []

li = []

for i in sorted([int(i) for i in label_info.keys()]):
    li.append(len(label_info[str(i)].keys()))


for index, node in enumerate(li):
    outputs.append(Dense(node, activation="softmax", name=(str(index)))(nn))

model = Model(inputs=[x], outputs=outputs)

opt = RMSprop(lr=0.001)

model.compile(
    optimizer=opt,
    metrics=["accuracy"],
    loss="sparse_categorical_crossentropy"
)

model.summary()

model.load_weights("f0_model.pkl")
model.save("model.h5")
