from __future__ import division
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python import keras as K
from keras_gat import GraphAttention
from keras_gat.utils import load_data, preprocess_features

# Read data
A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = load_data('cora')

# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
n_classes = Y_train.shape[1]  # Number of classes
F_ = 8                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate applied to the input of GAT layers
l2_reg = 5e-4                 # Regularization rate for l2
learning_rate = 5e-3          # Learning rate for SGD
epochs = 100                 # Number of epochs to run for
es_patience = 200             # Patience fot early stopping

print(X.shape)
print(A.shape)
print(Y_train.shape)

l2 = K.regularizers.l2

# Preprocessing operations
X = preprocess_features(X)
A = A + np.eye(A.shape[0])  # Add self-loops

# Model definition (as per Section 3.3 of the paper)
X_in = K.layers.Input(shape=(F,))
A_in = K.layers.Input(shape=(N,))

dropout1 = K.layers.Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg))([dropout1, A_in])
dropout2 = K.layers.Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

# Build model
model = K.models.Model(inputs=[X_in, A_in], outputs=graph_attention_2)
optimizer = K.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Callbacks
es_callback = K.callbacks.EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = K.callbacks.TensorBoard(batch_size=N)

# Train model
validation_data = ([X, A], Y_val, idx_val)
model.fit([X, A],
          Y_train,
          sample_weight=idx_train,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[es_callback, tb_callback])

# Load best model
model.load_weights('logs/best_model.h5')

# Evaluate model
eval_results = model.evaluate([X, A],
                              Y_test,
                              sample_weight=idx_test,
                              batch_size=N,
                              verbose=0)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

