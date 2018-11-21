import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Dense, Dropout


class GraphAttentionLayer(Dense):

    def __init__(self,
                 feature_units,
                 attn_heads=1,
                 attn_heads_reduction="concat",  # {"concat", "average"}
                 dropout_rate=0.5,
                 activation="relu",
                 attn_kernel_initializer="glorot_uniform",
                 attn_kernel_regularizer=None,
                 attn_kernel_constraint=None,
                 attention=True,
                 return_attention=False,
                 **kwargs):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        super(GraphAttentionLayer, self).__init__(units=feature_units,
                                                  activation=activation,
                                                  **kwargs)

        # Number of attention heads (K in the paper)
        self.attn_heads = attn_heads
        # Eq. 5 and 6 in the paper
        self.attn_heads_reduction = attn_heads_reduction
        # Internal dropout rate
        self.dropout_rate = dropout_rate

        self.attn_kernel_initializer \
            = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer \
            = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attention = attention
        self.return_attention = return_attention
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]
        self.supports_masking = False

        # Populated by build()
        self.kernels = []
        self.biases = []
        self.neighbor_kernels = []
        self.attn_kernels = []
        self.attention_biases = []

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F")
            self.output_dim = self.units * self.attn_heads
        else:
            # Output will have shape (..., F")
            self.output_dim = self.units

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 2
        assert len(A_dims) == 2

        _, F = X_dims
        _, N = A_dims

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.units),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="kernel_{}".format(head))
            self.kernels.append(kernel)

            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.units,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name="bias_{}".format(head))
                self.biases.append(bias)

            if not self.attention:
                continue

            # Attention kernels
            neighbor_kernel = self.add_weight(
                                    shape=(F, self.units),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    name="kernel_neighbor_{}".format(head))

            attn_kernel = self.add_weight(
                                    shape=(self.units, 1),
                                    initializer=self.attn_kernel_initializer,
                                    regularizer=self.attn_kernel_regularizer,
                                    constraint=self.attn_kernel_constraint,
                                    name="attn_kernel_{}".format(head))

            self.neighbor_kernels.append(neighbor_kernel)
            self.attn_kernels.append(attn_kernel)

            if self.use_bias:
                biases = []
                for kind in ["self", "neigbor"]:
                    name = "bias_attn_{}_{}".format(kind, head)
                    bias = self.add_weight(shape=(N,),
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint,
                                           name=name)
                    biases.append(bias)
                self.attention_biases.append(biases)

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        X_dims, A_dims = [t.get_shape().as_list() for t in inputs]
        _, F = X_dims
        _, N = A_dims

        outputs = []
        attentions = []
        for head in range(self.attn_heads):
            # W in the paper (F x F")
            kernel = self.kernels[head]

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F")
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F")

            if not self.attention:
                attention = A
                node_features = K.dot(attention, dropout_feat)  # (N x F")
            else:
                # Attention kernel a in the paper (2F" x 1)
                neighbor_kernel = self.neighbor_kernels[head]
                attn_kernel = self.attn_kernels[head]

                neighbor_features = K.dot(X, neighbor_kernel)

                attn_self = K.dot(features, attn_kernel)
                attn_neighbor = K.dot(neighbor_features, attn_kernel)

                if self.use_bias:
                    self_attn_bias, neigbor_attn_bias = self.attention_biases[head]
                    attn_self = K.bias_add(K.reshape(attn_self, (-1, N)),
                                           self_attn_bias)
                    attn_neighbor = K.bias_add(K.reshape(attn_neighbor, (-1, N)),
                                               neigbor_attn_bias)

                attention = attn_neighbor + tf.transpose(attn_self)
                attention = tf.nn.tanh(attention)
                attention = K.reshape(attention, (N, N))
                mask = -10e9 * (1.0 - A)
                attention += mask

                attention = tf.nn.softmax(attention)
                dropout_attn = Dropout(self.dropout_rate)(attention)

                node_features = K.dot(dropout_attn, dropout_feat)

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            if self.return_attention:
                attentions.append(attention)
            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads" output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs, axis=-1)  # (N x KF")
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (N x F")
            # If "average", compute the activation here (Eq. 6)

        output = self.activation(output)

        if self.return_attention:
            attentions = K.stack(attentions, axis=1)
            return (output, attentions)
        else:
            return output

    def compute_output_shape(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 2
        assert len(A_dims) == 2
        output_shape = X_dims[0], self.output_dim

        if self.return_attention:
            return (tf.TensorShape(output_shape),
                    tf.TensorShape(A_dims.insert(1, self.attn_heads)))
        else:
            return tf.TensorShape(output_shape)
