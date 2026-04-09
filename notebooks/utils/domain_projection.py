
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

# Masked MAE metric
@register_keras_serializable()
def masked_mae(y_true, y_pred):
    mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Masked MSE metric
@register_keras_serializable()
def masked_mse(y_true, y_pred):
    # Mask samples where ALL regression targets are nan
    sample_mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)

    # Select only unmasked samples
    y_true_masked = tf.boolean_mask(y_true, sample_mask)
    y_pred_masked = tf.boolean_mask(y_pred, sample_mask)

    return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))

# R^2 metric with masking
@register_keras_serializable()
def masked_r2(y_true, y_pred):
    mask = ~tf.reduce_all(tf.math.is_nan(y_true), axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + 1e-7)



# Residual Linear Layer
@register_keras_serializable()
class ResidualLinear(layers.Layer):
    def __init__(self, proj_dim, domain_name, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.linear = layers.Dense(proj_dim, activation=None,
                                   name=f"res_linear_{domain_name}")
        

    def call(self, inputs):
        return inputs + self.linear(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Bottleneck Adapter Layer
@register_keras_serializable()
class AdapterLayer(layers.Layer):
    def __init__(self, proj_dim, domain_name, bottleneck_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim

        self.down = layers.Dense(bottleneck_dim, activation='relu',
                                 name=f"adapter_down_{domain_name}")
        self.up = layers.Dense(proj_dim, activation=None,
                               name=f"adapter_up_{domain_name}")

    def call(self, inputs):
        h = self.down(inputs)
        return self.up(h)
    
    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable()
class AdapterLayerV2(layers.Layer):
    def __init__(self, proj_dim, domain_name, bottleneck_dim=32, dropout=0.1,
                 activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        self.down = layers.Dense(
            bottleneck_dim,
            activation=activation,
            name=f"adapter_down_{domain_name}"
        )
        self.up = layers.Dense(
            proj_dim,
            activation=None,
            name=f"adapter_up_{domain_name}",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer="zeros",
        )

    def call(self, inputs, training=None):
        h = self.down(inputs)
        h = self.up(h)
        return h

    def compute_output_shape(self, input_shape):
        return input_shape  

    
# Residual Adapter Layer
@register_keras_serializable()
class ResidualAdapter(layers.Layer):
    def __init__(self, proj_dim, domain_name, bottleneck_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.down = layers.Dense(bottleneck_dim, activation='relu',
                                 name=f"res_adapter_down_{domain_name}")
        self.up = layers.Dense(proj_dim, activation=None,
                               name=f"res_adapter_up_{domain_name}")

    def call(self, inputs):
        h = self.down(inputs)
        return inputs + self.up(h)
    
    def compute_output_shape(self, input_shape):
        return input_shape

@register_keras_serializable()
class ResidualAdapterV2(layers.Layer):
    def __init__(self, proj_dim, domain_name, bottleneck_dim=32, dropout=0.1,
                 activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        self.down = layers.Dense(
            bottleneck_dim,
            activation=activation,
            name=f"adapter_down_{domain_name}"
        )

        self.up = layers.Dense(
            proj_dim,
            activation=None,
            name=f"adapter_up_{domain_name}",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer="zeros",
        )

    def call(self, inputs, training=None):
        h = self.down(inputs)
        h = self.up(h)
        return inputs + h
    
    def compute_output_shape(self, input_shape):
        return input_shape




# FiLM Layer
@register_keras_serializable()
class FiLMLayer(layers.Layer):
    def __init__(self, proj_dim, domain_name, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.proj_dim,),
            initializer="ones",
            trainable=True,
            name=f"film_gamma_{self.domain_name}"
        )

        self.beta = self.add_weight(
            shape=(self.proj_dim,),
            initializer="zeros",
            trainable=True,
            name=f"film_beta_{self.domain_name}"
        )

    def call(self, inputs):
        return self.gamma * inputs + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Mask Layer
@register_keras_serializable()
class MaskLayer(layers.Layer):
    """
    A Keras Layer that applies a trainable, element-wise mask to its input.
    Used for domain-specific projection where each domain learns its own mask vector.
    
    Args:
        proj_dim (int): The dimensionality of the projection.
        domain_name (str): The name of the domain for identification.
    """
    def __init__(self, proj_dim, domain_name, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name


    def build(self, input_shape):
        # create trainable mask
        self.mask = self.add_weight(
            shape=(self.proj_dim,),
            initializer="ones",
            trainable=True,
            name=f"mask_{self.domain_name}"
        )

    def call(self, inputs):
        return inputs * self.mask
    
    def compute_output_shape(self, input_shape):
        return input_shape  

@register_keras_serializable()
class NonLinearProjection(layers.Layer):
    def __init__(self, proj_dim=128, domain_name=None, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name

        self.fc1 = layers.Dense(
            proj_dim,
            activation=activation,
            name=f"two_nl_fc1_{domain_name}"
        )
        self.fc2 = layers.Dense(
            proj_dim,
            activation=activation,
            name=f"two_nl_fc2_{domain_name}"
        )

    def call(self, inputs):
        h = self.fc1(inputs)
        return self.fc2(h)
    
    def compute_output_shape(self, input_shape):
        return input_shape

@register_keras_serializable()
class LinearThenAdapter(layers.Layer):
    def __init__(self, proj_dim=128, domain_name=None, bottleneck_dim=32,
                 adapter_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim

        self.linear = layers.Dense(
            proj_dim, activation=None,
            name=f"lin_{domain_name}"
        )

        # your existing non-residual adapter style
        self.down = layers.Dense(
            bottleneck_dim, activation=adapter_activation,
            name=f"lin_ad_down_{domain_name}"
        )
        self.up = layers.Dense(
            proj_dim, activation=None,
            name=f"lin_ad_up_{domain_name}"
        )

    def call(self, inputs):
        h = self.linear(inputs)
        h = self.down(h)
        return self.up(h)

    def compute_output_shape(self, input_shape):
        return input_shape
    
@register_keras_serializable()
class NonLinearThenAdapter(layers.Layer):
    def __init__(self, proj_dim=128, domain_name=None, bottleneck_dim=32,
                 proj_activation="tanh", adapter_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim

        # Stage 1: nonlinear projection
        self.proj = layers.Dense(
            proj_dim, activation=proj_activation,
            name=f"nl_{domain_name}"
        )

        # Stage 2: your non-residual adapter style
        self.down = layers.Dense(
            bottleneck_dim, activation=adapter_activation,
            name=f"nl_ad_down_{domain_name}"
        )
        self.up = layers.Dense(
            proj_dim, activation=None,
            name=f"nl_ad_up_{domain_name}"
        )

    def call(self, inputs):
        h = self.proj(inputs)
        h = self.down(h)
        return self.up(h)
    def compute_output_shape(self, input_shape):
        return input_shape
    
@register_keras_serializable()
class LinearThenResidualAdapter(layers.Layer):
    def __init__(self, proj_dim=128, domain_name=None, bottleneck_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim

        # Stage 1: linear reshape
        self.linear = layers.Dense(
            proj_dim, activation=None,
            name=f"lin_{domain_name}"
        )

        # Stage 2: safe residual adapter delta
        self.adapter = ResidualAdapter(
            proj_dim=proj_dim,
            domain_name=f"{domain_name}_stage2",
            bottleneck_dim=bottleneck_dim,
            name=f"lin_resad_{domain_name}"
        )

    def call(self, inputs, training=None):
        h = self.linear(inputs)
        return self.adapter(h, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable()
class NonLinearThenResidualAdapter(layers.Layer):
    def __init__(self, proj_dim=128, domain_name=None, bottleneck_dim=32,
                 proj_activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.domain_name = domain_name
        self.bottleneck_dim = bottleneck_dim

        # Stage 1: nonlinear projection
        self.proj = layers.Dense(
            proj_dim, activation=proj_activation,
            name=f"nl_{domain_name}"
        )

        # Stage 2: stable residual adapter delta
        self.adapter = ResidualAdapter(
            proj_dim=proj_dim,
            domain_name=f"{domain_name}_stage2",
            bottleneck_dim=bottleneck_dim,
            name=f"lin_resad_{domain_name}"
        )
    def call(self, inputs, training=None):
        h = self.proj(inputs)
        return self.adapter(h, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape
    

# Domain Projection Layer
def get_domain_projection_layer(domain, proj_dim, projection_type='linear',bottleneck_dim=64):
    """
    Returns a dict: domain -> projection layer/module
    projection_type: 'linear', 'nonlinear', 'mask'
    """

    if projection_type == 'linear':
        proj_layer = layers.Dense(
            proj_dim, activation=None, name=f"proj_linear_{domain}"
        )

    elif projection_type == 'nonlinear':
        proj_layer = layers.Dense(proj_dim, activation='relu', name=f"proj_nonlinear_{domain}")

    elif projection_type == 'nonlinear_v2':
        proj_layer = layers.Dense(proj_dim, activation='tanh', name=f"proj_nonlinear_v2_{domain}")

    elif projection_type == 'nonlinear_v3':
        proj_layer = NonLinearProjection(proj_dim=proj_dim, domain_name=domain,
                                        name=f"proj_nonlinear_v3_{domain}")

    elif projection_type == 'nonlinear_v4':
        proj_layer = NonLinearProjection(proj_dim=proj_dim, domain_name=domain,activation='tanh',
                                        name=f"proj_nonlinear_v4_{domain}")

    elif projection_type == 'mask':
        # learned multiplicative mask (element-wise)

        proj_layer = MaskLayer(proj_dim, domain_name=domain,name = f"mask_layer_{domain}")

    elif projection_type == 'film':
        proj_layer = FiLMLayer(proj_dim, domain_name=domain,
                         name=f"film_layer_{domain}")

    elif projection_type == 'residual_linear':
        proj_layer = ResidualLinear(proj_dim, domain_name=domain,
                              name=f"res_linear_wrapper_{domain}")

    elif projection_type == 'adapter':
        proj_layer = AdapterLayer(proj_dim, domain_name=domain,bottleneck_dim=bottleneck_dim,
                            name=f"adapter_layer_{domain}")
        
    elif projection_type == 'adapter_v2':
        proj_layer = AdapterLayerV2(proj_dim, domain_name=domain,bottleneck_dim=bottleneck_dim,
                            name=f"adapter_v2_layer_{domain}")

    elif projection_type == 'residual_adapter':
        proj_layer = ResidualAdapter(proj_dim, domain_name=domain,bottleneck_dim=bottleneck_dim,
                               name=f"res_adapter_layer_{domain}")
        
    elif projection_type == 'residual_adapter_v2':
        proj_layer = ResidualAdapterV2(proj_dim, domain_name=domain,bottleneck_dim=bottleneck_dim,
                               name=f"res_adapter_v2_layer_{domain}")
        
    elif projection_type == "linear_then_adapter":
        proj_layer = LinearThenAdapter(
            proj_dim, domain_name=domain, bottleneck_dim=bottleneck_dim,
            name=f"linear_then_adapter_{domain}"
    )

    elif projection_type == "linear_then_residual_adapter":
        proj_layer = LinearThenResidualAdapter(
            proj_dim, domain_name=domain, bottleneck_dim=bottleneck_dim,
            name=f"linear_then_res_adapter_{domain}"
        )
    elif projection_type == "nonlinear_then_adapter":
        proj_layer = NonLinearThenAdapter(
            proj_dim,
            domain_name=domain,
            bottleneck_dim=bottleneck_dim,
            name=f"nonlinear_then_adapter_{domain}"
    )

    elif projection_type == "nonlinear_then_residual_adapter":
        proj_layer = NonLinearThenResidualAdapter(
            proj_dim,
            domain_name=domain,
            bottleneck_dim=bottleneck_dim,
            name=f"nonlinear_then_res_adapter_{domain}"
        )


    return proj_layer

# Janossy Pooling Layer
@register_keras_serializable()
class JanossyPooling(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Shared Encoder
def get_shared_encoder(rnn_type='gru', rnn_units=80, num_permutations=6, input_dim=8):

    def encoder(x):
        # x shape: (batch, num_permutations, seq_len, input_dim)

        # Mask padded zeros
        x = layers.TimeDistributed(
                layers.Masking(mask_value=0.0),
                name="masking_layer"
            )(x)

        # h(x): Dense embedding per timestep
        x = layers.TimeDistributed(
                layers.TimeDistributed(
                    layers.Dense(64, activation='relu', name="dense_embedding")
                ),
                name="time_distributed_embedding"
            )(x)

        # RNN across each permutation
        if rnn_type.lower() == 'gru':
            x = layers.TimeDistributed(
                layers.GRU(rnn_units, name="gru_rnn"),
                name="time_distributed_rnn"
            )(x)
        else:
            x = layers.TimeDistributed(
                layers.LSTM(rnn_units, name="lstm_rnn"),
                name="time_distributed_rnn"
            )(x)

        # Janossy pooling (usually mean over k permutations)
        x = JanossyPooling(name="janossy_pooling")(x)

        # Final shared representation
        x = layers.Dense(128, activation='tanh', name="shared_dense")(x)
        x = layers.Dropout(0.2, name="shared_dropout")(x)

        return x
    
    return encoder

# Task-specific Shared Regression heads
def get_regression_head(output_dim, activation='linear'):

    def head(shared_repr):
        r = layers.Dense(128, activation='relu', name='regression_dense')(shared_repr)
        r = layers.Dropout(0.2, name='regression_dropout')(r)
        out = layers.Dense(output_dim, activation=activation, name="regression_output")(r)
        return out
    
    return head

# Task-specific Shared Classification heads
def get_classification_head(activation='sigmoid'):

    def head(shared_repr):
        c = layers.Dense(128, activation='relu', name='classification_dense')(shared_repr)
        c = layers.Dropout(0.2, name='classification_dropout')(c)
        out = layers.Dense(1, activation=activation, name="classification_output")(c)
        return out

    return head



# Full model builder
def build_janossy_rnn(
        reg_output_dim,
        input_dim,
        domain,
        domain_proj_dim=128,
        projection_type='linear',
        rnn_type='gru',
        rnn_units=80,
        num_permutations=6,
        reg_output_activation='linear',
        reg_loss_function=masked_mse,
        cls_output_activation='sigmoid',
        cls_loss_function='binary_crossentropy'):

    # ---------------------------------
    # Inputs
    # ---------------------------------
    input_layer = keras.Input((num_permutations, None, input_dim), name="input_layer")

    # ---------------------------------
    # Shared Encoder
    # ---------------------------------
    encoder = get_shared_encoder(
        rnn_type=rnn_type,
        rnn_units=rnn_units,
        num_permutations=num_permutations,
        input_dim=input_dim
    )

    shared_repr = encoder(input_layer)

    # ----------------------
    # Domain-specific projection
    # ----------------------
    domain_proj_layer = get_domain_projection_layer(domain, proj_dim=domain_proj_dim, 
                                              projection_type=projection_type)
    projected_repr = domain_proj_layer(shared_repr)


    # ---------------------------------
    # Task-specific heads
    # ---------------------------------
    reg_head = get_regression_head(reg_output_dim, reg_output_activation)
    cls_head = get_classification_head(cls_output_activation)

    reg_output = reg_head(projected_repr)
    cls_output = cls_head(projected_repr)

    # ---------------------------------
    # Final model
    # ---------------------------------
    model = keras.Model(
        inputs=input_layer,
        outputs=[reg_output, cls_output],
        name="janossy_multitask_domain_model"
    )

    model.compile(
        optimizer='adam',
        loss={
            'regression_output': reg_loss_function,
            'classification_output': cls_loss_function
        },
        loss_weights={
            'regression_output': 5.0,
            'classification_output': 1.0
        },
        metrics={
            'regression_output': [masked_mae, masked_r2],
            'classification_output': ['accuracy']
        }
    )

    return model

# Utility to clone model with weights
def clone_model_with_weights(model):
    """
    Creates a true deep copy of a Keras model (architecture + weights).
    Freezing this copy will NOT affect the original.
    """
    cloned_model = keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    return cloned_model

# Add projection layer after encoder and train only that
def add_projection_train_only_projection(
    model,
    domain="target",
    proj_dim=128,
    projection_type="linear",
    insert_after="dropout_final",
    bottleneck_dim=64
):
    """
    Replaces the projection layer and trains ONLY the new projection.
    Everything else (backbone + heads) is frozen.
    """

    # 1. Clone model safely
    model = clone_model_with_weights(model)

    # 2. Freeze absolutely everything
    for layer in model.layers:
        layer.trainable = False

    # 3. Locate insertion point
    shared_tensor = model.get_layer(insert_after).output

    # 4. Create new projection
    new_proj_layer = get_domain_projection_layer(
        domain=domain,
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )

    new_proj_output = new_proj_layer(shared_tensor)

    # 5. Reconnect FROZEN original heads
    reg_adapter = model.get_layer("regression_dense")
    clf_adapter = model.get_layer("classification_dense")

    reg_dropout = model.get_layer("regression_dropout")
    clf_dropout = model.get_layer("classification_dropout")

    reg_out = model.get_layer("regression_output")
    clf_out = model.get_layer("classification_output")

    reg_x = reg_adapter(new_proj_output)
    reg_x = reg_dropout(reg_x)
    reg_output = reg_out(reg_x)

    clf_x = clf_adapter(new_proj_output)
    clf_x = clf_dropout(clf_x)
    clf_output = clf_out(clf_x)

    # 6. Build new model
    new_model = tf.keras.Model(
        inputs=model.input,
        outputs=[reg_output, clf_output],
        name=f"{model.name}_projection_only_{domain}"
    )

    # 7. VERY IMPORTANT: Only new projection is trainable
    for layer in new_model.layers:
        layer.trainable = False

    new_proj_layer.trainable = True  # Only this one trains

    # Also train all BN/LN layers
    for layer in new_model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization,
                              tf.keras.layers.LayerNormalization)):
            layer.trainable = True

    return new_model


# Add input projection layer after masking and before encoding and train only that
def add_input_projection_train_only_projection(
    model,
    domain="target",
    proj_dim=8,  # must match input feature dim
    projection_type="linear",
    insert_after="masking_layer",
    bottleneck_dim=64
):
    """
    Adds a domain-specific projection at the BEGINNING of the model.
    Only the projection layer is trainable.
    """

    # 1 Clone
    base_model = clone_model_with_weights(model)

    # 2 Freeze all
    for layer in base_model.layers:
        layer.trainable = False

    # 3 Get insertion tensor
    x = base_model.get_layer(insert_after).output
    # shape: (None, 6, None, 8)

    # 4 Projection (TimeDistributed!)
    proj_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.TimeDistributed(
        get_domain_projection_layer(
            domain=domain,
            proj_dim=proj_dim,
            projection_type=projection_type,
            bottleneck_dim=bottleneck_dim,)
            ),
        name=f"time_distributed_proj_{domain}",
    )

    x = proj_layer(x)

    # 5 Continue original graph
    embedding = base_model.get_layer("time_distributed_embedding")
    rnn = base_model.get_layer("time_distributed_rnn")
    pooling = base_model.get_layer("janossy_pooling")
    dense = base_model.get_layer("dense_final")
    dropout = base_model.get_layer("dropout_final")

    x = embedding(x)
    x = rnn(x)
    x = pooling(x)
    x = dense(x)
    x = dropout(x)

    # Heads
    reg_adapter = base_model.get_layer("regression_dense")
    clf_adapter = base_model.get_layer("classification_dense")

    reg_dropout = base_model.get_layer("regression_dropout")
    clf_dropout = base_model.get_layer("classification_dropout")

    reg_out = base_model.get_layer("regression_output")
    clf_out = base_model.get_layer("classification_output")

    reg_x = reg_adapter(x)
    reg_x = reg_dropout(reg_x)
    reg_output = reg_out(reg_x)

    clf_x = clf_adapter(x)
    clf_x = clf_dropout(clf_x)
    clf_output = clf_out(clf_x)

    # 6 Build model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_output, clf_output],
        name=f"{base_model.name}_input_proj_{domain}",
    )

    # 7 Train ONLY projection
    for layer in new_model.layers:
        layer.trainable = False

    proj_layer.trainable = True

    return new_model



# Add input projection layer after embedding and before RNN and train only that
def add_projection_after_embedding_train_only_projection(
    model,
    domain="target",
    proj_dim=64,  # must match input feature dim
    projection_type="linear",
    insert_after="time_distributed_embedding",
    bottleneck_dim=64
):
    """
    Adds a domain-specific projection at the BEGINNING of the model.
    Only the projection layer is trainable.
    """

    # 1 Clone
    base_model = clone_model_with_weights(model)

    # 2 Freeze all
    for layer in base_model.layers:
        layer.trainable = False

    # 3 Get insertion tensor
    x = base_model.get_layer(insert_after).output
    # shape: (None, 6, None, 8)

    # 4 Projection (TimeDistributed!)
    proj_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.TimeDistributed(
        get_domain_projection_layer(
            domain=domain,
            proj_dim=proj_dim,
            projection_type=projection_type,
            bottleneck_dim=bottleneck_dim,)
        ),
        name=f"time_distributed_proj_{domain}",
    )

    x = proj_layer(x)

    # 5 Continue original graph
    rnn = base_model.get_layer("time_distributed_rnn")
    pooling = base_model.get_layer("janossy_pooling")
    dense = base_model.get_layer("dense_final")
    dropout = base_model.get_layer("dropout_final")

    x = rnn(x)
    x = pooling(x)
    x = dense(x)
    x = dropout(x)

    # Heads
    reg_adapter = base_model.get_layer("regression_dense")
    clf_adapter = base_model.get_layer("classification_dense")

    reg_dropout = base_model.get_layer("regression_dropout")
    clf_dropout = base_model.get_layer("classification_dropout")

    reg_out = base_model.get_layer("regression_output")
    clf_out = base_model.get_layer("classification_output")

    reg_x = reg_adapter(x)
    reg_x = reg_dropout(reg_x)
    reg_output = reg_out(reg_x)

    clf_x = clf_adapter(x)
    clf_x = clf_dropout(clf_x)
    clf_output = clf_out(clf_x)

    # 6 Build model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_output, clf_output],
        name=f"{base_model.name}_input_proj_{domain}",
    )

    # 7 Train ONLY projection
    for layer in new_model.layers:
        layer.trainable = False

    proj_layer.trainable = True

    return new_model


# Add projection layer after RNN and train only that
def add_projection_after_rnn_train_only_projection(
    model,
    domain="target",
    proj_dim=80,
    projection_type="linear",
    insert_after="time_distributed_rnn",
    bottleneck_dim=64
):
    """
    Inserts a domain-specific projection AFTER the TimeDistributed RNN.
    Backbone + heads are frozen. ONLY the projection is trainable.
    """

    # 1 Clone model safely
    base_model = clone_model_with_weights(model)

    # 2 Freeze everything
    for layer in base_model.layers:
        layer.trainable = False

    # 3 Get RNN output
    rnn_layer = base_model.get_layer(insert_after)
    x = rnn_layer.output   # (B, 6, 80)

    # 4 Projection (TimeDistributed)
    proj_layer = tf.keras.layers.TimeDistributed(
        get_domain_projection_layer(
        domain=domain,
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    ),
    name=f"time_distributed_proj_{domain}"
    )
    x = proj_layer(x)  # (B, 6, 80)

    # 5 Continue original frozen backbone
    x = base_model.get_layer("janossy_pooling")(x)
    x = base_model.get_layer("dense_final")(x)
    x = base_model.get_layer("dropout_final")(x)

    # Heads (unchanged)
    reg_x = base_model.get_layer("regression_dense")(x)
    reg_x = base_model.get_layer("regression_dropout")(reg_x)
    reg_out = base_model.get_layer("regression_output")(reg_x)

    clf_x = base_model.get_layer("classification_dense")(x)
    clf_x = base_model.get_layer("classification_dropout")(clf_x)
    clf_out = base_model.get_layer("classification_output")(clf_x)

    # 6 Build new model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_out, clf_out],
        name=f"{base_model.name}_proj_after_rnn_{domain}",
    )

    # 7 Ensure ONLY projection trains
    for layer in new_model.layers:
        layer.trainable = False
    proj_layer.trainable = True

    return new_model



# Add projection layer after Janossy pooling and train only that
def add_projection_after_janossy(
    model,
    domain="target",
    proj_dim=80,
    projection_type="linear",
    insert_after="janossy_pooling",
    bottleneck_dim=64
):
    """
    Adds a domain-specific projection layer AFTER Janossy pooling.
    Trains ONLY the new projection; everything else is frozen.
    """

    # 1 Clone model (NO side effects)
    base_model = clone_model_with_weights(model)

    # 2 Freeze everything
    for layer in base_model.layers:
        layer.trainable = False

    # 3 Locate Janossy output
    janossy = base_model.get_layer(insert_after)
    x = janossy.output  # (B, 80)

    # 4 Create projection
    proj_layer = get_domain_projection_layer(
        domain=domain,
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )

    proj_layer._name = f"proj_after_janossy_{domain}"
    x = proj_layer(x)

    # 5 Reconnect existing frozen layers
    dense_final = base_model.get_layer("dense_final")
    dropout_final = base_model.get_layer("dropout_final")

    x = dense_final(x)
    x = dropout_final(x)

    # Heads
    reg_adapter = base_model.get_layer("regression_dense")
    clf_adapter = base_model.get_layer("classification_dense")

    reg_dropout = base_model.get_layer("regression_dropout")
    clf_dropout = base_model.get_layer("classification_dropout")

    reg_out = base_model.get_layer("regression_output")
    clf_out = base_model.get_layer("classification_output")

    reg_x = reg_adapter(x)
    reg_x = reg_dropout(reg_x)
    reg_output = reg_out(reg_x)

    clf_x = clf_adapter(x)
    clf_x = clf_dropout(clf_x)
    clf_output = clf_out(clf_x)

    # 6 Build new model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_output, clf_output],
        name=f"{model.name}_proj_after_janossy_{domain}"
    )

    # 7 Ensure ONLY projection is trainable
    for layer in new_model.layers:
        layer.trainable = False

    proj_layer.trainable = True

    return new_model




# Add task-specific projections after encoder
def add_task_specific_projections_after_encoder(
    model,
    domain="target",
    proj_dim=128,
    projection_type="linear",
    bottleneck_dim=64
):
    """
    Adds task-specific domain projections AFTER the shared encoder.
    Only the projections are trainable.
    """

    # 1 Clone model to avoid side effects
    base_model = clone_model_with_weights(model)

    # 2 Freeze EVERYTHING
    for layer in base_model.layers:
        layer.trainable = False

    # 3 Encoder output
    encoder_out = base_model.get_layer("dense_final").output  # (B, 128)
    encoder_out = base_model.get_layer("dropout_final")(encoder_out)

    # 4 Create task-specific projections
    reg_proj = get_domain_projection_layer(
        domain=f"{domain}_regression",
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )
    reg_proj._name = f"proj_regression_{domain}"

    clf_proj = get_domain_projection_layer(
        domain=f"{domain}_classification",
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )
    clf_proj._name = f"proj_classification_{domain}"

    reg_x = reg_proj(encoder_out)
    clf_x = clf_proj(encoder_out)

    # 5 Reuse existing frozen heads
    reg_adapter = base_model.get_layer("regression_dense")
    reg_dropout = base_model.get_layer("regression_dropout")
    reg_out = base_model.get_layer("regression_output")

    clf_adapter = base_model.get_layer("classification_dense")
    clf_dropout = base_model.get_layer("classification_dropout")
    clf_out = base_model.get_layer("classification_output")

    reg_x = reg_adapter(reg_x)
    reg_x = reg_dropout(reg_x)
    reg_output = reg_out(reg_x)

    clf_x = clf_adapter(clf_x)
    clf_x = clf_dropout(clf_x)
    clf_output = clf_out(clf_x)

    # 6 Build new model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_output, clf_output],
        name=f"{model.name}_task_proj_{domain}"
    )

    # 7 Ensure ONLY projections train
    for layer in new_model.layers:
        layer.trainable = False

    reg_proj.trainable = True
    clf_proj.trainable = True

    return new_model



def add_task_specific_projection_before_output(
    model,
    domain="target",
    proj_dim=128,
    projection_type="linear",
    bottleneck_dim=64
):
    """
    Adds task-specific projection layers right BEFORE each output layer.
    Only the projections are trainable.
    """

    # 1 Clone model to avoid side effects
    base_model = clone_model_with_weights(model)

    # 2 Freeze everything
    for layer in base_model.layers:
        layer.trainable = False

    # =========================
    # Regression branch
    # =========================
    reg_dropout = base_model.get_layer("regression_dropout")
    reg_x = reg_dropout.output  # (B, 128)

    reg_proj = get_domain_projection_layer(
        domain=f"{domain}_regression",
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )
    reg_proj._name = f"proj_regression_before_out_{domain}"

    reg_x = reg_proj(reg_x)

    reg_out_layer = base_model.get_layer("regression_output")
    reg_output = reg_out_layer(reg_x)

    # =========================
    # Classification branch
    # =========================
    clf_dropout = base_model.get_layer("classification_dropout")
    clf_x = clf_dropout.output  # (B, 128)

    clf_proj = get_domain_projection_layer(
        domain=f"{domain}_classification",
        proj_dim=proj_dim,
        projection_type=projection_type,
        bottleneck_dim=bottleneck_dim
    )
    clf_proj._name = f"proj_classification_before_out_{domain}"

    clf_x = clf_proj(clf_x)

    clf_out_layer = base_model.get_layer("classification_output")
    clf_output = clf_out_layer(clf_x)

    # 3 Build new model
    new_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[reg_output, clf_output],
        name=f"{model.name}_proj_before_out_{domain}"
    )

    # 4 Ensure ONLY projections train
    for layer in new_model.layers:
        layer.trainable = False

    reg_proj.trainable = True
    clf_proj.trainable = True

    return new_model
