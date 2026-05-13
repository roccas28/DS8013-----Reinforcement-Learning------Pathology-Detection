import tensorflow as tf
from tensorflow.keras import layers, models

def build_dqn_agent(input_shape=(84, 84, 1), action_size=7):    # Builds the CNN Q-Network
    print("Building DQN Architecture...")
    
    inputs = layers.Input(shape=input_shape)
    
    # Visual Feature Extraction (Lightweight CNN Backbone) 
    x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    
    # Action Selection (Value Function Approximation)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(action_size, activation=None)(x) # Raw Q-values -> No Activation
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
