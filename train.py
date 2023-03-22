from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
import tensorflow as tf

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 128
SAVE_MODEL_PATH = 'model2.h5'

def build_model(output_units: int, num_units: list[int], loss: str, learning_rate: float) -> tf.keras.Model:

    # create the model architecture
    input = tf.keras.layers.Input(shape=(None, output_units))
    x = input
    for i in range(len(num_units)-1):
        x = tf.keras.layers.LSTM(num_units[i], return_sequences=True)(x)
    x = tf.keras.layers.LSTM(num_units[-1])(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    output = tf.keras.layers.Dense(output_units, activation='softmax')(x)
    
    model = tf.keras.Model(input, output)
    
    # compile model
    model.compile(
        loss=loss, 
         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
         metrics=['accuracy'])
    
    model.summary()
    
    return model
    

def train(output_units: int = OUTPUT_UNITS, num_units: list[int] = NUM_UNITS, loss: str = LOSS, learning_rate: float = LEARNING_RATE):
    
    #with tf.device('cpu:0'):
    
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, MAPPING_PATH)
    
    # build the network
    model: tf.keras.Model = build_model(output_units, num_units, loss, learning_rate)
    
    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # save model
    model.save(SAVE_MODEL_PATH)
    
    
if __name__ == '__main__':
    train()