from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(126, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    return model

def train_model(model, X_train, y_train_encoded, X_val, y_val_encoded):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(optimizer=RMSprop(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train_encoded, 
                        epochs=100, 
                        batch_size=32, 
                        validation_data=(X_val, y_val_encoded), 
                        callbacks=[early_stopping])
    return model, history
