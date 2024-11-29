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


# student model

def build_student_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def train_student_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.compile(optimizer=RMSprop(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    history = model.fit(X_train, y_train, 
                        epochs=100, 
                        batch_size=32, 
                        validation_data=(X_val, y_val), 
                        callbacks=[early_stopping])
    return model, history