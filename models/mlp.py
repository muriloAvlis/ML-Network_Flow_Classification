# Algoritmo de Rede Neural Perceptron Multicamadas

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from joblib import dump, load
from sklearn.model_selection import train_test_split


def build_model():
    # definição da estrutura da rede neural
    model = Sequential(layers=[
        Dense(units=256, activation='sigmoid', input_shape=(X.shape[1],)),
        Dense(units=128, activation='sigmoid'),
        Dense(units=22, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

    # treinando modelo
    model.fit(X_train, y_train, epochs=40, validation_split=0.3)

    # salva o modelo
    model.save('./models/ANNs/mlp')

    return model


if __name__ == '__main__':
    # Carregando dados pré-processados
    X = pd.read_csv('./data/preprocessed_input.csv')
    y = pd.read_csv('./data/preprocessed_output.csv')

    # aplicando técnica de holdout nos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values.ravel(), test_size=0.4, random_state=32)

    try:
        model = tf.keras.models.load_model('./models/ANNs/mlp')
    except IOError:
        print('Modelo não encontrado!')
        print('Construindo modelo...')
        model = build_model()

    # testando o modelo
    results = model.evaluate(X_test, y_test)
    print(f'Perda e acurácia do modelo no conjunto de teste: {results}')
