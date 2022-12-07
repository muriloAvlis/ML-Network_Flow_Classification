# Algoritmo Árvore de Decisão

import pandas as pd
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def build_model():
    # Definição de hiperparâmetros para teste
    parameters = {'criterion': ['gini', 'entropy', 'log_loss']}

    # parâmentros a serem avaliados do modelo
    scoring = ['accuracy', 'precision_weighted',
               'recall_weighted', 'f1_weighted']

    # criando o modelo
    dt = DecisionTreeClassifier()

    # aplicando validação cruzada variando os hiperparâmetros
    clf = GridSearchCV(estimator=dt, param_grid=parameters,
                       cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)

    # treinando modelo
    clf.fit(X.values, y.values.ravel())

    # salva o modelo
    dump(clf, './models/decision_tree.joblib')

    return clf


if __name__ == '__main__':
    # Carregando dados pré-processados
    X = pd.read_csv('./data/preprocessed_input.csv')
    y = pd.read_csv('./data/preprocessed_output.csv')

    try:
        clf = load('./models/decision_tree.joblib')
    except IOError:
        print('Modelo não encontrado!')
        print('Construindo modelo...')
        clf = build_model()

    print(
        f'Acurácias médias do modelo: {clf.cv_results_["mean_test_accuracy"]}')
    print(f'Melhor acurácia média do modelo: {clf.best_score_}')
    print(
        f'Precisões médias do modelo: {clf.cv_results_["mean_test_precision_weighted"]}')
    print(
        f'Recalls médios do modelo: {clf.cv_results_["mean_test_recall_weighted"]}')
    print(
        f'F1 Scores médios do modelo: {clf.cv_results_["mean_test_f1_weighted"]}')
