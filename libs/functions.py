# Funções utilizadas no projeto

import matplotlib.pyplot as plt
import numpy as np


def plotCorrelationMatrix(df, graphWidth) -> None:
    '''
    Créditos: Kaggle Kerneler / Kaggle Team | Licença: Apache 2.0 open source license

    Plota a matriz de correlação a partir das características (colunas) de um Pandas DataFrame passado na entrada.

    :param df: Pandas.Dataframe
    :param graphWidth: int 
    :return: None
    '''
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(
            f'Nenhum gráfico de correlação mostrado: o número de colunas (não-NaN) ou constantes é menor que 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth),
               dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(
        f'Matriz de Correlação das Características\n(Pós-processamento)', fontsize=19)
    plt.show()


def plot_features(X, features: list) -> None:
    '''
    Plota a variação do valor de características dadas
    para as 200 primeiras instâncias
    '''
    features = features
    fig, axs = plt.subplots(ncols=len(features), figsize=(21, 9))
    fig.suptitle('Dados de Entrada', fontsize=18)
    fig.tight_layout()
    for i in range(len(features)):
        if features[i] in X.columns.to_list():
            axs[i].set(xlabel='Instância (n)', ylabel='Valor')
            axs[i].set_title(features[i].title())
            axs[i].plot(X[features[i]].values[0:200])
    plt.show()


def plot_label_hist(y, le) -> None:
    '''
    Plota o histograma das classes da saída
    '''

    plt.figure(figsize=(16, 9))
    plt.title('Frequência de cada Classe no Dataset')

    plt.xlabel('Classe')
    plt.ylabel('Frequência Absoluta')

    categorical_count = np.unique(
        le['application_protocol'].inverse_transform(y), return_counts=True)

    bars = categorical_count[0]
    x_pos = np.arange(len(bars))

    plt.bar(x_pos, categorical_count[1], align='center', width=1)
    plt.xticks(x_pos, bars, rotation=30)
    plt.show()
