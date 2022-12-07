# Código que pré-processa o conjunto de dados (dataset), bem como, faz algumas avaliações e gera alguns gráficos

# importação de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from collections import defaultdict
from libs.functions import plotCorrelationMatrix, plot_features, plot_label_hist


class Preprocessing(object):
    def __init__(self, df) -> None:
        self.df = df

    def drop_uninteresting_data(self) -> pd.DataFrame:
        '''
        Remove dados desconhecidos e desinteressantes do Pandas DataFrame
        e retorna um novo Dataframe com esses dados removidos
        '''
        # remove linhas com valores NaN
        self.df = self.df.dropna(axis=0)

        # remove linhas com valores duplicados
        self.df = self.df.drop_duplicates()

        # remove linhas que contenham valores desconhecidos
        for column in self.df.columns:
            self.df = self.df.drop(
                self.df[self.df[column].isin(['Unknown'])].index, axis=0)

        # remova colunas desinteressantes ou redundantes (flow_key)
        self.df = self.df.drop('flow_key', axis=1)

        return self.df

    def label_encoder(self) -> pd.DataFrame:
        '''
        Transforma colunas em formato string para numéricos
        e retorna o Dataframe transformado
        '''
        le = defaultdict(LabelEncoder)
        for column in self.df.select_dtypes('object').columns.to_list():
            self.df[column] = le[self.df[column].name].fit_transform(
                self.df[column])
        self.le = le

        return self.df

    def split_in_out_data(self) -> list:
        '''
        Separa os dados de entrada e saída retornando uma lista [X,y]
        '''
        return [self.df.drop('application_protocol', axis=1), self.df['application_protocol']]

    def selection_attr(self, X, y) -> pd.DataFrame:
        '''
        Seleciona as melhores características baseado no feature_selection e retorna um novo DataFrame
        '''
        selector = SelectKBest(score_func=chi2, k=15).fit(X, y)
        columns = selector.get_support(indices=True)
        X = X.iloc[:, columns]
        return X

    def standard_attr(self, X) -> pd.DataFrame:
        '''Retorna dados de entrada padronizados'''
        X_scaled = StandardScaler().fit_transform(X.values)
        X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X

    def save_in_out_csv(self, X, y) -> None:
        '''Salva dados de entrada e saída em arquivos csv separados'''
        X.to_csv('./data/preprocessed_input.csv', index=False)
        y.to_csv('./data/preprocessed_output.csv', index=False)


if __name__ == '__main__':
    # carrega os dados
    df = pd.read_csv(
        './data/Unicauca-dataset-April-June-2019-Network-flows.csv')

    # imprime algumas informações sobre o dataframe
    print(f'{20*"*"}DADOS ANTES DO PRÉ-PROCESSAMENTO{20*"*"}')
    print(
        f'Proporção da entrada: ({df.shape[0]},{df.shape[1] - 1})')
    print(f'Quantidade de valores NaN no dataframe: {sum(df.isna().sum())}')
    print(
        f'Quantidade de valores duplicados no dataframe: {df.duplicated().sum()}\n')

    # Chama classe de pré-processamento
    pp = Preprocessing(df)

    # remove dados/colunas desinteressantes
    df = pp.drop_uninteresting_data()

    # transforma dados em string para numéricos
    df = pp.label_encoder()

    # separa os dados de entrada e saída
    X, y = pp.split_in_out_data()

    # aplica o feature_selection
    X = pp.selection_attr(X, y)

    # padroniza os dados de entrada
    X = pp.standard_attr(X)

    # salva dados em um arquivo csv
    pp.save_in_out_csv(X, y)

    # plota o comportamento de algumas features
    plot_features(X, features=['octetTotalCount',
                  'max_ps', 'dst_port', 'max_ps'])

    # plota histograma dos rótulos de saída
    plot_label_hist(y, pp.le)

    # plota matriz de correlação da entrada
    plotCorrelationMatrix(X, 14)

    # imprime algumas informações após o pré-processamento
    print(f'{20*"*"}DADOS APÓS O PRÉ-PROCESSAMENTO{20*"*"}')
    print(f'Proporção da entrada: ({df.shape[0]},{df.shape[1] -1})')
    print(f'Quantidade de valores NaN no dataframe: {sum(df.isna().sum())}')
    print(
        f'Quantidade de valores duplicados no dataframe: {df.duplicated().sum()}')
