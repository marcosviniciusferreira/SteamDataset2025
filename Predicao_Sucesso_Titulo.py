# %% [markdown]
# # Projeto de Análise Preditiva: Mercado Steam 2025
# **Objetivo:** Apoiar decisões de investimento de publicadoras através da predição de sucesso de jogos.
# **Autor:** Marcos Vinicius Ferreira Paim
# [passo 1: Carregamento de Dados e Bibliotecas]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Carregamento das bases brutas
df_applications = pd.read_csv("Base/applications.csv")
df_reviews = pd.read_csv("Base/reviews.csv")
app_publishers = pd.read_csv("Base/application_publishers.csv")
app_genres = pd.read_csv('Base/application_genres.csv')
genres = pd.read_csv('Base/genres.csv')
reviews = pd.read_csv('Base/reviews.csv', usecols=['appid', 'voted_up'])

# %%

df_reviews.columns.to_list()

# %% [passo 2: Integração e Filtragem de Escopo]
# Seleção inicial de colunas técnicas

cols_app = [
    'appid', 'type', 'is_free', 'required_age', 
    'mat_initial_price', 'mat_discount_percent', 'mat_achievement_count', 
    'mat_supports_windows', 'mat_supports_mac', 'mat_supports_linux', 
    'supported_languages', 'metacritic_score', 'recommendations_total'
]

df_final = df_applications[cols_app].copy()
# Filtro de Negócio: Focar apenas em jogos e títulos com relevância estatística (>50 reviews)
df_final = df_final[df_final['type'] == 'game'].copy()

# %%
# %%
plt.figure(figsize=(10, 6))
plt.hist(df_final['recommendations_total'], bins=50, color='teal', edgecolor='black', log=True)

plt.title('Distribuição da Quantidade de Avaliações (Escala Log)')
plt.xlabel('Total de Recomendações')
plt.ylabel('Frequência de Jogos (Log)')
plt.grid(axis='y', alpha=0.3)
plt.show()


#cruzando tabela de jogos com a tabela de generos para obter apenas os generos que tem na tabela de jogos
df_genres = pd.merge(app_genres, genres, left_on='genre_id', right_on='id')
df_genres = df_genres.drop_duplicates(subset=['appid'])
df_final = pd.merge(df_final, df_genres[['appid', 'name']], on='appid', how='left')
df_final.rename(columns={'name': 'genre'}, inplace=True)

# %% [passo 3: Saneamento e Limpeza de Dados (Data Cleaning)]
# Tratamento de nulos básicos

df_final['mat_achievement_count'] = df_final['mat_achievement_count'].fillna(0)
df_final = df_final.dropna(subset=['mat_initial_price', 'genre'])

# Coerção de tipos e limpeza de ruído (Ex: strings de JS em campos numéricos)
df_final['required_age'] = pd.to_numeric(df_final['required_age'], errors='coerce')
df_final['required_age'] = df_final['required_age'].fillna(0)
df_final['metacritic_score'] = pd.to_numeric(df_final['metacritic_score'], errors='coerce')
df_final['metacritic_score'] = df_final['metacritic_score'].fillna(df_final['metacritic_score'].mean())
df_final['is_free'] = df_final['is_free'].astype(int)

# Engenharia de Atributos: Contagem de idiomas suportados
df_final['num_languages'] = df_final['supported_languages'].apply(lambda x: len(str(x).split(',')))
# %% [passo 4: Engenharia de Atributos - Target e Reputação]
# Cálculo do Target (Sucesso definido como > 80% de aprovação)

taxa_sucesso = reviews.groupby('appid')['voted_up'].mean().reset_index()
taxa_sucesso['target'] = (taxa_sucesso['voted_up'] > 0.8).astype(int)

# Cálculo da Reputação da Publicadora (Média de sucesso histórico da empresa)
reputacao_pub = pd.merge(app_publishers, taxa_sucesso[['appid', 'target']], on='appid')
status_empresa = reputacao_pub.groupby('publisher_id')['target'].mean().reset_index()
status_empresa.rename(columns={'target': 'publisher_reputation_score'}, inplace=True)

# Integração das métricas calculadas ao DataFrame principal
df_final = pd.merge(df_final, taxa_sucesso[['appid', 'target']], on='appid', how='inner')
df_final = pd.merge(df_final, app_publishers, on='appid', how='left')
df_final = pd.merge(df_final, status_empresa, on='publisher_id', how='left')

# Preenchimento de empresas novas com a média global para evitar viés
df_final['publisher_reputation_score'] = df_final['publisher_reputation_score'].fillna(df_final['publisher_reputation_score'].mean())


# %% [passo 5: Preparação para Machine Learning]
# Transformação de variáveis categóricas (One-Hot Encoding)
df_final_model = pd.get_dummies(df_final, columns=['type', 'genre'])

# Remover colunas de texto que não servem para o cálculo
df_final_model = df_final_model.drop(columns=['supported_languages', 'appid'])

# Remoção de identificadores e colunas de texto para o treino
X = df_final_model.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
y = df_final_model['target']

# Dividir 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [passo 6: Treinamento e Avaliação do Modelo]
# Aplicação do algoritmo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Métricas de performance
previsoes = modelo.predict(X_test)
print(classification_report(y_test, previsoes))
# %%

# %% [passo 7: Visualização de Resultados e Insights de Negócio]
# Gráfico de Importância de Atributos (Gini Importance)
importancias = modelo.feature_importances_
colunas = X.columns

# Matriz de Confusão para análise de erros
cm = confusion_matrix(y_test, previsoes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsão da IA')
plt.ylabel('Realidade (Sucesso ou não)')
plt.title('Matriz de Confusão: Onde o modelo está errando?')
plt.show()

