# %% [markdown]
# # Análise Preditiva: Títulos Steam 2025
# Uma publicadora de jogos usaria isso para decidir se investe em um novo projeto antes de ele ser lançado.
# Imports e criação dos DF
import pandas as pd

df_applications = pd.read_csv("Base/applications.csv")
df_reviews = pd.read_csv("Base/reviews.csv")
app_publishers = pd.read_csv("Base/application_publishers.csv")

# %%
df_applications['type'].unique

# %%
# Preparando a tabela base para o modelo

cols_app = [
    'appid', 'type', 'is_free', 'required_age', 
    'mat_initial_price', 'mat_discount_percent', 'mat_achievement_count', 
    'mat_supports_windows', 'mat_supports_mac', 'mat_supports_linux', 
    'supported_languages', 'metacritic_score', 'recommendations_total'
]

df_final = df_applications[cols_app].copy()

#importando generos dos jogos
app_genres = pd.read_csv('Base/application_genres.csv')
#importando generos disponiveis
genres = pd.read_csv('Base/genres.csv')
#cruzando tabela de jogos com a tabela de generos para obter apenas os generos que tem na tabela de jogos
df_genres = pd.merge(app_genres, genres, left_on='genre_id', right_on='id')
#transformando a tabela de generos em valores unicos
df_genres = df_genres.drop_duplicates(subset=['appid'])

#cruzando tabela de jogos com a tabela de generos. trazendo o app id e o nome
df_final = pd.merge(df_final, df_genres[['appid', 'name']], on='appid', how='left')
df_final.rename(columns={'name': 'genre'}, inplace=True)
#contando qtd de linguas
df_final['num_languages'] = df_final['supported_languages'].apply(lambda x: len(str(x).split(',')))


# %%
# Limpeza e Tratamento de Nulos

df_final['mat_achievement_count'] = df_final['mat_achievement_count'].fillna(0)
df_final = df_final.dropna(subset=['mat_initial_price', 'genre'])

# 1. Limpeza de 'Lixo' em colunas numéricas
# Vamos tentar converter required_age para número e o que não for número vira NaN
df_final['required_age'] = pd.to_numeric(df_final['required_age'], errors='coerce')

# 2. Agora preenchemos os NaNs (os erros de javascript viraram NaN) com a idade 0
df_final['required_age'] = df_final['required_age'].fillna(0)
df_final['metacritic_score'] = pd.to_numeric(df_final['metacritic_score'], errors='coerce')
df_final['metacritic_score'] = df_final['metacritic_score'].fillna(df_final['metacritic_score'].mean())

# Tente também o 'is_free' convertido para 0 e 1
df_final['is_free'] = df_final['is_free'].astype(int)
# %%
# 1. Carregar reviews (apenas colunas necessárias)
reviews = pd.read_csv('Base/reviews.csv', usecols=['appid', 'voted_up'])

# 2. Calcular a taxa de recomendação por jogo (0.0 a 1.0)
taxa_sucesso = reviews.groupby('appid')['voted_up'].mean().reset_index()

# 3. Definir Target: Sucesso (1) se > 80% de recomendações positivas
taxa_sucesso['target'] = (taxa_sucesso['voted_up'] > 0.8).astype(int)
reputacao_pub = pd.merge(app_publishers, taxa_sucesso[['appid', 'target']], on='appid')
status_empresa = reputacao_pub.groupby('publisher_id')['target'].mean().reset_index()
status_empresa.rename(columns={'target': 'publisher_reputation_score'}, inplace=True)

# 4. Unir ao seu df_final
df_final = pd.merge(df_final, taxa_sucesso[['appid', 'target']], on='appid', how='inner')

df_final = pd.merge(df_final, app_publishers, on='appid', how='left')
df_final = pd.merge(df_final, status_empresa, on='publisher_id', how='left')
df_final['publisher_reputation_score'] = df_final['publisher_reputation_score'].fillna(df_final['publisher_reputation_score'].mean())

df_final = df_final[df_final['type'] == 'game'].copy()

# %%
# Transformar Categóricas em Numéricas (Dummies)
df_final_model = pd.get_dummies(df_final, columns=['type', 'genre'])
df_final = df_final[df_final['recommendations_total'] > 50].copy()
# Remover colunas de texto que não servem para o cálculo
df_final_model = df_final_model.drop(columns=['supported_languages', 'appid'])
# %%
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Separar Features e Target
# 3. Verifique se existe algum outro campo de texto "escondido"
# O scikit-learn só aceita números. Vamos filtrar apenas colunas numéricas 
# e as colunas que criamos com o get_dummies
X = df_final_model.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
y = df_final_model['target']

# Dividir 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o Modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Ver o resultado
previsoes = modelo.predict(X_test)
print(classification_report(y_test, previsoes))
# %%
import matplotlib.pyplot as plt
import numpy as np

# Pegando a importância das colunas
importancias = modelo.feature_importances_
colunas = X.columns

# Criando um gráfico de barras
plt.figure(figsize=(10, 6))
indices = np.argsort(importancias)[-10:] # Top 10 variaveis
plt.barh(range(len(indices)), importancias[indices], align='center')
plt.yticks(range(len(indices)), [colunas[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.title('Quais fatores mais influenciam o Sucesso na Steam?')
plt.show()
# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, previsoes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsão da IA')
plt.ylabel('Realidade (Sucesso ou não)')
plt.title('Matriz de Confusão: Onde o modelo está errando?')
plt.show()
# %%
