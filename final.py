import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("--- Etapa 1: Coleta e Preparação dos Dados ---")
data = {
    'data_venda': pd.to_datetime(pd.date_range(start='2023-01-01', periods=365, freq='D')),
    'valor_venda': np.random.uniform(50, 500, 365).round(2),
    'categoria_produto': np.random.choice(['Eletrônicos', 'Vestuário', 'Casa e Jardim'], 365),
    'custo_marketing': np.random.uniform(10, 100, 365).round(2)
}
df_vendas = pd.DataFrame(data)
df_vendas.loc[df_vendas.sample(frac=0.05).index, 'valor_venda'] = np.nan
media_vendas = df_vendas['valor_venda'].mean()
df_vendas['valor_venda'].fillna(media_vendas, inplace=True)
print("Dados preparados.\n")

print("--- Etapa 2: Cálculos da Análise Exploratória ---")
print("Estatísticas descritivas do valor de venda:")
print(df_vendas['valor_venda'].describe().round(2))
vendas_por_categoria = df_vendas.groupby('categoria_produto')['valor_venda'].sum().sort_values(ascending=False)
print("\nCálculos para os gráficos concluídos.")

print("\n--- Etapa 3: Aplicação de Estatística Básica (Teste ANOVA) ---")
vendas_eletronicos = df_vendas[df_vendas['categoria_produto'] == 'Eletrônicos']['valor_venda']
vendas_vestuario = df_vendas[df_vendas['categoria_produto'] == 'Vestuário']['valor_venda']
vendas_casa_jardim = df_vendas[df_vendas['categoria_produto'] == 'Casa e Jardim']['valor_venda']
f_statistic, p_valor = f_oneway(vendas_eletronicos, vendas_vestuario, vendas_casa_jardim)
print(f"Resultado do Teste ANOVA (P-valor): {p_valor:.4f}")
if p_valor < 0.05:
    print("Conclusão: Há uma diferença significativa nas vendas entre as categorias.")
else:
    print("Conclusão: Não há diferença significativa nas vendas entre as categorias.")

print("\n--- Etapa 4: Modelagem de Machine Learning ---")
X = df_vendas[['custo_marketing']]
y = df_vendas['valor_venda']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Modelo treinado. Performance:")
print(f"  - Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"  - Coeficiente de Determinação (R²): {r2:.2f}")

print("\n--- Etapa 5: Gerando Dashboard de Visualização ---")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), constrained_layout=True)
fig.suptitle('Dashboard de Análise de Vendas', fontsize=20)

axes[0, 0].set_title('1. Tendência de Vendas ao Longo do Tempo')
sns.lineplot(ax=axes[0, 0], x='data_venda', y='valor_venda', data=df_vendas)
axes[0, 0].set_xlabel('Data')
axes[0, 0].set_ylabel('Valor da Venda (R$)')

axes[0, 1].set_title('2. Distribuição dos Valores de Venda')
sns.histplot(ax=axes[0, 1], data=df_vendas['valor_venda'], kde=True, bins=30)
axes[0, 1].set_xlabel('Valor da Venda (R$)')
axes[0, 1].set_ylabel('Frequência')

axes[1, 0].set_title('3. Total de Vendas por Categoria')
sns.barplot(ax=axes[1, 0], x=vendas_por_categoria.index, y=vendas_por_categoria.values)
axes[1, 0].set_xlabel('Categoria do Produto')
axes[1, 0].set_ylabel('Soma Total das Vendas (R$)')

axes[1, 1].set_title('4. Performance do Modelo: Real vs. Previsto')
sns.scatterplot(ax=axes[1, 1], x=y_test, y=y_pred, alpha=0.6)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
axes[1, 1].set_xlabel('Vendas Reais')
axes[1, 1].set_ylabel('Vendas Previstas')
axes[1, 1].text(0.05, 0.9, f'R² = {r2:.2f}', transform=axes[1, 1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

print("\nDashboard gerado com sucesso!")