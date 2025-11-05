# 📊 Pipeline de Análise de Vendas com Python

Este repositório contém um script Python que demonstra um pipeline completo de análise de dados. O script simula todo o processo, desde a geração de dados sintéticos e preparação, passando por análise estatística, modelagem de machine learning, até a visualização final em um dashboard.

Este projeto serve como um exemplo prático de como integrar diversas bibliotecas populares de Data Science (Pandas, Scikit-learn, Matplotlib/Seaborn) em um único fluxo de trabalho coeso.

## 🚀 Funcionalidades Principais

* **Geração de Dados Sintéticos:** Cria um conjunto de dados de vendas (Data, Valor, Categoria, Custo de Marketing) usando `pandas` e `numpy`.
* **Limpeza de Dados:** Simula e trata valores ausentes (NaN) usando preenchimento pela média.
* **Análise Exploratória (EDA):** Calcula estatísticas descritivas básicas e agrega dados para visualização.
* **Teste de Hipótese:** Aplica um teste ANOVA (usando `scipy.stats`) para verificar se há diferenças estatisticamente significativas nas vendas entre diferentes categorias de produtos.
* **Machine Learning:** Treina um modelo de Regressão Linear Simples (usando `scikit-learn`) para prever o `valor_venda` com base no `custo_marketing`.
* **Visualização de Dados:** Gera um "dashboard" 2x2 com `matplotlib` e `seaborn` para apresentar os resultados da análise.

---

## 📈 O Pipeline de Análise (Etapas)

O script é dividido nas seguintes etapas sequenciais:

1.  **Coleta e Preparação dos Dados:**
    * Cria um DataFrame com 365 dias de dados de vendas.
    * Insere 5% de valores nulos nas vendas e os preenche com a média da coluna.

2.  **Cálculos da Análise Exploratória:**
    * Calcula estatísticas descritivas (média, mediana, desvio padrão, etc.) para `valor_venda`.
    * Agrupa as vendas totais por `categoria_produto`.

3.  **Aplicação de Estatística (ANOVA):**
    * Testa se as médias de vendas das categorias ('Eletrônicos', 'Vestuário', 'Casa e Jardim') são significativamente diferentes.
    * Imprime o P-valor e uma conclusão no console.

4.  **Modelagem de Machine Learning:**
    * Separa os dados em conjuntos de treino e teste.
    * Treina um modelo de Regressão Linear.
    * Avalia o modelo usando o Erro Quadrático Médio (MSE) e o Coeficiente de Determinação (R²).

5.  **Geração do Dashboard:**
    * Cria uma única figura com quatro subplots para visualizar os principais insights da análise.

---

## 🖥️ Visualização Final (Dashboard)

O script exibe um dashboard composto por quatro gráficos principais:

1.  **Tendência de Vendas ao Longo do Tempo:** Um gráfico de linha mostrando a variação das vendas diárias.
2.  **Distribuição dos Valores de Venda:** Um histograma com uma curva de densidade (KDE) mostrando a frequência dos valores de venda.
3.  **Total de Vendas por Categoria:** Um gráfico de barras comparando a soma total das vendas para cada categoria de produto.
4.  **Performance do Modelo (Real vs. Previsto):** Um gráfico de dispersão que compara os valores reais de venda com as previsões do modelo de regressão, incluindo o R² no gráfico.

### Exemplo do Resultado

(Ao executar o script, esta janela será exibida)

![Exemplo do dashboard gerado pelo script](https://i.imgur.com/seu_nome_de_imagem.png)
*> **Nota:** Substitua o link acima por uma captura de tela do dashboard gerado após executar o script.*

---

## 🔧 Tecnologias Utilizadas

Este projeto utiliza as seguintes bibliotecas Python:

* **Pandas:** Para manipulação e análise de dados.
* **NumPy:** Para operações numéricas e geração de dados.
* **Matplotlib:** Para a criação dos gráficos e do dashboard.
* **Seaborn:** Para visualização estatística de dados (baseada no Matplotlib).
* **Scikit-learn (sklearn):** Para a divisão de dados e o modelo de Regressão Linear.
* **SciPy:** Para a execução do teste estatístico ANOVA.

---

## ▶️ Como Executar

Para rodar este projeto em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    (Você pode criar um arquivo `requirements.txt` com as bibliotecas abaixo)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```

4.  **Execute o script Python:**
    (Assumindo que o nome do arquivo seja `analise_vendas.py`)
    ```bash
    python analise_vendas.py
    ```

Ao executar, os resultados dos testes e da performance do modelo serão impressos no console, e uma janela pop-up exibirá o dashboard de visualização.