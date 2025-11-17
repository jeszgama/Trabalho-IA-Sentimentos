# Classificação de Sentimentos em Comentários de Notícias com BERT

Este projeto estabelece um pipeline de Processamento de Linguagem Natural (PLN) que utiliza modelos da família BERT para classificar comentários de notícias em três tipos de sentimento: negativo, neutro e positivo.  O estudo abrange todas as fases fundamentais do processo: preparação dos dados, tokenização, definição do modelo, treinamento, avaliação e análise de erros.
Todos os resultados produzidos durante o experimento podem ser vistos diretamente no notebook.

---

## 1. Objetivo

Treinar modelos independentes para cada coluna de interesse do conjunto de dados, avaliando o desempenho do BERT na classificação de textos curtos.
Foram utilizadas as colunas:

* onça
* caseiro
* notícia

---

## 2. Base de Dados

Os dados estão disponíveis em:

Google Sheets:
[https://docs.google.com/spreadsheets/d/17aHYyRNfbmde8bVOR_HX_BmNUEdkygPuaGO4lJj26jg/edit?usp=sharing](https://docs.google.com/spreadsheets/d/17aHYyRNfbmde8bVOR_HX_BmNUEdkygPuaGO4lJj26jg/edit?usp=sharing)

Os rótulos foram convertidos para valores numéricos:

* 0 = negativo
* 1 = neutro
* 2 = positivo

---

## 3. Preparação dos Dados

As seguintes etapas foram realizadas:

* Leitura e limpeza dos dados
* Remoção de valores nulos ou duplicados
* Conversão dos rótulos para inteiros
* Divisão estratificada em:

  * Treino: 70%
  * Validação: 15%
  * Teste: 15%

---

## 4. Tokenização

Foi utilizado o tokenizador do modelo:

```
neuralmind/bert-base-portuguese-cased
```

Para cada texto foram gerados os tensores `input_ids` e `attention_mask`, necessários para o processamento pelo BERT

---

## 5. Modelo

A arquitetura base é:

```
AutoModelForSequenceClassification
```

Configurações principais:

* `num_labels = 3`
* Utilização explícita do token [CLS]
* Treinamento em precisão mista (FP16), com `autocast` e `GradScaler`

---

## 6. Treinamento

Configurações adotadas:

* Otimizador: AdamW
* Learning rate: 2e-5
* Número de épocas: 10
* Batch size ajustado conforme a capacidade da GPU

Durante cada época, o modelo apresenta:

* Loss e acurácia de treino
* Loss e acurácia de validação

Ao final, é exibido o gráfico de evolução do loss (treino vs validação)

---

## 7. Avaliação

A avaliação é realizada sobre o conjunto de teste (15%)

### Relatório de Classificação (sklearn)

Para cada classe são apresentados:

* Precision
* Recall
* F1-score

### Análise de Erros

O notebook exibe textos em que o modelo falhou, incluindo:

* Texto original
* Rótulo verdadeiro
* Rótulo previsto

### Exportação de Resultados

As previsões são salvas em arquivos CSV no formato:

```
resultados_onca_teste.csv
```

---

## 9. Execução

1. Abrir o notebook no Google Colab e rodar o código
