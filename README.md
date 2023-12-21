# Códigos e métodos para filtrar, visualizar e classificar sinais obtidos através de TDR numa linha de transmissão

Esse repositório foi elaborado para avaliações diversas sobre series temporais e fornece diferentes pacotes.

## Funcionalidades

### Pacote dados

Conjunto de módulos que leem, armazenam e mantém os dados sobre os quais serão realizadas análises.

#### dados

Implementa a classe principal de armazenamento dos dados, com funções de remoção de outliers, equalização de amostras, redução da taxa de amostragem e também para impressão das amostras obtidas, por classe em um gráfico.

#### leitura

Implementa funcionalidades para obtenção de um objeto Dados com todas as amostras em determinada pasta. Espera encontrar cada classe em um arquivo .csv diferente e a ordem das classes pode ser enfatizada adicionando uma numeração no início do nome, seguida de um '-' exemplso: '01-classe1.csv' '02-classe2.csv'.

#### escrita

Implementa uma funcionalidade para salvar os dados em formato csv, por exemplo após aplicação de remoção de outliers ou redução de amostragem, de forma que fica salvo. Cada classe terá um arquivo diferente

### Pacote metricas

Oferece ferramentas para manipulações como filtragens, ordenamentos dos dataframes e classificações. Retornam diferentes tipos de resultados.

#### dividir_amostras

Implementa ambas modalidades de divisão de amostras: treino regular, que oferece um split basico estratificado, e treino media que retorna um conjunto de treino com acumulação das amostras.

#### metricas

Implementa diversas funcionalidade com objetivo de realizar análises de distância e dispersão dos valores em dataframes.

#### previsao_correlacao

Implementa uma série de funcionalidades em cadeia para realização de múltiplas classificações com parâmetros variaveis para os classificadores baseados em correlação, permitindo a inserção de multiplos tamanhos de treino e outros parâmetros de configuração como número de repetições para gerar as médias, salvar ou não as matrizes de confusão e os relatórios de classificação e número de repetições para gerar a moda da classificação, por instância de classificação. Retorna um dicionário com os valores médios obtidos para tempo de execução, pico de consumo de memória e acurácia.

#### previsao_correlacao_multi_processos

Utiliza as mesmas funcionalidades da previsão_correlação porém com auxílio de processamento paralelo para agilizar o processo. Por isso, pico de consumo de memória e tempo de execução não são salvos no resultado final, apenas as acurácias. Nome das funções são iguais e funcionam de forma análoga, mas tem parâmetros opcionais a mais.

#### previsao_KNN

Implementa uma série de funcionalidades em cadeia para realização de múltiplas classificações com parâmetros variaveis para os classificadores baseados em KNN, permitindo a inserção de multiplos tamanhos de treino e valores para k, bem como diferentes métodos para divisão das amostras. Outros parâmetros de configuração como número de repetições para gerar as médias, salvar ou não as matrizes de confusão e os relatórios de classificação e número de repetições para gerar a moda da classificação, por instância de classificação também são oferecidos. Retorna um dicionário com os valores médios obtidos para tempo de execução, pico de consumo de memória e acurácia.

#### previsao_KNN_multi_processos

Utiliza as mesmas funcionalidades da previsão_KNN porém com auxílio de processamento paralelo para agilizar o processo. Por isso, pico de consumo de memória e tempo de execução não são salvos no resultado final, apenas as acurácias. Nome das funções são iguais e funcionam de forma análoga, mas tem parâmetros opcionais a mais.

### Pacote modelos

#### classificadores_correlação

Implementação dos classificadores baseados em correlação.

### Pacote visualização

Consiste de  diversas ferramentas gráficas configuráveis para exibição das métricas obtidas com o pacote métricas ou dos resultados de previsões do pacote modelos.

#### tabelas

Gera tabelas específicas, não genéricas tipo latex a partir dos resultados das classificações. Provavelmente precisará adaptar as funções para que os títulos e a quantidade de testes sejam adequadas ao seu caso.

#### visualizacao

Contém multiplas implementações com diferentes plotagens gráficas de resultados e dispersões.

## Gerenciando dependências do projeto

Todos os comandos abaixo assumem que você esta na pasta raiz do projeto.

### Instale o gerenciador de ambientes virtuais python e as depts

```sh
sudo apt install python3.10-venv
sudo apt install python3-tk
python3 -m pip install ---user virtualenv
```

### Crie um ambiente virtual para o projeto com o comando

```sh
python3 -m venv env
```

### Entrando e saindo do ambiente

Para ativar o ambiente:

```sh
source env/bin/activate
```

Para desativar o ambiente:

```sh
deativate
```

### Instalando as dependências do projeto

Com o ambiente virtual ativado:

```sh
python3 -m pip install -r requirements.txt
```

## Executando

No ambiente virtual, utilize os exemplos em `classificar.py` para gerar suas próprias rotinas de testes.  

`gerar_tabelas.py` é um exemplo do uso das funções geradoras de tabelas latex que depende dos resultados de classificação serem salvos em arquivos .pkl.

`imprimir_amostras.py` é um exemplo do uso das funções geradores de plotagens gráficas.

Um exemplo de uso é:  

```bash
python classificar.py
```
