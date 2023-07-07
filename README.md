# Classificador KNN para identificação de cargas em TDR aplicada numa linha de transmissão

## Gerenciando dependências do projeto

Todos os comandos abaixo assumem que você esta na pasta raiz do projeto.

### Instale o gerenciador de ambientes virtuais python

```sh
sudo apt install python3.10-venv
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

No ambiente virtual, utilize o comando abaixo para executar uma rotina de treino e avaliação pré programada

```sh
python3 Knearest.py 
```
