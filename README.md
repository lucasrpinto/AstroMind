# AstroMind

AstroMind é um projeto (Em andamento) de Machine Learning em Python voltado para análise e classificação de imagens astronômicas obtidas a partir de dados públicos dos telescópios espaciais Hubble e James Webb.

O objetivo inicial do projeto é construir um pipeline capaz de baixar dados científicos no formato FITS, processar esses arquivos, converter as imagens para formatos utilizáveis em visão computacional e, posteriormente, treinar um modelo para classificar objetos astronômicos.

## Objetivo do projeto

O projeto tem como objetivo principal desenvolver uma solução de visão computacional capaz de identificar e classificar imagens astronômicas em categorias como:

- Galáxias
- Estrelas
- Nebulosas
- Aglomerados
- Ruído ou artefatos instrumentais
- Objetos muito brilhantes
- Possíveis anomalias

Nesta primeira etapa, o foco está na construção do pipeline de dados, desde o download dos arquivos astronômicos até o pré-processamento das imagens.

## Fonte dos dados

Os dados utilizados no projeto são obtidos a partir do MAST, Mikulski Archive for Space Telescopes, arquivo público mantido pelo Space Telescope Science Institute.

O MAST disponibiliza dados científicos de diversas missões astronômicas, incluindo:

- Hubble Space Telescope, HST
- James Webb Space Telescope, JWST
- TESS
- Kepler
- entre outras missões

Os arquivos baixados são, em geral, disponibilizados no formato FITS, padrão amplamente utilizado em astronomia.

## Tecnologias utilizadas

O projeto utiliza as seguintes tecnologias e bibliotecas:

- Python
- Astroquery
- Astropy
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn
- PyTorch
- Torchvision

## Estrutura inicial do projeto

```text
ml-telescopios/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── images/
│   │   └── arrays/
│   └── labels/
│
├── notebooks/
│   └── 01_explorar_fits.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
│
├── models/
│
├── outputs/
│   ├── images/
│   └── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Etapas já implementadas

### 1. Estrutura inicial do projeto

Foi criada a estrutura base de pastas e arquivos para organizar o projeto.

### 2. Download dos dados astronômicos

O arquivo src/download_data.py realiza a busca e o download de arquivos FITS públicos disponíveis no MAST.

Nesta etapa, o projeto busca observações astronômicas, filtra produtos científicos em formato FITS e salva os arquivos na pasta:
```
data/raw/
```
Também são gerados relatórios CSV com informações sobre as observações encontradas, os produtos selecionados e o manifesto de download.

Os relatórios são salvos em:
```
outputs/reports/
```
### 3. Pré-processamento dos arquivos FITS

O arquivo ```src/preprocess.py``` realiza o processamento inicial dos arquivos FITS.

Esse processamento inclui:

- leitura dos arquivos FITS;
- identificação da imagem científica dentro do arquivo;
- tratamento de valores inválidos;
- normalização dos dados;
- redimensionamento da imagem;
- conversão para PNG;
- salvamento da imagem como array NumPy.

As imagens convertidas são salvas em:

```data/processed/images/```

Os arrays NumPy são salvos em:

```data/processed/arrays/```

## Como instalar o projeto

Clone o repositório:

git clone https://github.com/seu-usuario/AstroMind.git

Acesse a pasta do projeto:

```cd AstroMind```

Crie e ative um ambiente virtual:
```
python -m venv .venv
source .venv/bin/activate
```
Instale as dependências:

```pip install -r requirements.txt```

## Como baixar os dados

Execute o comando abaixo na raiz do projeto:

```python -m src.download_data```

Esse comando irá buscar dados públicos no MAST e salvar os arquivos FITS em:

```data/raw/```

## Como processar os arquivos FITS

Depois de baixar os arquivos, execute:

```python -m src.preprocess```

Esse comando irá converter os arquivos FITS em imagens PNG e arrays NumPy.

## Status atual do projeto

Atualmente, o projeto já possui:

- estrutura inicial organizada;
- configuração centralizada em src/config.py;
- download de dados públicos do MAST;
- processamento de arquivos FITS;
- conversão de imagens astronômicas para PNG;
- geração de arrays NumPy;
- geração de relatórios CSV.

## Próximas etapas

As próximas etapas previstas são:

1. Criar o arquivo de rótulos labels.csv;
2. Implementar o carregamento das imagens com src/dataset.py;
3. Criar o primeiro modelo de classificação com PyTorch;
4. Treinar o modelo com imagens astronômicas rotuladas;
5. Criar o arquivo src/predict.py para realizar previsões;
6. Expandir a base de dados com diferentes tipos de objetos astronômicos;
7. Melhorar o pré-processamento das imagens;
8. Criar uma API para disponibilizar a classificação das imagens.

## Objetivo final

O objetivo final do AstroMind é criar uma inteligência artificial capaz de analisar imagens astronômicas, identificar padrões visuais e fornecer uma classificação inicial sobre o tipo de objeto observado.

No futuro, o projeto poderá evoluir para combinar visão computacional, metadados astronômicos e técnicas de processamento de linguagem natural para gerar explicações mais completas sobre os dados analisados.

## Autor

Projeto desenvolvido por Lucas Rodrigo Pinto como estudo prático de Machine Learning, Visão Computacional, Python e análise de dados astronômicos.
