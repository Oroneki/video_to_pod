# O que esse negócio faz?
1. Verifica se há um novo podcast do programa "O É da Coisa" no YouTube e baixa o audio
2. Extrai os comerciais
3. Gera um MP3 com correção de volumes
4. Serve o arquivo por feed de podcast na rede local

# Pra rodar...

## Pré-requisitos
- ffmpeg versão 3 (https://www.ffmpeg.org/) (de preferência no PATH)
- Python 3.6 (https://www.python.org/) -> Pro raspberrypi prefira o berryconda (https://github.com/jjhelmus/berryconda)
- No Arch Linux precisei instalar o tkinter também: *pacman -S tk*

## Preparo (opcional - mas recomendado)
1. Crie um ambiente virtual
>```python3.6 -m venv /caminho/pro/seu/ambiente```
2. Carregue seu ambiente
>```source /caminho/pro/seu/ambiente/bin/activate```

## Configuração inicial (apenas na primeira vez)
1. Instalar os pacotes 
>```pip install -r requirements.txt```
2. Compilar o modelo a partir dos dados (O compilador vai gerar o arquivo **novo_modelo_treinado.pkl**)
>```python treinador.py```
3. Gerar o arquivo .env e **editar as variáveis de sistema** de acordo com seu ambiente (instruções no próprio arquivo - vide abaixo)
>```cp .env.example .env```

## Pra rodar
1. Baixar, extrair audio, converter, compactar e atualizar feed:
>```python main.py```
2. Servir o feed RSS do podcast:
>```python serve.py```
3. Enjoy :)

***

## Variáveis de Ambiente do arquivo _.env_
Verifique o próprio arquivo .env (ou .env.example) para exemplos.


### ARQUIVO_MODELO
Caminho do arquivo com o modelo de Machine Learning compilado


### PASTA_DOWNLOAD
Pasta de Download do Sistema pra baixar do YouTube


### PASTA_SERVER
Pasta que será servida com os arquivos do podcast

OBS: Todos os arquivos da pasta serão públicos


### PASTA_LOG
PASTA para salvar estatisticas e etc...


### FFMPEG_3_PATH
O ideal é o ffmpeg estar no PATH do sistema. Se não estiver, informar o caminho completo.

A versão deve ser a 3 ou maior. Algumas versões do linux vem com versão 2. Atualizar pelo apt-get nem sempre funciona.

Recomendo o ppa *jonathonf/ffmpeg-3* para usuários Ubuntu ou Linux Mint.

### SERVER_END
Endereço do servidor na rede. OBS: Configurar o roteador para IP Fixo da Máquina (RaspberryPi)

exemplo: `SERVER_END="http://192.168.1.106:8000"`

### MPLBACKEND
Backend do matplotlib: para rodar num servidor e gerar as imagens de log sem monitor (raspberry pi).

### TEMPORARIO_DIR
Pasta de arquivos temporários