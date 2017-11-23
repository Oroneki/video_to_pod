# O que esse negócio faz?
1. Verifica se há um novo podcast do programa "O É da Coisa" no YouTube e baixa o audio
2. Extrai os comerciais
3. Gera um MP3 com correção de volumes
4. Serve o arquivo por feed de podcast na rede local

# Pra rodar...
## Pré-requisitos
- ffmpeg versão 3 (https://www.ffmpeg.org/) (de preferência no PATH)
- Pipenv (https://github.com/kennethreitz/pipenv)
- Python 3.6 (https://www.python.org/)

## Configuração inicial
1. Renomear o arquivo .env 
>``` cp .env.example .env```
2. Instalar dependências
>```pipenv install```
3. Editar arquivo .env com as variáveis de sistema (instruções no próprio arquivo)
4. Rodar o programa para preparar tudo:
>```pipenv run python main.py```
5. Depois é só servir
>```pipenv run python serve.py```
6. Enjoy :)

***

## Variáveis de Ambiente do arquivo _.env_
Verifique o próprio arquivo .env (ou .env.example) para exemplos.


#### ARQUIVO_MODELO
Modelo de Machine Learning Compilado


#### PASTA_DOWNLOAD
Pasta de Download do Sistema pra baixar do YouTube


#### PASTA_SERVER
Pasta que será servida com os arquivos do podcast

OBS: Todos os arquivos da pasta serão públicos


#### PASTA_LOG
PASTA para salvar estatisticas e etc...


#### FFMPEG_3_PATH
O ideal é o ffmpeg estar no PATH do sistema. Se não estiver, informar o caminho completo.

A versão deve ser a 3 ou maior. Algumas versões do linux vem com versão 2. Atualizar pelo apt-get nem sempre funciona.

Recomendo o ppa *jonathonf/ffmpeg-3* para usuários Ubuntu ou Linux Mint.

#### SERVER_END
Endereço do servidor na rede. OBS: Configurar o roteador para IP Fixo da Máquina (RaspberryPi)

exemplo: `SERVER_END="http://192.168.1.106:8000"`

