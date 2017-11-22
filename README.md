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
4. Rodar o programa (usar a mesma porta do arquivo .env):
>```pipenv run python main.py && pipenv run python -m http.server 8000```
5. Enjoy :)

## Variáveis de Ambiente do arquivo _.env_
```
# Modelo de Machine Learning Compilado
ARQUIVO_MODELO="novo_modelo.pkl"

# Pasta de Download do Sistema pra baixar do YouTube
PASTA_DOWNLOAD="/home/USER/Downloads"

# Pasta que será servida com os arquivos do podcast
# OBS: Todos os arquivos da pasta serão públicos
PASTA_SERVER="/home/USER/Downloads/podcasts"

# PASTA para salvar estatisticas e etc...
PASTA_LOG="/home/USER/Downloads/logs"

# O ideal é o ffmpeg estar no PATH do sistema. Se não estiver, informar o caminho completo abaixo
# ex: "/usr/bin/ffmpeg"
FFMPEG_3_PATH="ffmpeg"

# Endereço do servidor na rede. OBS: Configurar o roteador para IP Fixo da Máquina (RaspberryPi)
# SERVER_END="http://fixed_ip_number:port"
SERVER_END="http://192.168.1.106:8000"
```
