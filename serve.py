import http.server
import socketserver
import os
import sys

from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

PORT = os.environ.get('SERVER_END').split(':')[-1]
try:
    PORT = int(PORT)
except:
    print('Verificar a configuração da variável SERVER_END no arquivo .env')
    sys.exit()

pasta = os.environ.get('PASTA_SERVER')

os.chdir(pasta)

Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)
print("Servindo na porta", PORT)
httpd.serve_forever()