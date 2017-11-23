
from grab import atualizaListaYouTube, baixarNovos, pasta_download
from trataAudio import transformarEAtualizar
from feed import makeFeed, pasta
from datetime import datetime as dt
import os

def main():
    print('Inicio')
    h1 = dt.now()
    atualizaListaYouTube()
    h2 = dt.now()
    baixarNovos(pasta_download)
    h3 = dt.now()
    transformarEAtualizar()
    h4 = dt.now()
    xml = makeFeed(pasta)
    h5 = dt.now()
    print('CONCLUIDO!')
    print('INICIO :', h1, sep="\t")
    print('TERMINO:', h5, sep ="\t")
    print('baixar :', h3-h2, sep ="\t")
    print('magica :', h4-h3, sep ="\t")
    print('TEMPO  :', h5-h1, sep ="\t")
    print(f'''\nACESSE EM:\n{os.environ.get("SERVER_END")/{xml}}\n''')

    
if __name__ == '__main__':
    main()
