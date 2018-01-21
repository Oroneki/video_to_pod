from handleenv import handleenv
handleenv()
import os
from datetime import datetime as dt
from feed import makeFeed, pasta
from grab import atualizaListaYouTube, baixarNovos, pasta_download
from trataAudio import transformarEAtualizar
from feed import makeFeed, pasta
from datetime import datetime as dt
import os
import tempfile
import pathlib

CONTROL_FILE = os.path.join(tempfile.gettempdir(),'oroneki_video_to_pod.oro')

def main():
    if os.path.exists(CONTROL_FILE):
        print('Já está rodando...')
        return
    pathlib.Path(CONTROL_FILE).touch()
    print('Inicio')
    try:
        h1 = dt.now()
        atualizaListaYouTube()
        h2 = dt.now()
        try:
            baixarNovos(pasta_download)
        except:
            print('Erro no download')    
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
        end = os.environ.get('SERVER_END')
        print(f'''\nACESSE EM:\n{end}/{xml}\n''')
    except Exception as exc:
        import sys
        print(exc)
        print('Erro!', sys.exc_info()[0])
    finally:
        os.remove(CONTROL_FILE)
        print('FIM')
        


if __name__ == '__main__':
    main()
