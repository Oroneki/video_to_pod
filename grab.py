import os
import re
from datetime import datetime
import requests
import youtube_dl
from bs4 import BeautifulSoup as BS4

from model import Podcast

pasta_download = os.environ['PASTA_DOWNLOAD']


def atualizaListaYouTube():

    reg_search = re.compile(r'.*O\s+É\s+da\s+Coisa.*\d{2}.*')
    reg_id = re.compile(r"\/watch\?v=(?P<id>.*)")

    r = requests.get(
        "https://www.youtube.com/channel/UCWijW6tW0iI5ghsAbWDFtTg/videos", verify=False)
    soup = BS4(r.content, "html.parser")
    col = soup.find_all(text=reg_search)

    for el in col:

        youtube_id = reg_id.match(el.parent['href']).group('id')
        try:
            Podcast.get(Podcast.youtube_id == youtube_id)
            continue
        except:
            pod = Podcast(
                youtube_id=youtube_id,
                nome=str(el),
                data=getData(el),
                fase=0,
                baixado=False
            )
            print(pod)
            pod.save()

    return col


def getData(string_com_data):
    reg_data = re.compile(r"\d{2}\/\d{2}\/\d{4}")
    res = reg_data.search(string_com_data).group()
    return datetime.strptime(res, "%d/%m/%Y").date()


def baixaLista(pasta_downloads, lista_de_ids_):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': str(os.path.join(pasta_downloads, 'REI_%(id)s.%(ext)s')),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio'
        }],
        'nocheckcertificate': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(lista_de_ids_)


def baixarNovos(pasta_downloads):
    lista_ids = []
    for pod in Podcast.select().where(Podcast.baixado == False).order_by(Podcast.data):
        lista_ids.append(pod.youtube_id)
    if len(lista_ids) > 0:
        baixaLista(pasta_downloads, lista_ids)
    else:
        print('Nada pra baixar... :)')
        return
    print('Atualizar banco de dados...')
    for _, _, la in os.walk(pasta_downloads):
        for arq in la:
            if not arq.startswith('REI_'):
                continue
            if arq.endswith('.part'):
                continue
            for id_ in lista_ids:
                if id_ in arq:
                    pod = Podcast.get(Podcast.youtube_id == id_)
                    pod.arquivo_baixado = arq
                    pod.baixado = True
                    pod.fase = 1
                    pod.baixado_em = datetime.now()
                    pod.save()
        break


def main():
    atualizaListaYouTube()


if __name__ == '__main__':
    main()
