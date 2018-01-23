import os
import json
from model import Podcast
from feedgen.feed import FeedGenerator

pasta = os.environ['PASTA_SERVER']
server_end = os.environ['SERVER_END']


def makeFeed(pasta):

    nome_arquivo_xml = 'edacoisa.xml'

    endereco_feed = os.path.join(pasta, nome_arquivo_xml)
    fg = FeedGenerator()

    fg.title('O É da Coisa - BandNewsFM')
    fg.author({'name': 'Reinaldo Azevedo', 'email': 'john@example.de'})
    fg.logo(f'{server_end}/edacoisa.jpg')
    fg.image(f'{server_end}/edacoisa.jpg')
    fg.subtitle('BandNewsFM')
    fg.description('O E da Coisa - BandNewsFM')
    fg.link(href=f'{server_end}/edacoisa.xml', rel='self')
    fg.language('pt-br')

    for _, pod in enumerate(Podcast.select().where(Podcast.fase == 2)):
        fe = fg.add_entry()
        fe.guid(pod.youtube_id)
        fe.title(pod.nome.replace(', com Reinaldo Azevedo -', ''))
        fe.link(href=f'{server_end}/{pod.arquivo_podcast}')
        stats = json.loads(pod.stats)
        corte = int(stats['segundos_cortados']) / 60
        estatisticas = json.dumps(stats, sort_keys=True, indent=2)
        fe.description(f'O É da Coisa - BandNewsFM.\n{pod.nome}\n{pod.youtube_id}\n{corte:.2f} minutos cortados.\n\n{estatisticas}')

    fg.rss_file(endereco_feed)
    return nome_arquivo_xml


def main():
    makeFeed(pasta)


if __name__ == '__main__':
    main()
