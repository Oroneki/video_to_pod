from feedgen.feed import FeedGenerator
import os
from model import Podcast

pasta = os.environ['PASTA_SERVER']
server_end = os.environ['SERVER_END']


def makeFeed(pasta):

    endereco_feed = os.path.join(pasta, 'edacoisa.xml')
    fg = FeedGenerator()

    fg.title('O É da Coisa - BandNewsFM')
    fg.author( {'name':'Reinaldo Azevedo','email':'john@example.de'} )
    fg.logo(f'{server_end}/edacoisa.jpg')
    fg.image(f'{server_end}/edacoisa.jpg')
    fg.subtitle('BandNewsFM')
    fg.description('O E da Coisa - BandNewsFM')
    fg.link( href=f'{server_end}/edacoisa.xml', rel='self' )
    fg.language('pt-br')

    for i, pod in enumerate(Podcast.select().where(Podcast.fase == 2)):
        fe = fg.add_entry()
        fe.guid(pod.youtube_id)
        fe.title(pod.nome.replace(', com Reinaldo Azevedo -', ''))
        fe.link(href=f'{server_end}/{pod.arquivo_podcast}')
        fe.description(f'O É da Coisa - BandNewsFM.\n{pod.nome}\n{pod.youtube_id}')

    fg.rss_file(endereco_feed)
    return endereco_feed

def main():
    makeFeed(pasta)

if __name__ == '__main__':
    main()
