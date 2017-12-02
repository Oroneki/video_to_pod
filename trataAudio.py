import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from subprocess import PIPE, run
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn import preprocessing as pp
from sklearn.externals import joblib

from algoritmos import main as algmain
from feed import pasta
from grab import pasta_download
from model import Podcast

arquivo_modelo = os.environ.get('ARQUIVO_MODELO', None)
if not arquivo_modelo:
    print('É necessário especificar um modelo no arquivo .env! Leia as instruções.')
    sys.exit()
if not os.path.isfile(arquivo_modelo):
    print('Arquivo de modelo não existe: {}'.format(arquivo_modelo))
    sys.exit()

pasta_log = os.environ.get('PASTA_LOG', None)
if not pasta_log:
    pasta_log = pasta_download

ffcmd = os.environ.get('FFMPEG_3_PATH', 'ffmpeg')


def smoothSeq(seq, step):
    l = len(seq)
    new_seq = []
    for a in range(0, l, step):
        sub = seq[a:a + step]
        med = sum(sub) / step
        for _ in range(len(sub)):
            new_seq.append(med)
    return np.array(new_seq)


def decisaoSimples(seq_prob_ok, seq_prob_not_ok, seq_silencio):
    if not len(seq_prob_ok) == len(seq_prob_not_ok) == len(seq_silencio):
        print('Sequencias devem ser iguais')
        return None
    nova_seq = []
    for b, r, s in zip(seq_prob_ok, seq_prob_not_ok, seq_silencio):
        if s > 0.85:
            nova_seq.append(0)
            continue
        if r - b > 0.15:
            nova_seq.append(0)
            continue
        else:
            if b > 0.6:
                nova_seq.append(1)
                continue
            else:
                if r > 0.6:
                    nova_seq.append(0)
                    continue
                else:
                    nova_seq.append(1)
                    continue
    return nova_seq


def facaAmagica(arquivo_de_audio, novo_nome):

    arquivo_de_audio, tmp_dir = fmpeg_convert_to_ogg(arquivo_de_audio)

    rfc = joblib.load(arquivo_modelo)  # atualizar modelo
    rate = sf.info(arquivo_de_audio).samplerate
    channels = sf.info(arquivo_de_audio).channels
    endian = sf.info(arquivo_de_audio).endian
    block_gen = sf.blocks(arquivo_de_audio, blocksize=rate)
    tudo = []
    print('Iterando pelos blocos...')
    for u, bl in enumerate(block_gen):
        y = np.mean(bl, axis=1)
        m1 = librosa.feature.melspectrogram(y)
        lis = []
        for el in m1:
            lis.append(el.mean())
        tudo.append(lis)
        time.sleep(.005)
        if u % 10 == 0:
            print('.', end='')
        if u % 1500 == 0:
            print('!', end='\n')

    max_blocks = len(tudo)
    print(max_blocks, 'blocos.')
    mm = pp.MinMaxScaler()
    tudo = mm.fit_transform(np.array(tudo).transpose()).transpose()
    print('transfomando pra escalar')
    probs = rfc.predict_proba(tudo)
    print('Array com probabilidades ok')
    dfprobs = pd.DataFrame(probs, columns=rfc.classes_)
    print('dataframe')
    seq_diff = dfprobs['COMERCIAL'] + dfprobs['SILENCIO'] - dfprobs['REI']
    decisao_seq = algmain(seq_diff)
    print('temos a sequencia de 0 e 1s')

    plt.figure(figsize=(16, 6))
    plt.plot(seq_diff, 'y-', alpha=0.8)
    plt.plot(decisao_seq, 'r-', alpha=1)
    plt.savefig(os.path.join(pasta_log, novo_nome + '.png'))

    block_gen = sf.blocks(arquivo_de_audio, blocksize=rate)

    dst = os.path.join(pasta, novo_nome + '.mp3')
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        src = os.path.join(tmpdirname, 'f.ogg')
        sffile = sf.SoundFile(
            src,
            'w',
            samplerate=rate,
            channels=channels,
            format='ogg',
            subtype='vorbis',
            endian=endian
        )

        print('iterar novamente')

        # print('ufa... gravar audio.')
        for i, bl in enumerate(block_gen):
            if decisao_seq[i] == 1:
                continue
            # print(bl)
            # sys.exit()
            sffile.write(bl)
            if i % 80 == 0:
                time.sleep(.01)
                print('.', end='')
        sffile.close()

        mp3_convert = convertToMP3(src, tmpdirname)

        shutil.move(mp3_convert, dst)
        tmp_dir.cleanup()
    return dst


def transformarEAtualizar():
    for pod in Podcast.select().where(Podcast.fase == 1):
        new_file = '{}_{}_{}-{}'.format(
            pod.data.year,
            pod.data.month,
            pod.data.day,
            'edacoisa'
        )
        print(new_file)
        pod_file = facaAmagica(
            os.path.join(pasta_download, pod.arquivo_baixado),
            new_file
        )
        pod.arquivo_podcast = os.path.basename(pod_file)
        pod.fase = 2
        pod.save()


def fmpeg_convert_to_ogg(intro_file):
    dire = tempfile.TemporaryDirectory()

    dst = os.path.join(dire.name, 'temp_media.ogg')

    ll = [
        ffcmd,
        '-i',
        intro_file,
        '-c:a',
        'libvorbis',
        dst
    ]

    run(ll, stdout=PIPE)

    return dst, dire


def convertToMP3(intro_file, dirname):

    outfile = os.path.join(dirname, 'output.mp3')

    ll = [
        ffcmd,
        '-i',
        intro_file,
        '-codec:a',
        'libmp3lame',
        '-qscale:a',
        '7',
        '-filter:a',  # apenas pra ffmpeg acima da versao 3
        'loudnorm=I=-5',  # esse tb
        outfile
    ]

    run(ll, stdout=PIPE)

    return outfile


def main():
    print(datetime.now())
    transformarEAtualizar()
    print(datetime.now())


if __name__ == '__main__':
    main()
