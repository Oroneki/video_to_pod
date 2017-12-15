import numpy as np
from sklearn.externals import joblib
import soundfile as sf

from sklearn import preprocessing as pp
# from sklearn.ensemble import RandomForestClassifier as RFC
import sys
import os
import shutil
import time
import tempfile
from model import Podcast
from datetime import datetime
from subprocess import run, PIPE
import pathlib
import random
import string

from feed import pasta
from grab import pasta_download
from algoritmos import main as algmain, labels_from_0e1s
from matematicas import AudioFile

TMP_DIR = os.environ.get('TEMPORARIO_DIR', None)

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


def facaAmagica(arquivo_de_audio, 
    novo_nome, 
    log_png=True, 
    log_output_labels=True, 
    keep_files = False
    ):

    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    PASTA_TEMP = pathlib.Path(TMP_DIR).joinpath(novo_nome + '_' + rand)
    PASTA_TEMP.mkdir()   

    arquivo_de_audio = fmpeg_convert_to_ogg(
        arquivo_de_audio, 
        novo_nome + '_integral.ogg',
        PASTA_TEMP)

    rate = sf.info(arquivo_de_audio).samplerate
    channels = sf.info(arquivo_de_audio).channels
    endian = sf.info(arquivo_de_audio).endian
    rfc = joblib.load(arquivo_modelo)
    # -----------------------------------------------------------
    audio_file_obj_inst = AudioFile(arquivo_de_audio)
    audio_file_obj_inst._lazy_load()
    probas = audio_file_obj_inst.array_of_probas(
        modelo=rfc,
        scaler=pp.MaxAbsScaler()
    )

    seq_diff = probas[4] + probas[1] - probas[0]
    decisao_seq = algmain(seq_diff)
    print('temos a sequencia de 0 e 1s')
    segundos_cortados = decisao_seq.count_nonzero()
    print(segundos_cortados, 'segundos cortados')
    # -------------------------------------------------
    if log_output_labels:
        labels_from_0e1s(decisao_seq, os.path.join(
            pasta_log, novo_nome + '_labels.txt'))
    if log_png:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(seq_diff, 'y-', alpha=0.8)
        plt.plot(decisao_seq, 'r-', alpha=1)
        plt.savefig(os.path.join(pasta_log, novo_nome + '.png'))

    block_gen = sf.blocks(arquivo_de_audio, blocksize=rate)

    # arquivo final mp3
    src = str(PASTA_TEMP.joinpath('final_cortado.ogg'))
    dst = str(os.path.join(pasta, novo_nome + '.mp3'))

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

    mp3_convert = convertToMP3(src, PASTA_TEMP)

    shutil.move(mp3_convert, dst)

    if not keep_files:
        shutil.rmtree(PASTA_TEMP)

    return dst, segundos_cortados


def transformarEAtualizar():
    for pod in Podcast.select().where(Podcast.fase == 1):
        new_file = '{}_{}_{}-{}'.format(
            pod.data.year,
            pod.data.month,
            pod.data.day,
            'edacoisa'
        )

        pod_file, segundos_cortados = facaAmagica(
            os.path.join(pasta_download, pod.arquivo_baixado),
            new_file
        )
        pod.arquivo_podcast = os.path.basename(pod_file)
        pod.fase = 2
        pod.segundos_cortados = segundos_cortados
        pod.save()


def fmpeg_convert_to_ogg(intro_file, novo_nome, dir_):
    
    dst = str(dir_.joinpath(novo_nome))

    ll = [
        ffcmd,
        '-i',
        intro_file,
        '-c:a',
        'libvorbis',
        dst
    ]

    run(ll, stdout=PIPE)

    return dst


def convertToMP3(intro_file, dirname):

    outfile = str(dirname.joinpath('output.mp3'))

    ll = [
        ffcmd,
        '-i',
        intro_file,
        '-ar',
        '44100',
        '-codec:a',
        'libmp3lame',
        '-qscale:a',
        '8',
        '-filter:a',  # apenas pra ffmpeg acima da versao 3
        'loudnorm=I=-5',  # esse tb
        outfile
    ]

    run(ll, stdout=PIPE)

    return outfile


def teste_magic(arquivo_baixado):
    print(datetime.now())
    print('Arquivo baixado:', arquivo_baixado)
    dst = facaAmagica(arquivo_baixado, 't_' +
                      datetime.now().strftime(r'%y_%j_%H_%M_%S'), keep_files=True)
    print(datetime.now())
    print(dst)


if __name__ == '__main__':
    teste_magic(sys.argv[1])
