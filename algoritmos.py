import numpy as np


def seqCoiso(seq, raio=16, corte=0):
    #     setcinco = np.percentile(dfprobs['COMERCIAL'], 75)
    arr = np.zeros(len(seq))
    points = []
    for i in range(raio + 1, len(seq), 2 * raio):
        try:
            subarray = seq[i - raio:i + raio]
        except:
            pass
        med = np.mean(subarray)
        if med > corte:
            arr[i - raio:i + raio] = 1
            points.append(i)
    unique, counts = np.unique(arr, return_counts=True)
    prop_d = dict(zip(unique, counts))
    return arr, points, prop_d


def getBiggerSlicex(seq, p, raio=16, disloc=10, step=1):
    lis = []
    for p_ in range(-disloc, disloc, step):
        try:
            sub = seq[p - raio + p_:p + raio + p_]
            # le = len(sub)
            mean = float(np.mean(sub))
            if np.isnan(mean):
                continue
        except:
            continue
        lis.append((mean, p_))
    esse = sorted(lis, reverse=True)
    return esse[0][1]


def cortaLegal(seq, raio=15, corte=0):
    _, points, _ = seqCoiso(seq, raio=raio, corte=corte)
    neww = np.zeros(len(seq))
    new_points = []
    prev = None
    acc = 0
    for el in points:
        if prev is None:
            prev = el
            continue
        if el - prev == 2 * raio:
            acc = acc + 1
            # print('', el, end='\t')
        else:
            # print()
            diff = (acc + 1) * raio
            new_raio = diff
            centro = prev - diff + raio

            # print('{:>6}  |  acc: {:>2}  |  centro: {:>6} | new_raio: {:>5}  |  diff: {:>5}  |  prev: {:>6}'
            #  .format(el, acc, centro, new_raio, diff, prev))
            # print('-'*40)
            acc = 0
            new_points.append((centro, new_raio))
        prev = el
    diff = (acc + 1) * raio
    new_raio = diff
    centro = prev - diff + raio
    # print('{:>6}  |  acc: {:>2}  |  centro: {:>6} | new_raio: {:>5}  |  diff: {:>5}  |  prev: {:>6}'
    #      .format(el, acc, centro, new_raio, diff, prev))
    # print('-'*40)
    acc = 0
    new_points.append((centro, new_raio))

    # print(new_points)
    for el in new_points:
        neww[el[0] - el[1]:el[0] + el[1]] = 1
    return neww, new_points


def ajustaMelhorMedia(seqq, new_points):
    mais_uma = np.zeros(len(seqq))
    points = []
    for point in new_points:
        dis = getBiggerSlicex(seqq, point[0], raio=point[1], disloc=15, step=1)
        print(point, dis)
        points.append((point[0] + dis, point[1]))
        mais_uma[point[0] - point[1] + dis:point[0] + point[1] + dis] = 1
    return mais_uma, points


def main(seqq):
    _, pontos = cortaLegal(seqq)
    _, pontos = ajustaMelhorMedia(seqq, pontos)
    sewq, _ = ajustaMediaPeloTamanho(seqq, pontos)
    return sewq


def labels_from_0e1s(seq, arquivo):
    # ini = None
    pos = 0
    last = None
    lis = []
    lab = None
    while True:
        if lab is None:
            lab = seq[pos]
            pos = pos + 1
#             print('lab')
            continue
        try:
            if seq[pos] == lab:
                #             print('...cont', pos, seq[pos])
                pos = pos + 1
                continue
            if last is None:
                last = 0
            print('mudou', lab, pos, seq[pos])
            lis.append((last, pos - 1, lab))
            last = pos
            lab = seq[pos]
            pos = pos + 1
        except:
            #             print('EXCEPT')
            break
    with open(arquivo, 'w') as f:
        for t in lis:
            if t[2] == 0:
                continue
            lin = '{}\t{}\t{}\r\n'.format(t[0], t[1], 'corta')
            f.write(lin)


def ajustaMediaPeloTamanho(sewq, points_da_melhor_media_ajustada, ponto_de_corte_peq_grande=10):
    neww_seq = np.zeros(len(sewq))
#     if len(points_da_melhor_media_ajustada) > 16:
#         import operator
#         print('lista mt grande')
#         points_da_melhor_media_ajustada = sorted(points_da_melhor_media_ajustada, key=operator.itemgetter(1), reverse=True)[:16]
#         pprint(points_da_melhor_media_ajustada)
    points = []
    for idx, point in enumerate(points_da_melhor_media_ajustada):

        i = point[0] - point[1]
        f = point[0] + point[1]
        diff_ref = f - i
        print('diff_ref', diff_ref)
        if idx == 0:
            points.append((i, f))
            continue
        # tam_orig = 2 * point[1] - 1
        med_orig = np.mean(sewq[i:f])
        meds = [(med_orig, (i, f))]
        subraio = int(round(point[1] / 2))
        if subraio > 10:
            subraio = 10
        poss = [a for a in range(-subraio, subraio)]
        inis = [i + a for a in poss]
        fins = [f + a for a in poss]
        for ii in inis:
            for ff in fins:
                diff = ff - ii
                if diff < ponto_de_corte_peq_grande:
                    if diff > diff_ref:
                        continue
                else:
                    if diff < diff_ref:
                        continue
                meds.append(
                    (np.mean(sewq[ii:ff]), (ii, ff))
                )

        if len(meds) == 0:
            continue

        sort_meds = sorted(meds, reverse=True)

#         pprint(sort_meds)
        ponto = sort_meds[0][1]
        points.append(ponto)
    for ponto in points:
        print(ponto)
        neww_seq[ponto[0]:ponto[1]] = 1
    return neww_seq, points
