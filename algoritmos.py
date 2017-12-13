import numpy as np

def seqCoiso(seq, raio=16, corte=0):
#     setcinco = np.percentile(dfprobs['COMERCIAL'], 75)
    arr = np.zeros(len(seq))
    points = []
    for i in range(raio+1, len(seq), 2*raio):
        try:
            subarray = seq[i-raio:i+raio]
        except:
            pass
        med = np.mean(subarray)
        if med > corte:
            arr[i-raio:i+raio] = 1
            points.append(i)
    unique, counts = np.unique(arr, return_counts=True)
    prop_d = dict(zip(unique, counts))   
    return arr, points, prop_d

def getBiggerSlicex(seq, p, raio=16, disloc=10, step=1):
    lis = []
    for p_ in range(-disloc, disloc, step):  
        try:
            sub = seq[p-raio+p_:p+raio+p_]
            le = len(sub)            
            mean = float(np.mean(sub))
            if np.isnan(mean):
                continue
        except:
            continue
        lis.append((mean, p_))
    esse = sorted(lis, reverse=True)
    return esse[0][1]

def cortaLegal(seq, raio=16, corte=0):
    _, points, _ = seqCoiso(seq, raio=raio, corte=corte)
    neww = np.zeros(len(seq))
    new_points = []
    prev = None
    acc = 0
    for el in points:
        if prev is None:
            prev = el
            continue
        if el - prev == 2*raio:
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
        neww[el[0]-el[1]:el[0]+el[1]] = 1
    return neww, new_points

def ajustaMelhorMedia(seqq, new_points):
    mais_uma = np.zeros(len(seqq))
    for point in new_points:
        dis = getBiggerSlicex(seqq, point[0], raio=point[1], disloc=15, step=1)
        print(point, dis)
        mais_uma[point[0]-point[1]+dis:point[0]+point[1]+dis] = 1
    return mais_uma

def main(seqq):
    _, pontos = cortaLegal(seqq)
    return ajustaMelhorMedia(seqq, pontos)

def labels_from_0e1s(seq, arquivo):
    ini = None
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
            lis.append((last, pos-1, lab))
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