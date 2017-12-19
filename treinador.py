import os
import sqlite3

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def treina(db_valores, db_labels, caminho_modelo):
    conn = sqlite3.connect(db_valores)
    c = conn.cursor()
    c.execute('select * from mel')
    treino = np.array(c.fetchall(), dtype=np.float32)
    conn.close()
    conn = sqlite3.connect(db_labels)
    c = conn.cursor()
    c.execute('select * from mel')
    labels = np.array(c.fetchall(), dtype=np.int8)
    conn.close()
    print('Shape valores:', treino.shape)
    print('Shape labels :', labels.shape)
    print('Treinando modelo...')
    modelo = RandomForestClassifier(n_estimators=9)
    modelo.fit(treino, labels.ravel())
    del treino
    del labels
    joblib.dump(modelo, caminho_modelo, compress=True)
    print('Salvo em', caminho_modelo)   

if __name__ == '__main__':
    treina(
        os.path.join('dados', 'maxminscaled_mel_values.sqlite3'),
        os.path.join('dados', 'maxminscaled_mel_labels_int.sqlite3'),
        'novo_modelo_treinado.pkl'
        )
    print(r'Atualizar o .env pra constar o novo modelo!')
