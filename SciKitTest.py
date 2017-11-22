
# coding: utf-8

# In[1]:


import numpy as np
# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
get_ipython().magic('matplotlib inline')
# and IPython.display for audio output
import IPython.display
# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
import datetime
import soundfile as sf
print(datetime.datetime.now())


# In[3]:


rate = sf.info('test.wav').samplerate
block_gen = sf.blocks('test.wav', blocksize=rate)
print(rate)


# In[4]:


tudo = []
for bl in block_gen:
    y = np.mean(bl, axis=1)
#     print(y, len(y))
    m1 = librosa.feature.melspectrogram(y)
#     print(len(m1), len(m1[2]))
    lis = []
    for el in m1:
        lis.append(el.mean()) 
    tudo.append(lis)
print(len(tudo))


# In[5]:


def round_better(num):
    try:
        num_ = float(num)
        num_ = round(num_)
    except:
        num_ = 0
    if num_ < 0:
        return 0
    return num_

def parseAudacityLabels(filepath):
    with open(filepath) as f:
        labels =  f.read().splitlines()
    infos = []
    for el in labels:
        info = el.split('\t')
    #     print(info)
        info[0] = round_better(info[0])
        info[1] = round_better(info[1])
        infos.append(info)
    return infos


# In[6]:


infos = parseAudacityLabels("LABES.txt")


# In[7]:


def makeArrayLabels(infos_audacity_parseadas, tamanho_maximo, vazio):
    labels = []
        
    for lis in infos_audacity_parseadas:
        print('len:', len(labels), 'prim:', lis[0], lis[2])            
        while len(labels) < lis[0]:
            labels.append(vazio)
        for i in range(lis[0], lis[1]):
            labels.append(lis[2])
        print('len:', len(labels))
        print('-----')
        
    
    print(set(labels))
    print('final', len(labels))
    return labels[:tamanho_maximo]

arr_labels = makeArrayLabels(infos, 5020, 'REI')
infos


# In[8]:


#TEST
for lis in infos:
    for i in range(lis[0], lis[1]):
        try:
            if not arr[i] == lis[2]:
                print(i, arr[i], lis[2])
        except:
            print('--', i)
            break


# In[9]:


arr_labels[204]


# In[10]:


import pandas as pd


# In[11]:


df = pd.DataFrame(tudo)


# In[12]:


df.head()


# In[13]:


from sklearn import preprocessing as pp
mm = pp.MinMaxScaler(feature_range=(-1, 1))


# In[14]:


arr = mm.fit_transform(tudo)


# In[15]:


arr


# In[16]:


df = pd.DataFrame(arr)
df


# In[17]:


plt.plot(df.loc[:].sum())


# In[18]:


plt.plot(df.sum(axis=1))
plt.axis([0, 6000, -128, -100])


# In[19]:


# treino = np.full(5020, 0)
# treino


# In[20]:


dt_res = pd.DataFrame(arr_labels)
dt_res


# In[21]:


# dt_res.loc[0:242] = 0
# dt_res.loc[242:353] = 1
# dt_res.loc[354:375] = 2
# dt_res.loc[378:466] = 1
# dt_res.loc[467:566] = 1
# dt_res.loc[567:662] = 4
# dt_res.loc[663:735] = 1
# dt_res.loc[736:802] = 2
# dt_res.loc[803:1736] = 1
# dt_res.loc[1737:1904] = 9
# dt_res.loc[1905:1912] = 6
# dt_res.loc[1913:2022] = 1
# dt_res.loc[2023:2038] = 6
# dt_res.loc[2039:2471] = 1
# dt_res.loc[2472:2630] = 9
# dt_res.loc[2631:2639] = 6
# dt_res.loc[2640:2748] = 1


# In[22]:


dt_res.sum()


# In[23]:


df['class'] = dt_res
df.head()


# In[24]:


somas = df.iloc[ :, :128].sum(axis=1)


# In[25]:


import scipy.stats as stats
plt.figure(figsize=(15,6))
h_ = sorted(somas)
pdf_ = stats.norm.pdf(h_, np.mean(h_), np.std(h_))
plt.plot(h_, pdf_)
plt.hist(h_, normed=True, bins=100)
plt.axis([-128, -119, 0, 0.3])


# In[26]:


# dt_res.loc[0:242] = 1
# dt_res.loc[242:353] = 1
# dt_res.loc[354:375] = 1
# dt_res.loc[378:466] = 1
# dt_res.loc[467:566] = 1
# dt_res.loc[567:662] = 1
# dt_res.loc[663:735] = 1
# dt_res.loc[736:802] = 1
# dt_res.loc[803:1736] = 1
# dt_res.loc[1737:1904] = 0
# dt_res.loc[1905:1912] = 0
# dt_res.loc[1913:2022] = 1
# dt_res.loc[2023:2038] = 0
# dt_res.loc[2039:2471] = 1
# dt_res.loc[2472:2630] = 0
# dt_res.loc[2631:2639] = 0
# dt_res.loc[2640:2748] = 1


# In[27]:


# from sklearn.tree import DecisionTreeClassifier


# In[28]:


# df.iloc[0:2747, 0:128].info()


# In[29]:


df.describe()


# In[30]:


# DTC = DecisionTreeClassifier()
# DTC.fit(df.iloc[0:2747, 0:128], df.loc[0:2746]['class'])


# In[31]:


# df.iloc[2747:, :128]


# In[32]:


# sera = DTC.predict(df.iloc[2747:, :128])
# plt.plot(sera)


# In[76]:


def smoothSeq(seq, step):
    l = len(seq)
    new_seq = []    
    for a in range(0, l, step):
        sub = seq[a:a+step]
        med = sum(sub)/step              
        for el in range(len(sub)):
            new_seq.append(med)
    return np.array(new_seq)        


# In[34]:


df.iloc[0:5020, 0:128]


# In[35]:


df.loc[0:5020]['class']


# In[36]:


from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC(n_estimators=500)
rfc.fit(df.iloc[0:5020, 0:128], df.loc[0:5020]['class'])


# In[37]:


rate = sf.info('confirma.wav').samplerate
block_gen = sf.blocks('confirma.wav', blocksize=rate)
print(rate)


# In[38]:


tudo_confirma = []
for bl in block_gen:
    y = np.mean(bl, axis=1)
#     print(y, len(y))
    m1 = librosa.feature.melspectrogram(y)
#     print(len(m1), len(m1[2]))
    lis = []
    for el in m1:
        lis.append(el.mean()) 
    tudo_confirma.append(lis)
print(len(tudo_confirma))


# In[39]:


arr_confirma = mm.fit_transform(tudo_confirma)


# In[40]:


sera_rdf = rfc.predict(arr_confirma)


# In[41]:


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(15,6))
plt.plot(sera_rdf[0])


# In[42]:


probs = rfc.predict_proba(arr_confirma)


# In[53]:


plt.figure(figsize=(20, 4))
plt.plot(probs)
print(rfc.classes_)
print(len(probs))


# In[60]:


dfprobs = pd.DataFrame(probs, columns=rfc.classes_)


# In[62]:


dfprobs.head()


# In[70]:


plt.figure(figsize=(20, 4))
plt.plot(dfprobs['COMERCIAL'], 'm')
plt.axis([0, 5050, 0, 1])


# In[71]:


plt.figure(figsize=(20, 4))
plt.plot(dfprobs['REI'])
plt.axis([0, 5050, 0, 1])


# In[72]:


plt.figure(figsize=(20, 4))
plt.plot(dfprobs['SILENCIO'])
plt.axis([0, 5050, 0, 1])


# In[74]:


plt.figure(figsize=(20, 4))
plt.plot(dfprobs['MUSICA'])
plt.axis([0, 5050, 0, 1])


# In[75]:


plt.figure(figsize=(20, 4))
plt.plot(dfprobs['VINHETA'])
plt.axis([0, 5050, 0, 1])


# In[79]:


plt.figure(figsize=(20, 4))
plt.plot(smoothSeq(dfprobs['COMERCIAL'], 25), 'm')
plt.axis([0, 5050, 0, 1])


# In[83]:


plt.figure(figsize=(20, 4))
plt.plot(smoothSeq(dfprobs['REI'], 30), 'r')
plt.axis([0, 5050, 0, 1])


# In[87]:


#atrasa...
def smoothListGaussian(list,strippedXs=False,degree=5):
     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(len(list)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  
     return smoothed


# In[110]:


plt.figure(figsize=(20, 4))
# plt.plot(dfprobs['REI'], 'y:')
# plt.plot(smoothListGaussian(dfprobs['REI'], degree=30), 'r')
plt.plot(smoothSeq(dfprobs['REI'], 60), 'k')
plt.plot(smoothSeq(dfprobs['COMERCIAL'], 25), 'm:')
plt.grid(color='y', linestyle=':', linewidth=1)


# In[128]:


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


# In[121]:


plt.plot(decisaoSimples(dfprobs['REI'],dfprobs['COMERCIAL'], dfprobs['SILENCIO']))


# In[133]:


decisao_seq = decisaoSimples(
    smoothSeq(dfprobs['REI'], 40),
    smoothSeq(dfprobs['COMERCIAL'], 40),
    smoothSeq(dfprobs['SILENCIO'], 40)
)
plt.figure(figsize=(20, 5))
# plt.plot(dfprobs['REI'], 'y:')
# plt.plot(smoothListGaussian(dfprobs['REI'], degree=30), 'r')
# plt.grid(color='y', linestyle='-', linewidth=1)
plt.plot(smoothSeq(dfprobs['REI'], 60), 'r')
plt.plot(smoothSeq(dfprobs['COMERCIAL'], 25), 'm:')
plt.plot(decisao_seq, 'k--')


# In[134]:


from sklearn.externals import joblib
joblib.dump(rfc, 'modelo.pkl')
#clf = joblib.load('filename.pkl')

