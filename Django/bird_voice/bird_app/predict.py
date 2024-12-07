import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from librosa import core, onset, feature, display
import soundfile as sf
import umap
from IPython.display import Audio
import sklearn
import os

def load_audio():
    print(os.getcwd() +'\\media\\result.flac')
    data, samplerate = sf.read(os.getcwd() +'\\media\\result.flac')
    s = len(data)/samplerate
    sg = feature.melspectrogram(y=data, sr=samplerate, hop_length=512)
    
    # Take mean amplitude M from frame with highest energy
    centerpoint = np.argmax(sg.mean(axis=0))
    M = sg[:,centerpoint].mean()
    
    # Filter out all frames with energy less than 5% of M
    mask = sg.mean(axis=0)>=M/20

    audio_mask = np.zeros(len(data), dtype=bool)
    for i in range(0,len(mask)):
        audio_mask[i*512:] = mask[i]
    return sg, mask, data, audio_mask, samplerate

def preprocess():
    df = pd.read_csv(os.getcwd() +'\\predict.csv')
    df['length'] = np.zeros(len(df))
    waves = {}
    for file_id in df['file_id']:
        sg, mask, data, audio_mask, sample_rate = load_audio()
        waves[file_id] = data[audio_mask]
        df.loc[df['file_id'] == file_id,'length'] = len(data[audio_mask])
    df['windows'] = df['length'].apply(lambda x: int(x/6.144000e+03))
    n_windows = df.groupby('species')['windows'].sum().min()
    windows = {}
    for file_id in df['file_id']:
        wave = waves[file_id]
        species = df[df['file_id']==file_id]['genus'].values[0] + "_" + df[df['file_id']==file_id]['species'].values[0]
        if species not in windows:
            windows[species] = []
        for i in range(0, int(len(wave)/6.144000e+03)):
            windows[species].append(wave[i:int(i+6.144000e+03)])
    windows_fixed = {}
    for species in windows.keys():
        windows_fixed[species] = []
        ws = windows[species]
        index = np.random.choice(len(ws), n_windows, replace=False)
        for i in range(0, len(ws)):
            if i in index:
                windows_fixed[species].append(ws[i])
    new_dataset = pd.DataFrame()

    for species in windows_fixed.keys():
        for i in range(0,n_windows):
            data_point = {'species':species.split('_')[1], 'genus':species.split('_')[0]}
            spec_centroid = feature.spectral_centroid(y=windows_fixed[species][i])[0]
            chroma = feature.chroma_stft(y=windows_fixed[species][i], sr=sample_rate)
            for j in range(0,13):
                data_point['spec_centr_'+str(j)] = spec_centroid[j]
                for k in range(0,12):
                    data_point['chromogram_'+str(k)+"_"+str(j)] = chroma[k,j]
            data_point = pd.DataFrame([data_point])
            new_dataset = pd.concat([new_dataset,data_point],ignore_index=True)
    features= list(new_dataset.columns)
    features.remove('species')
    features.remove('genus')

    X = new_dataset[features].values
    return X

def process(kind):
    print(kind)
    X=preprocess()
    if kind == 'svm':
        loaded_model = pickle.load(open(os.getcwd() +'\\svm.sav', 'rb'))
        print(loaded_model)
        print(kind)
    elif kind == 'nb':
        loaded_model = pickle.load(open(os.getcwd() +'\\naive.sav', 'rb'))
        print(kind)
    ans=loaded_model.predict(X)
    dic={}
    s=set(ans)
    for i in s:
        dic[i]=sum([1 for j in list(ans) if i==j])
    
    sol=None
    res=None
    for i in dic.items():
        if sol is None or sol<i[1]:
            sol=i[1]
            res=i[0]
    return res
