#!/usr/bin/env python
# coding: utf-8

# In[188]:
from sage.all import *

import numpy as np

import pandas,sys

import statsmodels.api as sm

from statsmodels.tsa.api import VAR

def parseMidi(fp,part=0):
    import os
    from music21 import converter
    print(fp)
    score = converter.parse(fp,quantizePost=True)
    print(list(score.elements[0].notesAndRests))
    #print([e.partAbbreviation for e in score.elements][0])
    from music21 import chord
    durs = []
    ll0 = []
    vols = []
    isPauses = []
    for p in score.elements[part].notesAndRests:
        #print(p)
        if type(p)==chord.Chord:
            pitches = median([e.pitch.midi-21 for e in p]) # todo: think about chords
            vol = median([e.volume.velocity for e in p])
            dur = float(p.duration.quarterLength)
            #print(pitches)
            ll0.append(pitches)
            isPause = False
        elif (p.name=="rest"):
            pitches = 89
            vol = 1
            dur = float(p.duration.quarterLength)
            ll0.append(pitches)
            isPause = True
        else:
            pitches = p.pitch.midi-21
            vol = p.volume.velocity
            dur = float(p.duration.quarterLength)
            ll0.append(pitches)
            isPause =  False
        durs.append(dur/(12*4.0))
        vols.append(vol*1.0/127.0)
        isPauses.append(isPause)
            #print(p.name,p.octave,p.duration.quarterLength)
    #print(dir(score)) 
    #print(ll0)
    #print(durs)
    return ll0,durs,vols,isPauses


def dist(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    return sqrt(2*(1-gcd(a,b)^2/(a*b)))

def kernPause(a1,a2):
    return  1*(a1==a2)

def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)

def kernDuration(k1,k2):
    return  log(k1)*log(k2)

def kernVolume(v1,v2):
    return log(v1)*log(v2)

def kern(t1,t2):
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    return kernPause(isPause1,isPause2)+kernPitch(pitch1,pitch2)+kernDuration(duration1,duration2)+kernVolume(volume1,volume2)

def getRational(k):
    alpha = 2**(1/12.0)
    x = RDF(alpha**k).n(50)
    return x.nearby_rational(max_error=0.01*x)


def ngrams(input, n):
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

def kernNgram(ngrams1,ngrams2):
    return sum([ kern(ngrams1[i], ngrams2[i]) for i in range(len(ngrams1))]) 



#a#lpha = (2**(1/12.0))
#qq=[RDF(alpha**k).n(50).nearby_rational(max_error=0.01*RDF(alpha**k).n(50)) for k in range(12)]


fn = "./input/beethoven.mid" if len(sys.argv)!=2 else sys.argv[1]

def forecastPart(fn=fn,p=0,Nforecast=20,Ndim=3,Nseq=8,Nlen=50,maxlags=6):
    pitches,durations,volumes,isPauses = parseMidi(fn,part=p)
    zz = zip(pitches,durations,volumes,isPauses)
    print(len(zz))
    M = matrix([[1]])
    c = Nseq+1
    while M.is_positive_definite() and Nlen>c:
        #Nseq += 1
        c+=1
        Z = ngrams([t for t in zz[0:c]],Nseq)
        print(len(Z))
        M = matrix([[ kernNgram(t1,t2) for t1 in Z] for t2 in Z],ring=RDF)
    c-=1
    Z = ngrams([t for t in zz[0:c]],Nseq)
    M = matrix([[kernNgram(t1,t2) for t1 in Z] for t2 in Z],ring=RDF)

#print(qq)
#M = matrix([[kern(k1,k2) for k1 in range(88)] for k2 in range(88)])
#print(M.str())
#print(M.is_positive_definite())


    CC = M.cholesky()
#Ch = M.cholesky().rows()

    from sklearn.decomposition import PCA

    nDim =Ndim
    pca = PCA(n_components=nDim)
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    stdScaler = StandardScaler()
    Ch = pca.fit_transform(stdScaler.fit_transform(CC))

    model = VAR(Ch)

    results = model.fit(maxlags=maxlags,trend="nc",ic="aic")
    print(results.summary())

    lag_order = results.k_ar

    preds = []
    X = [x for x in Ch]
    print(X)
    for k in range(Nforecast):
    #print(np.array(X[-lag_order:]))
        pred = results.forecast(np.array(X[k:(k+lag_order)]), 1)
        preds.append(pred[0])
        X.append(pred[0])
#v = -mean(Ch)
#v = v/np.sqrt(np.dot(v,v))
#Ch.insert(0,v)


# In[194]:


    from sklearn.neighbors import NearestNeighbors
#import numpy as np
#from scipy.linalg import inv,pinv2,sqrtm,expm,logm, block_diag

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Ch)

    def findBestMatch(nbrs,new_row):
        distances,indices = nbrs.kneighbors([np.array(new_row)])
        return indices[0][0]


    inds = []
    for x in preds:
        i = findBestMatch(nbrs,x)
        inds.extend(Z[i][0:3])

    return(inds)

durationslist = [[sum([(2**(n-i)) for i in range(d+1)]) for n in range(-8,3+1)] for d in range(1,2)]
notevalues = []
for i in range(len(durationslist)):
    notevalues.extend(durationslist[i])
    
notevalues = sorted(notevalues)    
print(notevalues)
print(len(notevalues))

def findNearestDuration(duration):
    return sorted([(abs(duration-nv),nv) for nv in notevalues])[0][1]

def writePitches(fn,inds,tempo=82):
    from midiutil import MIDIFile

    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats
    tempo    = tempo # In BPM
    volume   = 116 # 0-127, as per the MIDI standard

    ni = len(inds)
    MyMIDI = MIDIFile(ni,adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
    MyMIDI.addTempo(track,time, tempo)


    for k in range(ni):
        MyMIDI.addProgramChange(k,k,0,0)


    times = ni*[0]
    for k in xrange(len(inds)):
        channel = k
        track = k
        for i in range(len(inds[k])):
            pitch,duration,volume,isPause = inds[k][i]
            track = k
            channel = k
            duration = findNearestDuration(duration*12*4)
            #print(k,pitch,times[k],duration,100)
            if not isPause: #rest
                #print(volumes[i])
                MyMIDI.addNote(0, channel, pitch+21, times[k] , duration, int(127*volume))#*(ni-k+10.0)/(ni+10.0))
            times[k] += duration*1.0    
       
    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    print("written")    



#pps = [0,1,2,3,4,5]
pps = [0,1]
iinds = []
for p in pps:
    inds = forecastPart(fn,p=p,Nlen=30,Nforecast=60,Ndim=2,Nseq=8,maxlags=3)
    iinds.append(inds)
writePitches(fn+".mix.mid",iinds,tempo=72)    




