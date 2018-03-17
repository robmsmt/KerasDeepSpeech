## this file is created to record some sample sentences in your own voice
## it should be run on a desktop/laptop using a microphone to speak

#using silence is too complicated. Just use ctrl+c as works on all systems and can cope with background noise

import pyaudio
import wave
import pandas as pd
import os


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


#http://www.voxforge.org/home/submitspeech/windows/step-1/phoneme/e01
TRANSCRIPT_SOURCE = "./own/enron-src.csv"
OUTPUT_DIR = "./data/own/"


def clean(word):
    # token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def startloop(rec_number):
    ##read in data from csv
    df = pd.read_csv(TRANSCRIPT_SOURCE, sep=',', header=None)

    #HEADERS
    wav_filename = []
    wav_filesize = []
    transcript = []



    print("when ready press enter to start recording and then enter again to finish")
    # time.sleep(1)
    count = 0

    while count < int(rec_number):

        try:
            datalist = df.iloc[[count]].values.tolist()
        except IndexError:
            print("out of data")
            break


        trans = datalist[0][1]
        trans = clean(trans)
        trans = trans.decode('utf-8').encode('ascii', errors='ignore')
        filename = datalist[0][0]

        print(trans)
        ##get sample sentence


        inputvar = str(raw_input('ready? press enter to begin recording and ctrl+c to stop'))

        if inputvar == "":
            r = record(filename, OUTPUT_DIR, trans)

            # inputcheck = str(raw_input('press enter if you are happy, or r to redo.'))
            wav_filename.append(r)
            wav_filesize.append(os.path.getsize(r))
            transcript.append(trans)

        else:
            print 'Try another sentence'



        count = count+1


    a = {'wav_filename': wav_filename,
         'wav_filesize': wav_filesize,
         'transcript': transcript
         }

    df_train = pd.DataFrame(a, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_train.to_csv("./own/enron_train.csv", sep=',', header=True, index=False, encoding='ascii')


def record(name, dir, trans):


    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    #print("Recording to:", dir, name)
    print ("\n" * 30) #works on all OS
    print("RECORDING, press ctrl+c to stop recording")
    print(trans)
    frames = []
    #for i in range(0, int(RATE / CHUNK * RECORD_MINUTES * 60)):

    while 1:
        try:
            data = stream.read(CHUNK)
            frames.append(data)

        except KeyboardInterrupt:
            print("STOPPING")
            break


    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording to:", dir, name)

    fileindir = dir+name+".wav"

    wf = wave.open(fileindir, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return fileindir


rec_number = str(input('How many samples do you want to record? '))
if rec_number.isdigit():
    print 'Okay - ready for ' + rec_number + " samples"
    startloop(rec_number)
else:
    print 'error, not a number'

