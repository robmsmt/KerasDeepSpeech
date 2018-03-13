
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.utils import shuffle

import python_speech_features as p
import scipy.io.wavfile as wav
import soundfile
import scipy
from aubio import source, pvoc, mfcc
from numpy import vstack, zeros, diff




from keras.preprocessing.sequence import pad_sequences

from utils import text_to_int_sequence

# Data batch generator, responsible for providing the data to fit_generator

class BatchGenerator(object):
    def __init__(self, dataframe, dataproperties, training, batch_size=16, model_input_type="mfcc"):
        self.training_data = training
        self.model_input_type = model_input_type ##mfcc, mfcc-aubio, spectrogram, spectrogram-img
        # self.aubio = False
        self.df = dataframe.copy()
        #['wav_filesize','transcript','wav_filename']
        self.wavpath = self.df['wav_filename'].tolist()
        self.transcript = self.df['transcript'].tolist()
        self.finish = self.df['wav_filesize'].tolist()
        self.start = np.zeros(len(self.finish))
        self.length = self.finish
        self.shuffling = True

        self.batch_size = batch_size
        self.cur_index = 0

        self.feats_std = 0
        self.feats_mean = 0

        self.set_of_all_int_outputs_used = None

        #Free up memory of unneeded data
        del dataframe
        del dataproperties
        self.df = None
        del self.df

    def normalise(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def get_batch(self, idx):

        batch_x = self.wavpath[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_trans = self.transcript[idx * self.batch_size:(idx + 1) * self.batch_size]

        try:
            assert (len(batch_x) == self.batch_size)
            assert (len(batch_y_trans) == self.batch_size)
        except Exception as e:
            print(e)
            print(batch_x)
            print(batch_y_trans)

        # 1. X_data (the MFCC's for the batch)
        if(self.model_input_type == "spectrogram"):
            # 0. get the maximum time length of the batch
            x_val = [get_max_specto_time(file_name) for file_name in batch_x]
            max_val = max(x_val)
            # print("Max batch time value is:", max_val)
            X_data = np.array([make_specto_shape(file_name, padlen=max_val) for file_name in batch_x])
            assert (X_data.shape == (self.batch_size, max_val, 161))

        elif(self.model_input_type == "mfcc-aubio"):
            x_val = [get_max_aubio(file_name) for file_name in batch_x]
            max_val = max(x_val)
            # print("Max batch time value is:", max_val)
            X_data = np.array([make_aubio_shape(file_name, padlen=max_val) for file_name in batch_x])
            # print(X_data.shape)
            assert (X_data.shape == (self.batch_size, max_val, 26))

        elif(self.model_input_type == "mfcc"):
            # 0. get the maximum time length of the batch
            x_val = [get_max_time(file_name) for file_name in batch_x]
            max_val = max(x_val)
            # print("Max batch time value is:", max_val)

            X_data = np.array([make_mfcc_shape(file_name, padlen=max_val) for file_name in batch_x])
            assert (X_data.shape == (self.batch_size, max_val, 26))
        # print("1. X_data shape:", X_data.shape)
        # print("1. X_data:", X_data)


        # 2. labels (made numerical)
        # get max label length
        y_val = [get_maxseq_len(l) for l in batch_y_trans]
        # print(y_val)
        max_y = max(y_val)
        # print(max_y)
        labels = np.array([get_intseq(l, max_intseq_length=max_y) for l in batch_y_trans])
        # print("2. labels shape:", labels.shape)
        # print("2. labels values=", labels)
        assert(labels.shape == (self.batch_size, max_y))

        # 3. input_length (required for CTC loss)
        # this is the time dimension of CTC (batch x time x mfcc)
        #input_length = np.array([get_xsize(mfcc) for mfcc in X_data])
        input_length = np.array(x_val)
        # print("3. input_length shape:", input_length.shape)
        # print("3. input_length =", input_length)
        assert(input_length.shape == (self.batch_size,))

        # 4. label_length (required for CTC loss)
        # this is the length of the number of label of a sequence
        #label_length = np.array([len(l) for l in labels])
        label_length = np.array(y_val)
        # print("4. label_length shape:", label_length.shape)
        # print("4. label_length =", label_length)
        assert(label_length.shape == (self.batch_size,))

        # 5. source_str (used for human readable output on callback)
        source_str = np.array([l for l in batch_y_trans])



        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str
        }

        outputs = {'ctc': np.zeros([self.batch_size])}

        return (inputs, outputs)

    def next_batch(self):
        while 1:
            assert (self.batch_size <= len(self.wavpath))

            if (self.cur_index + 1) * self.batch_size >= len(self.wavpath) - self.batch_size:

                self.cur_index = 0

                if(self.shuffling==True):
                    print("SHUFFLING as reached end of data")
                    self.genshuffle()

            try:
                ret = self.get_batch(self.cur_index)
            except:
                print("data error - this shouldn't happen - try next batch")
                self.cur_index += 1
                ret = self.get_batch(self.cur_index)

            self.cur_index += 1

            yield ret

    def genshuffle(self):
        self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                             self.transcript,
                                                             self.finish)


    def export_test_mfcc(self):
        # this is used to export data e.g. into iOS

        testset = next(self.next_batch())[0]
        mfcc = testset['the_input'][0:self.batch_size]  ## export all mfcc's in batch #26 x 29 ?
        words = testset['source_str'][0:self.batch_size]
        labels = testset['the_labels'][0:self.batch_size]

        print("exporting:", type(mfcc))
        print(mfcc.shape)
        print(words.shape)
        print(labels.shape)

        # we save each mfcc/words/label as it's own csv file
        for i in range(0, mfcc.shape[0]):
            np.savetxt('./Archive/test_spectro/test_spectro_{}.csv'.format(i), mfcc[i, :, :], delimiter=',')

        print(words)
        print(labels)

        return



def get_normalise(self, k_samples=100):
    # todo use normalise from DS2 - https://github.com/baidu-research/ba-dls-deepspeech
    """ Estimate the mean and std of the features from the training set
    Params:
        k_samples (int): Use this number of samples for estimation
    """
    # k_samples = min(k_samples, len(self.train_audio_paths))
    # samples = self.rng.sample(self.train_audio_paths, k_samples)
    # feats = [self.featurize(s) for s in samples]
    # feats = np.vstack(feats)
    # self.feats_mean = np.mean(feats, axis=0)
    # self.feats_std = np.std(feats, axis=0)
    pass

def get_maxseq_len(trans):
    # PAD
    t = text_to_int_sequence(trans)
    return len(t)

def get_intseq(trans, max_intseq_length=80):
    # PAD
    t = text_to_int_sequence(trans)
    while (len(t) < max_intseq_length):
        t.append(27)  # replace with a space char to pad
    # print(t)
    return t

def get_max_time(filename):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    # print(r.shape)
    return r.shape[0]  #

def get_max_specto_time(filename):
    r = spectrogram_from_file(filename)
    # print(r.shape)
    return r.shape[0]  #


def get_max_aubio(filename):
    r = aubio(filename)  # 2D array -> timesamples x mfcc_features
    # print(r.shape)

    return r.shape[0]  #

def make_specto_shape(filename, padlen=778):
    r = spectrogram_from_file(filename)
    t = np.transpose(r)  # 2D array ->  spec x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T

    return X  # MAXtimesamples x specto {max x 161}

def make_aubio_shape(filename, padlen=778):
    r = aubio(filename)
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}

def make_mfcc_shape(filename, padlen=778):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}

def get_xsize(val):
    return val.shape[0]

def shuffle_data(self):
    self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                         self.transcript,
                                                         self.finish)
    return


def aubio(source_filename):

    # print("Usage: %s <source_filename> [samplerate] [win_s] [hop_s] [mode]" % sys.argv[0])
    # print("  where [mode] can be 'delta' or 'ddelta' for first and second derivatives")7

    n_filters = 40  # must be 40 for mfcc
    n_coeffs = 26
    win_s = 512
    hop_s = win_s // 4
    # mode = "default"
    samplerate = 0

    s = source(source_filename, samplerate, hop_s)
    samplerate = s.samplerate
    p = pvoc(win_s, hop_s)
    m = mfcc(win_s, n_filters, n_coeffs, samplerate)

    mfccs = zeros([n_coeffs, ])
    frames_read = 0
    while True:
        samples, read = s()
        spec = p(samples)
        mfcc_out = m(spec)
        mfccs = vstack((mfccs, mfcc_out))
        frames_read += read
        if read < hop_s: break

    return mfccs

##Require for DS2 - source: https://github.com/baidu-research/ba-dls-deepspeech
##############################################################################

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))

def featurise(audio_clip):
    """ For a given audio clip, calculate the log of its Fourier Transform
    Params:
        audio_clip(str): Path to the audio clip
    """

    step = 10
    window = 20
    max_freq = 8000

    return spectrogram_from_file(
        audio_clip, step=step, window=window,
        max_freq=max_freq)

