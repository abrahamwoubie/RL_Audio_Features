import scipy
import scipy.io
from scipy import signal
import scipy.io.wavfile
import numpy as np
import aubio
from scipy import signal # audio processing
from scipy.fftpack import dct
from pydub import AudioSegment
import librosa # library for audio processing
from GlobalVariables import  GlobalVariables

from aubio import pvoc, source, float_type
from numpy import zeros, log10, vstack

grid_size=GlobalVariables
class Extract_Features:


    def Extract_Samples(row, col):
        #print("Extract Sample of Row {} Col {} nRow {} nCol {}".format(row,col,nRow,nCol))

        fs = 100  # sample rate
        f = 2  # the frequency of the signal

        x = np.arange(fs)  # the points on the x axis for plotting
        samples = [np.sin(2 * np.pi * f * (i / fs)) for i in x]
        # compute the value (amplitude) of the sin wave at the for each sample
        # if letter in b'G':
        #if(row==grid_size.nRow-1 and col==grid_size.nCol-1):
        if (row == 3 and col == 3):
            samples = [100 + row + col + np.sin(2 * np.pi * f * (i / fs)) for i in x]
        else:
            samples = [np.sin(2 * np.pi * f * (i / fs)) for i in x]
        return samples

    def Extract_Spectrogram(row,col):
        filename = 'Test.wav'
        samplerate = 0
        win_s = 512  # fft window size
        hop_s = win_s // 2  # hop size
        fft_s = win_s // 2 + 1  # spectrum bins

        a = source(filename, samplerate, hop_s)  # source file
        if samplerate == 0: samplerate = a.samplerate
        pv = pvoc(win_s, hop_s)  # phase vocoder
        specgram = zeros([0, fft_s], dtype=float_type)  # numpy array to store spectrogram

        # analysis
        while True:
            samples, read = a()  # read file
            specgram = vstack((specgram, pv(samples).norm))  # store new norm vector
            if read < a.hop_size: break
        if (row == grid_size.nRow - 1 and grid_size == grid_size.nCol - 1):
            specgram=specgram+100
        #else:
            #specgram=specgram+row+col
        return specgram

    def Extract_Spectrogram11(row,col):
        sample_rate, data = scipy.io.wavfile.read('Test.wav')
        # Spectrogram of .wav file
        sample_freq, segment_time, spec_data = signal.spectrogram(data, sample_rate)
        if (row == grid_size.nRow - 1 and grid_size == grid_size.nCol - 1):
            spec_data=spec_data*100
        else:
            spec_data = spec_data + row + col
        return spec_data

    def Extract_MFCC(row,col):
        import sys
        from aubio import source, pvoc, mfcc
        from numpy import vstack, zeros, diff

        n_filters = 40  # must be 40 for mfcc
        n_coeffs = 13

        source_filename = 'Test.wav'
        samplerate = 44100
        win_s = 512
        hop_s = 128

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
        if (row == grid_size.nRow - 1 and grid_size == grid_size.nCol - 1):
            mfccs=mfccs+100
        return mfccs

    def Extract_Spectrogram_Sinusoid(row, col):
        fs = 10e3
        N = 1e5
        amp = 2 * np.sqrt(2)
        noise_power = 0.01 * fs / 2
        time = np.arange(N) / float(fs)
        mod = 500 * np.cos(2 * np.pi * 0.25 * time)
        carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
        # noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        # noise *= np.exp(-time / 5)
        # x = carrier + noise  # x is the sample
        x = carrier
        frequencies, times, spectrogram = signal.spectrogram(x, fs)
        if (row == grid_size.nRow and col == grid_size.nCol):
            spectrogram = spectrogram *100
        else:
            spectrogram = spectrogram + row + col
        return spectrogram

    def Extract_Pitch_Winwav(row, col):

        pitch_List = []
        sample_rate = 44100
        x = np.zeros(44100)
        for i in range(44100):
            x[i] = np.sin(2. * np.pi * i * 225. / sample_rate)

        # create pitch object
        p = aubio.pitch("yin", samplerate=sample_rate)

        # pad end of input vector with zeros
        pad_length = p.hop_size - x.shape[0] % p.hop_size
        x_padded = np.pad(x, (0, pad_length), 'constant', constant_values=0)
        # to reshape it in blocks of hop_size
        x_padded = x_padded.reshape(-1, p.hop_size)

        # input array should be of type aubio.float_type (defaults to float32)
        x_padded = x_padded.astype(aubio.float_type)

        if(row==grid_size.nRow and grid_size==grid_size.nCol):
            for frame, i in zip(x_padded, range(len(x_padded))):
                time_str = "%.2f" % (i * p.hop_size / float(sample_rate))
                pitch_candidate = p(frame)[0] + row + col + 100
                # print(pitch_candidate)
                pitch_List.append(pitch_candidate)
        else:
            for frame, i in zip(x_padded, range(len(x_padded))):
                time_str = "%.2f" % (i * p.hop_size / float(sample_rate))
                pitch_candidate = p(frame)[0] + row + col + 100
                # print(pitch_candidate)
                pitch_List.append(pitch_candidate)
        return pitch_List

    def Extract_Pitch(row, col):
        from aubio import source, pitch

        filename = 'Test.wav'

        downsample = 1
        samplerate = 44100 // downsample

        win_s = 4096 // downsample  # fft size
        hop_s = 512 // downsample  # hop size

        s = source(filename, samplerate, hop_s)
        samplerate = s.samplerate

        tolerance = 0.8

        pitch_o = pitch("yin", win_s, hop_s, samplerate)
        pitch_o.set_unit("midi")
        pitch_o.set_tolerance(tolerance)

        pitches = []
        confidences = []

        # total number of frames read
        total_frames = 0
        while True:
            samples, read = s()
            if (row == grid_size.nRow - 1 and grid_size == grid_size.nCol - 1):
                pitch = pitch_o(samples)[0] + 100
            else:
                pitch = pitch_o(samples)[0]
            # pitch = int(round(pitch))
            confidence = pitch_o.get_confidence()
            # if confidence < 0.8: pitch = 0.
            # print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
            pitches += [pitch]
            confidences += [confidence]
            total_frames += read
            if read < hop_s: break
        return pitches
