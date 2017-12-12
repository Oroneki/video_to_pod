import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# import IPython.display


class AudioSignalSegment:

    def __init__(self, segment, rate, verbose=True):
        self.segment = segment
        self.verbose = verbose
        self.frames = None
        self.frames_ = None
        self.y = None
        self.filter_banks = None
        self.rate = rate

    def load(self):
        data, rate = self.segment, self.rate
        self.data = data
        p = len(self.data) / self.rate
        if self.verbose:
            print('\nSegment: {} with rate {}'.format(
                len(self.segment), self.rate))
            print('Shape of data: ', self.data.shape)
            print('Arquivo com {:02f} segundos de duração'.format(p))
        return self.data, self.rate

    def get_mono_signal(self):
        y = np.mean(self.data, axis=1)
        self.y = y
        if self.verbose:
            print('Shape of y:', self.y.shape)
        return self.y

    def plot_mono(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.y)
        if self.verbose:
            print('Audio "Mono - Raw":')
        return plt.show()

    def emphasize_signal(self, pre_emphasis=0.95):
        signal = self.y
        self.emphasized_signal = np.append(
            signal[0], signal[1:] - pre_emphasis * signal[:-1])
        self.pre_emphasis = pre_emphasis
        if self.verbose:
            print('\npre-emphazis_rate: ', pre_emphasis, '.')

    def plot_emphasized(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.emphasized_signal)
        if self.verbose:
            print('Audio "Emphasized":')
        return plt.show()

    def make_frames(self, frame_size=0.025, frame_stride=0.01):
        sample_rate = self.rate
        emphasized_signal = self.emphasized_signal
        # Convert from seconds to samples
        frame_length = frame_size * sample_rate
        frame_step = frame_stride * sample_rate
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        # Make sure that we have at least 1 frame
        num_frames = int(
            np.ceil(
                float(np.abs(signal_length - frame_length)) /
                frame_step
            )
        )

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have
        #  equal number of samples without truncating any
        #  samples from the original signal
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(
            np.arange(
                0,
                frame_length),
            (num_frames, 1)) + np.tile(
                np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)).T
        frames_ = pad_signal[indices.astype(np.int32, copy=False)]
        self.frames_ = frames_
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.signal_length = signal_length
        self.num_frames = num_frames
        self.indices = indices
        if self.verbose:
            print('\nframe_size         ', frame_size, sep='\t')
            print('frame_stride       ', frame_stride, sep='\t')
            print('frame_length       ', frame_length, sep='\t')
            print('frame_step         ', frame_step, sep='\t')
            print('signal_lenght      ', signal_length, sep='\t')
            print('num_frames         ', num_frames, sep='\t')
            print('pad_signal_length  ', pad_signal_length, sep='\t')
            print('indices            ', self.indices.shape, sep='\t')
            print('\nframes_ shape ---> ', frames_.shape, sep='\t')

        return frames_

    # def audio_display_raw(self):
    #     if self.y is None:
    #         self.get_mono_signal()
    #     if self.verbose:
    #         print('Audio "raw":')
    #     return IPython.display.display(IPython.display.Audio(data=self.y, rate=self.rate))

    # def audio_display_emphasized(self):
    #     if self.y is None:
    #         self.get_mono_signal()
    #     if self.emphasized_signal is None:
    #         self.emphasize_signal()
    #     if self.verbose:
    #         print('Audio "Emphasized":')
    #     return IPython.display.display(IPython.display.Audio(data=self.emphasized_signal, rate=self.rate))

    def windowed_frames(self):
        if self.frames_ is None:
            self.make_frames()

        self.frames = self.frames_ * np.hamming(self.frame_length)
        if self.verbose:
            print('\nWindowed (frames):', self.frames.shape)
        return self.frames

    def nfft(self, NFFT=512):
        self.NFFT = NFFT
        if self.frames is None:
            self.windowed_frames()
        frames = self.frames
        mag_frames = np.absolute(np.fft.rfft(
            frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        self.pow_frames = pow_frames
        self.mag_frames = mag_frames
        if self.verbose:
            print('\nPower    :', self.pow_frames.shape)
            print('Magnitude:', self.mag_frames.shape)

    def plot_power_and_mag(self):
        pass

    def mel_pre(self, nfilt=40):
        NFFT = self.NFFT
        pow_frames = self.pow_frames
        mag_frames = self.mag_frames
        sample_rate = self.rate
        low_freq_mel = 0
        # Convert Hz to Mel
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        # Equally spaced in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(
            float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        self.filter_banks = filter_banks

        if self.verbose:
            print('\nnfilt .................', nfilt)
            print('low_freq_mel ..........', low_freq_mel)
            print('high_freq_mel .........', high_freq_mel)
            print('\nself.filter_banks(shape):', self.filter_banks.shape)

        return self.filter_banks

    def mel_mfcc(self, cep_lifter=4, num_ceps=12):
        # TODO cep_filter... pensar melhor o numero e fazer testes
        if self.filter_banks is None:
            self.mel_pre()
        from scipy.fftpack import dct
        filter_banks = self.filter_banks
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[
            :, 1: (num_ceps + 1)]  # Keep 2-13
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
        self.mfcc = mfcc
        if self.verbose:
            print('self.mfcc(shape):', self.mfcc.shape)

        return self.mfcc

    def parse_audacity_labels(self,
                              filepath,
                              framerate=None,
                              frame_lenght=None,
                              total_length=None,
                              sequnece_length=None,
                              empty_value=0):

        if framerate is None:
            framerate = self.rate

        if frame_lenght is None:
            frame_lenght = self.frame_length

        if total_length is None:
            total_length = self.num_frames

        if sequnece_length is None:
            sequnece_length = self.signal_length

        if self.verbose:
            print(framerate, frame_lenght, total_length, sequnece_length)

        with open(filepath) as f:
            lines = f.readlines()
        dic = {}
        dic['EMPTY'] = empty_value
        atual = 0
#         seq = np.zeros(int(round(total_length * framerate / frame_lenght)))
        seq = np.zeros(total_length)
        seq[:] = atual
        print('seq:', len(seq))
        div = sequnece_length / total_length
        for line in lines:
            print()
            i, f, l = line.split('\t')
            i, f, l = float(i), float(f), l.strip()
            if i < 0:
                i = float(0)
            print(i, f, l, sep='\t')
            inf = int(np.ceil((i * framerate) / div))
            sup = int(np.ceil((f * framerate) / div))
            print('inf:', inf, '     sup:', sup)
            l_num = dic.get(l)
            if not l_num:
                if atual == empty_value:
                    atual = atual + 1
                l_num = atual
                atual = atual + 1
                dic[l] = l_num
            seq[inf:sup] = l_num
        if self.verbose:
            for k, v in dic.items():
                print(k, v, sep='\t\t')
#         self.labels = seq
#         self.dic_labels = dic
        return seq, dic


class AudioFile:

    def __init__(self,
                 filepath,
                 verbose='',
                 ):
        self.filepath = filepath
        self.rate = sf.info(filepath).samplerate
        self.verbose = verbose
        self.frame_length = None
        self.num_frames = None
        self.signal_length = None
        self.mel = None

    def _lazy_load(self,
                   block_size=10,
                   pre_emphasis=0.97,
                   frame_size=0.05,
                   frame_stride=0.01,
                   nfilt=60,
                   cep_lifter=1,
                   num_ceps=12,
                   ):

        block_gen = sf.blocks(self.filepath, blocksize=self.rate * block_size)
        for block in block_gen:
            pedaco = AudioSignalSegment(block, self.rate, verbose=False)
            pedaco.load()
            pedaco.get_mono_signal()
            del pedaco.data
            pedaco.emphasize_signal(pre_emphasis=pre_emphasis)
            del pedaco.y
            pedaco.make_frames(frame_size=frame_size,
                               frame_stride=frame_stride)
            pedaco.nfft()
            del pedaco.emphasized_signal
            pedaco.mel_pre(nfilt=nfilt)
            mfcc_p = pedaco.mel_mfcc(
                cep_lifter=cep_lifter, num_ceps=num_ceps).astype(np.float32)
            if self.frame_length is None:
                self.frame_length = pedaco.frame_length
                self.num_frames = pedaco.num_frames
                self.signal_length = pedaco.signal_length
            else:
                #                 self.frame_length = self.frame_length + pedaco.frame_length
                self.num_frames = self.num_frames + pedaco.num_frames
                self.signal_length = self.signal_length + pedaco.signal_length

            if self.mel is None:
                self.mel = mfcc_p
                if self.verbose == 'v' or self.verbose == 'vv':
                    print('+', end='')
            else:
                self.mel = np.concatenate((self.mel, mfcc_p))
                if self.verbose == 'v' or self.verbose == 'vv':
                    print('.', end='')

        return self.mel

    def _yield_blocks_of_mel(self,
                             block_size=10,
                             pre_emphasis=0.97,
                             frame_size=0.05,
                             frame_stride=0.01,
                             nfilt=60,
                             cep_lifter=1,
                             num_ceps=12,
                             ):
        frames = 0
        block_gen = sf.blocks(self.filepath, blocksize=self.rate * block_size)
        for block in block_gen:
            pedaco = AudioSignalSegment(block, self.rate, verbose=False)
            pedaco.load()
            pedaco.get_mono_signal()
            del pedaco.data
            pedaco.emphasize_signal(pre_emphasis=pre_emphasis)
            del pedaco.y
            pedaco.make_frames(frame_size=frame_size,
                               frame_stride=frame_stride)
            pedaco.nfft()
            del pedaco.emphasized_signal
            pedaco.mel_pre(nfilt=nfilt)
            mfcc_p = pedaco.mel_mfcc(cep_lifter=cep_lifter, num_ceps=num_ceps)
            frames = frames + len(mfcc_p)
            yield mfcc_p

    def array_of_probas(self, modelo, scaler):
        aud = scaler.fit_transform(self.mel)
        probas = modelo.predict_proba(aud)
        del aud
        ###
        secs = self.signal_length / self.rate
        len_probas = len(probas)
        tam_jan = len_probas / secs
        print(secs, 'segundos para', len_probas, 'probas.')
        print(tam_jan, 'pb/sec.')
        tam_jan_int = int(np.ceil(tam_jan))
        print(tam_jan_int, '< -- tamanho janela')
        sobra = tam_jan - np.floor(tam_jan)
        print(sobra, '<- sobra')
        total_sobra = len_probas % int(np.floor(tam_jan))
        print(total_sobra, 'total_sobra')
        dic = {}
        probas_ = probas.transpose()
        del probas
        for i in range(len(probas_)):

            control = 0
            new_arr = np.array([])
            sup, control, desloc = tam_jan_int, 0, 0
            inf = 0
            ctt = 1
            while 1:
                try:
                    new_arr = np.concatenate(
                        [new_arr, [np.mean(probas_[i][inf:sup])]])
                    ctt = ctt + 1
                except:
                    print('Parou com:')
                    print('inf', inf, 'sup', sup, 'control', control)
                    break
                if sup > len_probas:
                    # print('break no if')
                    break
                control = control + sobra
                if desloc != int(np.floor(control)):
                    inf = inf - 1
                desloc = int(np.floor(control))
                inf = inf + tam_jan_int
                sup = inf + tam_jan_int
            dic[i] = new_arr
#         print(ctt, inf, sup, control, desloc, sep='\t')
#         print('FIM')
#         print(secs, 'segundos para', len_probas, 'probas.')
#         print(tam_jan, 'pb/sec.')
#         print(tam_jan_int, '< -- tamanho janela')
#         print(sobra, '<- sobra')
#         print(total_sobra, 'total_sobra')
#         print('conta:', ctt, '    probas:', len_probas, '    new_arr:', len(new_arr))
        return dic
