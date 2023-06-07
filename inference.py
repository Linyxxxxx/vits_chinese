import os
import numpy as np

import torch
import utils

from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


class VITS_CHINESE:

    def __init__(self, model_path, config_path='configs/bert_vits.json', bert_path='bert') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts_front = VITS_PinYin(bert_path, self.device)
        self.hps = utils.get_hparams_from_file(config_path)

        # model
        self.net_g = utils.load_class(self.hps.train.eval_class)(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)

        utils.load_model(model_path, self.net_g)
        self.net_g.eval()
        self.net_g.to(self.device)


    def tts(self, content, out_path='out.wav'):
        if (content == None or content == ""):
            return
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(content)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
                                length_scale=1)[0][0, 0].data.cpu().float().numpy()
        save_wav(audio, out_path, self.hps.data.sampling_rate)
