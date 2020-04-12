import random
from vi_utils.text import word2phone_2, phone2numeric
from ZaG2P.api import G2S, load_model

wind_sound_full_dict = {'sờ': 's',
                    'xờ' : 's',
                    'dờ': 'd',
                    'tờ': 't',
                    'vờ': 'v',
                    'bờ': 'b',
                    'pờ': 'p',
                    'cờ': 'c',
                    'đờ': 'đ',
                    'gờ': 'g',
                    'trờ': 'ch',
                    'chờ': 'ch',
                    'lờ': 'l',
                    'nờ': 'n',
                    'phờ': 'ph',
                    'thờ': 'th'}
wind_sound_list = ['vờ', 'đờ', 'pờ', 'bờ', 'gờ', 'phờ', 'trờ', 'chờ', 'cờ', 'xờ', 'sờ', 'thờ']
end_wind_sound_list = ['xờ', 'sờ']

class TextEmbedding:
    def __init__(self, hparams, word2phone_dict={}, symbol2numeric_dict={}, ws_list=wind_sound_list, end_ws_list=end_wind_sound_list, load_g2s=True):
        self.p_phone_mix = hparams.p_phone_mix
        self.punctuation = hparams.punctuation
        self.eos = hparams.eos
        self.connect_vn = hparams.connect_vn
        self.connect_oov = hparams.connect_oov
        if word2phone_dict:
            self.word2phone_dict = word2phone_dict
        else:
            self.word2phone_dict = word2phone_2(hparams.phone_vn_train, hparams.phone_oov_train, hparams.nucleus)
                   
        letters_lst = list(' ' + self.punctuation + self.eos + hparams.letters)
        self.word2phone_dict[' '] = 'space'
        for c in (self.punctuation + self.eos):
            self.word2phone_dict[c] = c
        if hparams.word_tokenize:
            letters_lst += list(self.connect_vn)
            self.word2phone_dict[self.connect_vn] = self.connect_vn         
        if hparams.spell_oov:
            letters_lst += list(self.connect_oov)  
            self.word2phone_dict[self.connect_oov] = self.connect_oov
        phone2numeric_dict = phone2numeric(self.word2phone_dict)
        phonemes_lst = list(phone2numeric_dict.keys())
        if 1 > self.p_phone_mix > 0:
            symbols = letters_lst + phonemes_lst
        elif self.p_phone_mix >= 1:
            symbols = phonemes_lst
        else:
            symbols = letters_lst
        if symbol2numeric_dict:
            self.symbol2numeric_dict = symbol2numeric_dict
        else:
            self.symbol2numeric_dict ={s: i for i, s in enumerate(symbols)}
        
        self.ws_full_dict = wind_sound_full_dict
        self.ws_list = ws_list
        self.end_ws_list = end_ws_list
        if load_g2s:
            self.g2s_model, self.g2s_dict = load_model()
        else:
            self.g2s_model, self.g2s_dict = None, None
            
        
    def g2s_oov(self, word):
        result = G2S(word, self.g2s_model, self.g2s_dict)
        syllables = ''
        if result != word:
            syl_list = result[0].split()[1:]
            if len([syl for syl in syl_list if '(' in syl]) != len(syl_list):
                while syl_list and '(' in syl_list[-1] and syl_list[-1].replace('(', '').replace(')', '') not in self.end_ws_list:
                    syl_list = syl_list[:-1]
                for syl in syl_list:
                    if '(' in syl:
                        syl = syl.replace('(', '').replace(')', '')
                        if syl in self.ws_list:
                            syl = self.ws_full_dict[syl]
                        else:
                            syl = ''
                    if syl:
                        syllables += syl + self.connect_oov
                syllables = syllables[:-1]
            else:
                syllables = ' '.join([syl.replace('(', '').replace(')', '') for syl in syl_list])
        syllables = syllables.replace('j', 'gi').replace('z', 'd').replace('f', 'ph').replace('w', 'g')
        return syllables

    
    def norm_oov(self, word):
        word_norm = ''
        syl_list = word.split(self.connect_oov)
        if len([syl for syl in syl_list if syl in [self.ws_full_dict[ws] for ws in self.ws_list]]) != len(syl_list):
            while syl_list and syl_list[-1] in self.ws_full_dict.values() and syl_list[-1] not in [self.ws_full_dict[ws] for ws in self.end_ws_list]:
                syl_list = syl_list[:-1]
            for syl in syl_list:
                if syl == 'x':
                    syl = 's'
                if syl == 'tr':
                    syl = 'ch'
                if syl not in self.ws_full_dict.values() or syl in [self.ws_full_dict[ws] for ws in self.ws_list]:
                    word_norm += syl + self.connect_oov
        return word_norm[:-1]
    
    
    def text_norm(self, text):
        words = text.split()
        while words[-1] in self.punctuation or words[-1] == self.eos:
            words = words[:-1]
        text_out = ''
        for word in words:
            if self.connect_oov in word:
                word_norm = self.norm_oov(word)
                text_out += word_norm + ' '
            else:
                text_out += word + ' '
        text_out = text_out.strip() + ' ' + self.eos
        return text_out

    
    def g2s(self, text):
        text_output = ''
        oov_g2s_dict = {}
        for word in text.split():
            if word and not word.startswith("#") and word not in self.word2phone_dict.keys() and self.connect_oov not in word and self.connect_vn not in word and word not in (self.punctuation + self.eos):
                syllables = self.g2s_oov(word)
                if word not in oov_g2s_dict.keys():
                    oov_g2s_dict[word] = syllables.replace(self.connect_oov, ' ')
                text_output += syllables + ' '
            else:
                if word.startswith("#") and word != self.eos:
                    oov_g2s_dict[word] = ''
                else:
                    word = word.replace('j', 'gi').replace('z', 'd').replace('f', 'ph').replace('w', 'g')
                    text_output += word + ' '
        return text_output.strip(), oov_g2s_dict
    
        
    def word2phone(self, word):
        phonemes = ''
        if self.connect_vn in word:
            word = word.replace(self.connect_vn, ' ' + self.connect_vn + ' ')
            for syl in word.split():
                if syl in self.word2phone_dict.keys():
                    phonemes += self.word2phone_dict[syl] + ' '
                elif syl in [self.ws_full_dict[ws] for ws in self.ws_list]:
                    phonemes += syl + ' '
            phonemes = phonemes[:-1]
        elif self.connect_oov in word:
            word = self.norm_oov(word)
            for syl in word.split(self.connect_oov):
                if syl in self.word2phone_dict.keys():
                    phonemes += self.word2phone_dict[syl] + ' '
                elif syl in [self.ws_full_dict[ws] for ws in self.ws_list]:
                    phonemes += syl + ' '
            phonemes = phonemes[:-1]
        else:
            if word in self.word2phone_dict.keys():
                phonemes = self.word2phone_dict[word]
        #print(phonemes)
        return phonemes    
                   

    def text2seq(self, text):
        sequence = []
        for word in text.split():
            if random.random() < self.p_phone_mix:
                phonemes = self.word2phone(word)
                for phoneme in phonemes.split():
                    if phoneme:
                        sequence.append(self.symbol2numeric_dict[phoneme])
                sequence.append(self.symbol2numeric_dict[self.word2phone_dict[' ']])
            elif self.p_phone_mix >= 1 and word not in self.word2phone_dict.keys():
                print(f"{word} not in phone_train_dict")
            else:
                for symbol in word:
                    if symbol in self.symbol2numeric_dict.keys():
                        sequence.append(self.symbol2numeric_dict[symbol])
                    else:
                        print(f"{symbol} not in symbols_dict\nText: {text}")
                sequence.append(self.symbol2numeric_dict[' '])
        return sequence[:-1]
    

if __name__ == '__main__':
    from hparams_zalo import create_hparams_and_paths
    hparams, path = create_hparams_and_paths()
    text_embedding = TextEmbedding(hparams)
    print(text_embedding.symbol2numeric_dict)
    print(text_embedding.word2phone('ủy_ban'))
    text = '# lyonnais price bình đẳng chính là việc _ - thúc đẩy các hành động nhằm giải quyết các vấn đề của phụ nữ qua các thếhệ , từ những năm đầu cho đến những năm về sau và ở đó phụ nữ và trẻ em gái đượ đặt ở vị trí price trung tâm a , b , n ~ đến_cùng'
    text_norm = text_embedding.text_norm(text)
    print(f'text={text}')
    text, oov_g2s_dict = text_embedding.g2s(text)
    print(f'text={text}')
    print(oov_g2s_dict)
    sequence = text_embedding.text2seq(text)
    print(sequence)

