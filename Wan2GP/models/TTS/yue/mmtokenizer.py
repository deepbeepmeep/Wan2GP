from abc import ABC
from abc import abstractmethod


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        name = 'SentencePieceTokenizer'
        super().__init__(name)

        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

class _MMSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids)


    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        _add_special_token('<SOA>')
        self._soa_id = self._vocab['<SOA>']
        _add_special_token('<EOA>')
        self._eoa_id = self._vocab['<EOA>']
        _add_special_token('<SOV>')
        self._sov_id = self._vocab['<SOV>']
        _add_special_token('<EOV>')
        self._eov_id = self._vocab['<EOV>']
        _add_special_token('<SOI>')
        self._soi_id = self._vocab['<SOI>']
        _add_special_token('<EOI>')
        self._eoi_id = self._vocab['<EOI>']
        _add_special_token('<s_local>')
        self._s_local_id = self._vocab['<s_local>']
        _add_special_token('<e_local>')
        self._e_local_id = self._vocab['<e_local>']
        _add_special_token('<s_global>')
        self._s_global_id = self._vocab['<s_global>']
        _add_special_token('<e_global>')
        self._e_global_id = self._vocab['<e_global>']
        _add_special_token('<stage_1>')
        self._stage_1_id = self._vocab['<stage_1>']
        _add_special_token('<stage_2>')
        self._stage_2_id = self._vocab['<stage_2>']
        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]
    
    @property
    def soa(self):
        return self._soa_id
    
    @property
    def eoa(self):
        return self._eoa_id
    
    @property
    def sov(self):
        return self._sov_id
    
    @property
    def eov(self):
        return self._eov_id
    
    @property
    def soi(self):
        return self._soi_id
    
    @property
    def eoi(self):
        return self._eoi_id
    
    @property
    def s_local(self):
        return self._s_local_id
    
    @property
    def e_local(self):
        return self._e_local_id
    
    @property
    def s_global(self):
        return self._s_global_id

    @property
    def e_global(self):
        return self._e_global_id
    
    @property
    def stage_1(self):
        return self._stage_1_id

    @property
    def stage_2(self):
        return self._stage_2_id
