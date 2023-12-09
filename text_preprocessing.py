"""文本预处理"""
import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f"# 文本总行数：{len(lines)}")
print(lines[0])
print(lines[10])


# 词元化 token
def tokenize(lines, token='word'):
    """将文本拆分为单词或字符词元

    Args:
        lines (list): 文本行
        token (str, optional): 拆分词元标准. Defaults to 'word'.
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f'错误：未知词元类型：{token}')
    
tokens = tokenize(lines)
print(tokens[0])
print(tokens[10])


# 词表
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is []:
            return []
        if reserved_tokens is None:
            reserved_tokens = []
        # tokens排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        
        # id_to_token:list   token_to_ids:dict
        # 初始化未知token
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


# 整合函数功能
def load_corpus_time_machine(max_tokens=-1):
    """ 返回corpus(词元列表, 将文本分成词元后映射到vocab下标上)；vocab(语料库的词表)"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
