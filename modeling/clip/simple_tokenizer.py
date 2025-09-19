import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    返回 UTF-8 字节和相应的 Unicode 字符串的映射。
    这个映射会为 BPE 编码提供字节到 Unicode 字符的查找表，避免将空白字符/控制字符映射到 BPE 上。
    """
    # 生成 ASCII 范围内的字节范围 [33, 126] 和部分扩展的字符范围 [161, 255]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # 初步复制 bs（用于映射的字符）
    n = 0

    # 为所有 256 个字节添加映射，避免与 bs 重复的字符
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)  # 映射到 Unicode 字符
            n += 1

    # 将所有映射值转换成对应的 Unicode 字符
    cs = [chr(n) for n in cs]

    # 返回字节到 Unicode 字符的字典映射
    return dict(zip(bs, cs))


def get_pairs(word):
    """返回一个单词中的符号对集合。单词以符号（变量长度字符串）元组表示"""
    pairs = set()  # 用于存储符号对
    prev_char = word[0]

    # 遍历 word 中每个字符，并获取相邻字符的符号对
    for char in word[1:]:
        pairs.add((prev_char, char))  # 将当前字符与前一个字符组合为一个符号对
        prev_char = char

    return pairs  # 返回符号对集合


def basic_clean(text):
    text = ftfy.fix_text(text)  # 修复文本中的编码错误
    text = html.unescape(html.unescape(text))  # 解码 HTML 实体
    return text.strip()  # 去除多余的前后空白字符



def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)  # 将多个空格替换为一个空格
    text = text.strip()  # 去掉前后空格
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        # 构造函数初始化字节到 Unicode 映射和 BPE 相关的各种字典
        self.byte_encoder = bytes_to_unicode()  # 字节到 Unicode 的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # Unicode 到字节的反向映射

        # 读取 BPE 合并文件并解析合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]  # 从文件中读取和截断不需要的行
        merges = [tuple(merge.split()) for merge in merges]  # 每个合并规则用元组表示

        # 初始化词汇表
        vocab = list(bytes_to_unicode().values())  # 将字节映射为字符的映射添加到词汇表
        vocab = vocab + [v + '</w>' for v in vocab]  # 将词汇表的每个字符添加 '/w' 后缀

        # 将 BPE 合并规则和结束符号添加到词汇表
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # 创建字典：词汇表 -> 索引，反向映射
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 创建 BPE 合并操作字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # 定义文本的正则表达式模式
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)  # 获取单词中的所有符号对

        if not pairs:
            return token + '</w>'  # 如果没有符号对，直接返回 token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break  # 如果没有找到合并对，则跳出循环
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
