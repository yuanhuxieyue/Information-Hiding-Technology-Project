from argparse import ArgumentParser
from typing import List, Union
from io import StringIO
import torch
from torch import Tensor
import torch.nn.functional as F
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer


class MaskedStego:
    def __init__(self, model_name_or_path: str = 'bert-base-cased') -> None:
        self._tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self._model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self._STOPWORDS: List[str] = stopwords.words('english')

    def __call__(self, cover_text: str, message: str, mask_interval: int = 3, score_threshold: float = 0.01) -> str:
        assert set(message) <= set('01')
        message_io = StringIO(message)  # 将字符串写入内存，这里直接在内存中读取文件，即为流读写，从而可以直接以读写文件的形式读写字符串
        processed = self._preprocess_text(cover_text, mask_interval)  # 掩码操作和模型处理，生成词汇表
        input_ids = processed['input_ids']  # 输入文本分词的ID
        masked_ids = processed['masked_ids']  # 掩码分词替换后的ID
        sorted_score, indices = processed['sorted_output']  # 概率和词汇表的分词索引编号
        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:  # 挑选出被掩码的分词
                continue
            # 选出当前分词的词汇表（即ids和scores）
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)  # 从词汇表找到符合条件的候选
            print(self._tokenizer.convert_ids_to_tokens(candidates))  # 显示符合条件的候选分词结果
            if len(candidates) < 2:    # 当词汇表的候选个数小于2也不予以考虑
                continue
            replace_token_id = self._block_encode_single(candidates, message_io).item()  # 块编码，选出用来替换的分词id
            print('replace', replace_token_id, self._tokenizer.convert_ids_to_tokens([replace_token_id]))
            input_ids[i_token] = replace_token_id
        encoded_message: str = message_io.getvalue()[:message_io.tell()]  # 读取到当前指针的位置，即编码结束的秘密信息位置，这是因为可能填充末尾0
        message_io.close()
        #  将数字id解码成文本
        stego_text = self._tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return { 'stego_text': stego_text, 'encoded_message': encoded_message }

    def decode(self, stego_text: str, mask_interval: int = 3, score_threshold: float = 0.01) -> str:
        decoded_message: List[str] = []
        processed = self._preprocess_text(stego_text, mask_interval)  # 掩码
        input_ids = processed['input_ids']
        masked_ids = processed['masked_ids']
        sorted_score, indices = processed['sorted_output']
        for i_token, token in enumerate(masked_ids):
            if token != self._tokenizer.mask_token_id:
                continue
            # 将该掩码分词的词汇表分词和概率选出来，并选出其中可用的候选值
            ids = indices[i_token]
            scores = sorted_score[i_token]
            candidates = self._pick_candidates_threshold(ids, scores, score_threshold)
            print(self._tokenizer.convert_ids_to_tokens(candidates))
            if len(candidates) < 2:  # 若该词的候选不足2，则表示能被该词选择的候选过于少，因此选择跳过该词不进行隐写的编码
                continue
            chosen_id: int = input_ids[i_token].item()
            print('choose', chosen_id, self._tokenizer.convert_ids_to_tokens([chosen_id]))
            decoded_message.append(self._block_decode_single(candidates, chosen_id))
        return {'decoded_message': ''.join(decoded_message)}

    def _preprocess_text(self, sentence: str, mask_interval: int) -> dict:
        encoded_ids = self._tokenizer([sentence], return_tensors='pt').input_ids[0]  # 将语句编码成数字，'pt'表示返回值为tensor
        masked_ids = self._mask(encoded_ids.clone(), mask_interval)  # clone() 进行深拷贝，跳转掩码函数，进行分词标记的替换
        sorted_score, indices = self._predict(masked_ids)  # 通过掩码函数的结果，得到概率排序和索引
        return { 'input_ids': encoded_ids, 'masked_ids': masked_ids, 'sorted_output': (sorted_score, indices) }

    # 将初始分词编码的文本进行mask掩码
    def _mask(self, input_ids: Union[Tensor, List[List[int]]], mask_interval: int) -> Tensor:
        length = len(input_ids)
        tokens: List[str] = self._tokenizer.convert_ids_to_tokens(input_ids)  # 将数字转化为分词列表
        # 设置掩码判断数目的初始大小(从而修改第一个被掩码词的位置)，此处设置为掩码间隔的一半（向下取整）+1
        mask_count = mask_interval // 2 + 1
        for i, token in enumerate(tokens):
            # 跳过初始子词
            if i + 1 < length and self._is_subword(tokens[i + 1]): continue
            # 判断当前分词的属性
            if not self._substitutable_single(token): continue
            # 若通过上述判断，则是可掩码分词，此时根据当前掩码判断的数量和掩码间隔的整除关系进行判断，每隔mask_interval（f）个分词就掩码一次
            if mask_count % mask_interval == 0:
                input_ids[i] = self._tokenizer.mask_token_id
                # print(i)
            mask_count += 1  # 每判断完一个分词，就将mask_count+1
        return input_ids

    # 利用模型对每个[MASK]分词进行原始分词的预测
    def _predict(self, input_ids: Union[Tensor, List[List[int]]]):
        self._model.eval()  # 现成模型，直接设置为评估状态
        with torch.no_grad():  # 无梯度更新
            output = self._model(input_ids.unsqueeze(0))['logits'][0]  # 使用MLM模型进行掩码原始词的预测
            softmaxed_score = F.softmax(output, dim=1)  # [word_len, vocab_len]，归一化处理，按列将输出整合为概率的形式
            return softmaxed_score.sort(dim=1, descending=True)  # 对结果概率分布进行倒序排序

    # 符合条件的候选，包括概率大于阈值，并且满足掩码策略的条件，即非子词、非标号、非停用词
    def _pick_candidates_threshold(self, ids: Tensor, scores: Tensor, threshold: float) -> List[int]:
        filtered_ids: List[int] = ids[scores >= threshold]

        def filter_fun(idx: Tensor) -> bool:
            return self._substitutable_single(self._tokenizer.convert_ids_to_tokens(idx.item()))
        return list(filter(filter_fun, filtered_ids))  # 如果不满足条件，则需要跳过

    # 子词和停用词和标点符号就跳过（此即便编码处理策略）
    def _substitutable_single(self, token: str) -> bool:
        if self._is_subword(token): return False
        if token.lower() in self._STOPWORDS: return False
        if not token.isalpha(): return False
        return True

    @staticmethod
    #  对剩余候选进行块编码，进行末尾的填0
    def _block_encode_single(ids: List[int], message: StringIO) -> int:
        capacity = len(ids).bit_length() - 1  # 此处实现了2^n<=c的要求
        # 读取capacity个字符
        bits_str = message.read(capacity)
        if len(bits_str) < capacity:  # 当我们的信息不够对于最后一个bit块选取完整编码时，则进行填充
            padding: str = '0' * (capacity - len(bits_str))
            bits_str = bits_str + padding
            message.write(padding)
        index = int(bits_str, 2)  # 按照隐写信息，如00、01、10、11之类的，将候选按顺序编码，因此在这里直接用index可以索引到隐藏了相应bit信息的候选
        return ids[index]

    @staticmethod
    def _block_decode_single(ids: List[int], chosen_id: int) -> str:
        capacity = len(ids).bit_length() - 1
        index = ids.index(chosen_id)
        return format(index, '0' + str(capacity) + 'b')  # 将索引转化为2进制，即为该词隐写的秘密信息

    @staticmethod
    def _is_subword(token: str) -> bool:
        return token.startswith('##')


if __name__ == "__main__":
    psr = ArgumentParser()
    psr.add_argument('text', type=str, help='Text to encode or decode message.')
    psr.add_argument('-d', '--decode', action='store_true', help='If this flag is set, decodes from the text.')
    psr.add_argument('-m', '--message', type=str, help='Binary message to encode consisting of 0s or 1s.')
    psr.add_argument('-f', '--mask_interval', type=int, default=3)
    psr.add_argument('-p', '--score_threshold', type=float, default=0.01)
    args = psr.parse_args()

    masked_stego = MaskedStego()

    if args.decode:
        print(masked_stego.decode(args.text, args.mask_interval, args.score_threshold))
    else:
        print(masked_stego(args.text, args.message, args.mask_interval, args.score_threshold))
