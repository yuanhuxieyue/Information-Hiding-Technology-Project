快速开始：代码运行示例如下，详细见实验报告

# Edited_Masked_Stego

## Example usage
编码：
```bash
$ python main.py "The quick brown fox jumps over the lazy dog." -m 010101

{'stego_text': 'The quick red fox jumps over the poor dog.', 'encoded_message': '010101'}
```

解码：

```bash
$ python main.py "The quick red fox jumps over the poor dog." --decode

{'decoded_message': '010101'}
```

# Addressing Segmentation Ambiguity(Generation-based )

## Example Usage
编码：
```bash
$ python encode.py "010101011111010101101010" --prompt "Hi Bob." --language "en"

 It would appear there's nothing new.
```
解码：
```bash
$ python decode.py " It would appear there's nothing new." --prompt "Hi Bob." --language "en"

01010101111101010110101000
