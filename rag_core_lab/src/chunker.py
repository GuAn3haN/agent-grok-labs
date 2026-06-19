import re

class SimpleChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        # 1. 基于标点分句
        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # 如果单个句子超过 chunk_size，需要硬切
            if len(sentence) > self.chunk_size:
                # 先把当前累积的块保存
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # 硬切长句：按 chunk_size - overlap 滑动（注意 overlap < chunk_size）
                step = self.chunk_size - self.chunk_overlap
                start = 0
                while start < len(sentence):
                    end = min(start + self.chunk_size, len(sentence))
                    chunks.append(sentence[start:end])
                    start += step
                continue

            # 正常累积
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                chunks.append(current_chunk)
                # 重叠处理
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:] + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks