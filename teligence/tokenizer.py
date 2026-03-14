from dataclasses import dataclass


class TokenizerBase:
    @property
    def bos_id(self) -> int:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode_text(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode_ids(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def filter_supported_text(self, text: str) -> str:
        return text


@dataclass
class CharTokenizer(TokenizerBase):
    stoi: dict
    itos: dict
    _bos_id: int

    @classmethod
    def from_texts(cls, texts: list[str]):
        uchars = sorted(set("".join(texts)))
        stoi = {ch: i for i, ch in enumerate(uchars)}
        itos = {i: ch for ch, i in stoi.items()}
        bos_id = len(uchars)
        return cls(stoi=stoi, itos=itos, _bos_id=bos_id)

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) + 1

    def encode_text(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode_ids(self, token_ids: list[int]) -> str:
        return "".join(self.itos[i] for i in token_ids if i in self.itos)

    def filter_supported_text(self, text: str) -> str:
        return "".join(ch for ch in text if ch in self.stoi)


class ByteTokenizer(TokenizerBase):
    def __init__(self):
        self._bos_id = 256

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def vocab_size(self) -> int:
        return 257

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode_ids(self, token_ids: list[int]) -> str:
        b = bytes(i for i in token_ids if 0 <= i < 256)
        return b.decode("latin-1", errors="ignore")


def make_tokenizer(tokenizer_mode: str, dataset_name: str, docs_all: list[str] | None = None):
    mode = tokenizer_mode.strip().lower()
    if mode not in ("auto", "byte", "char"):
        raise ValueError(f"Unsupported TOKENIZER='{tokenizer_mode}'. Use auto|byte|char")

    if mode == "auto":
        mode = "byte" if dataset_name == "enwik8" else "char"

    if mode == "byte":
        return ByteTokenizer()

    if docs_all is None:
        raise ValueError("Char tokenizer requires docs/text corpus to build vocabulary")
    return CharTokenizer.from_texts(docs_all)
