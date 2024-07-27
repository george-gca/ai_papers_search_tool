from dataclasses import dataclass, field


@dataclass
class PaperInfo:
    # indicates the weight of the word in this paper, being
    # key: index of word
    # value: weight of the word, associated with occurrence in title and/or abstract
    abstract_freq: dict[int, float] = field(default_factory=dict)
    abstract_url: str
    arxiv_id: None | str = None
    clean_title: str
    conference: str = ''
    pdf_url: str
    source_url: int
    title: str
    year: int = 0
