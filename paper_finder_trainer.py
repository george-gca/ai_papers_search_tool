import locale
import logging
import math
import multiprocessing
import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from fasttext import FastText
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from paper_finder import PaperFinder
from paperinfo import PaperInfo
from timer import Timer

# Use '' for auto, or force e.g. to 'en_US.UTF-8'
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
TQDM_NCOLS = 175


class PaperFinderTrainer(PaperFinder):
    # https://fasttext.cc/docs/en/unsupervised-tutorial.html
    def __init__(self, word_dim: int = 100, data_dir: Path = Path('data/'), model_dir: Path = Path('model_data/'), max_dictionary_words: int=30_000,
                 title_vector_weight: float = 3.0, title_search_weight: float = 3.0):
        PaperFinder.__init__(self, model_dir=model_dir)
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.count: Counter = None
        self.data_dir: Path = data_dir.expanduser()
        self.dictionary: Set[str] = None
        self.max_dictionary_words: int = max_dictionary_words
        self.max_ngram: int = 1
        self.model: Any = None
        self.title_search_weight: float = title_search_weight
        self.title_vector_weight: float = title_vector_weight
        self.words: List[str] = []
        self.word_dim: int = word_dim

        # values used for building dictionary
        # mark end of paper, used for avoiding aglutinating words from different papers
        self.eop: str = '<EOP>'
        # mark unknown word, used for replacing not frequent words and avoid aglutinating with common words
        self.unk: str = '<UNK>'

        if not self.data_dir.exists():
            self.logger.debug(f'Creating folder {self.data_dir}')
            self.data_dir.mkdir(parents=True)
        if not self.model_dir.exists():
            self.logger.debug(f'Creating folder {self.model_dir}')
            self.model_dir.mkdir(parents=True)

    def _create_ngrams_count(self, word_list: List[str], n: int, top_n: int) -> List[Tuple[List[str], int]]:
        with Timer(name=f'Creating {n}-grams'):
            ngrams = [l for l in tqdm(zip(
                *(word_list[i:] for i in range(n))), total=len(word_list)+1-n, unit='ngram', desc=f'Creating {n}-grams', ncols=TQDM_NCOLS) if not self.unk in l and not self.eop in l]
            self.logger.info(f'\nCreated {len(ngrams):n} {n}-grams')

        with Timer(name=f'Counting {n}-grams'):
            result = Counter(ngrams).most_common(top_n)

        return result

    def _create_ngrams_set(self, word_list: List[str], n: int, ngram_threshold: int = 1000) -> Set[str]:
        # remove composite words from word list
        word_list = [w for w in word_list if '_' not in w]

        # probably the number of most common n-grams that happens more than
        # ngram_threshold times is lesser than a portion of dictionary size
        top_n = self.max_dictionary_words // (3 * n)
        self.logger.info(
            f'Checking which {n}-grams occurs more than {ngram_threshold:n} times '
            f'from the top {top_n:n} most frequent ones\n')

        ngrams_counts = self._create_ngrams_count(
            word_list, n, top_n)
        ngrams_counts = [('_'.join(ngram), count) for ngram, count in ngrams_counts if count >= ngram_threshold]

        if len(ngrams_counts) == top_n:
            self.logger.info(f'The number of {n}-grams is larger than {top_n:n}. Checking for more')
            while len(ngrams_counts) == top_n:
                # since number of most common n-grams is equal to top_n,
                # double top_n size to check for more n-grams
                top_n *= 2
                ngrams_counts = self._create_ngrams_count(
                    word_list, n, top_n)
                ngrams_counts = [('_'.join(ngram), count) for ngram, count in ngrams_counts if count >= ngram_threshold]

        for ngram, count in ngrams_counts:
            self.logger.print(f'{ngram}: {count:n}')

        return {ngram for ngram, _ in ngrams_counts}

    def _replace_words_by_ngrams(self, text: str) -> str:
        words = text.split()
        for n in reversed(range(2, self.max_ngram)):
            self.logger.debug(f'Replacing {n}-grams')
            i = n
            while i < len(words):
                ngram = '_'.join(words[i-n:i])
                if ngram in self.dictionary:
                    words[i-n] = ngram
                    i -= n-1
                    j = n-1
                    while j > 0:
                        words.pop(i)
                        j -= 1

                else:
                    i += 1

        return ' '.join(words)

    def add_dictionary_from_file(self, file_path: Path, column: str = 'paper') -> None:
        def _add_words(abstract, words):
            words += abstract.split()
            words += [self.eop]

        df = pd.read_csv(file_path, sep='|', dtype=str, keep_default_na=False)
        words = []
        df[column].apply(_add_words, words=words)

        self.logger.print(f'File: {file_path} - Words: {len(words):n}')
        self.words += words

    def build_dictionary(self, rebuild=False) -> None:
        with Timer(name='Counting words occurrences'):
            if self.max_dictionary_words > 0:
                if rebuild:
                    self.count = Counter(self.words).most_common(self.max_dictionary_words + 1)
                else:
                    self.count = Counter(self.words).most_common(self.max_dictionary_words)
            else:
                self.count = Counter(self.words).most_common()

        self.dictionary = { w for w, _ in self.count if w != self.unk and w != self.eop }
        self.words = [w if w in self.dictionary else self.unk for w in tqdm(self.words, unit='word', desc='Rebuilding list of words', ncols=TQDM_NCOLS)]

        with Timer(name='Counting words occurrences after removing least frequent ones'):
            if self.max_dictionary_words > 0:
                self.count = Counter(self.words).most_common(self.max_dictionary_words + 1)
            else:
                self.count = Counter(self.words).most_common()

        # probably the most common word is self.unk, which will be in index 0
        self.logger.print(f'Finished building dictionary with {len(self.dictionary):n} words.\n'
              f'{self.count[0][1]:n} words replaced by {self.count[0][0]} since they are not frequent enough.')

    def build_paper_vectors(self, input_file: Path, suffix: str='', filter_titles: Optional[Set[str]] = None, filter_conferences: Optional[Set[str]] = None) -> None:
        extension = input_file.suffix[1:] # excluding first char since it is .
        self.load_paper_info(input_file.parent / f'paper_info{suffix}.{extension}')

        # load paper abstract and build paper representation vector

        # associates word to its index in the abstract_words list
        # e.g.: 'model': 1, 'learning': 2
        self.abstract_dict = {}
        self.abstract_words = []

        if 'csv' in extension:
            df = pd.read_csv(input_file, sep='|', dtype=str, keep_default_na=False)
        elif 'feather' in extension:
            df = pd.read_feather(input_file)
            df.dropna(inplace=True)

        # self.papers is built from paper_info_pwc.feather
        # df is built from abstracts_5gram.feather
        assert len(df) == len(self.papers), f'Sizes {len(df)} and {len(self.papers)} differ'

        if filter_titles is not None and len(filter_titles) > 0:
            self.logger.info(f'Filtering papers by title before building vectors')
            self.logger.info(f'Papers before: {len(df):n}')

            indices = df[df['title'].isin(filter_titles)].index
            df.drop(indices, inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.papers = [p for i, p in enumerate(self.papers) if i not in indices]

            self.logger.info(f'Papers after: {len(df):n}')
            self.n_papers = len(self.papers)

            assert len(df) == len(self.papers), f'Sizes {len(df)} and {len(self.papers)} now differ'

        if filter_conferences is not None and len(filter_conferences) > 0:
            self.logger.info(f'Filtering papers by conference before building vectors')
            self.logger.info(f'Papers before: {len(df):n}')

            indices = df[df['conference'].isin(filter_conferences)].index
            df.drop(indices, inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.papers = [p for i, p in enumerate(self.papers) if i not in indices]

            # remove papers from conferences like 'W18-5604' and 'C18-1211', which are usually from aclanthology and are not
            # with the correct conference name
            indices = df[df.conference.str.contains(r'[\w][\d]{2}-[\d]{4}')].index
            df.drop(indices, inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.papers = [p for i, p in enumerate(self.papers) if i not in indices]

            self.logger.info(f'Papers after: {len(df):n}')
            self.n_papers = len(self.papers)

            assert len(df) == len(self.papers), f'Sizes {len(df)} and {len(self.papers)} now differ'


        def _build_paper_vector(row):
            index = row.name
            title = row['clean_title']

            self.paper_vectors[index] += self.model.get_sentence_vector(
                title) * self.title_vector_weight

            self.paper_vectors[index] += self.model.get_sentence_vector(
                row['abstract'])

            for word in title.split():
                if word not in self.abstract_dict:
                    self.abstract_dict[word] = len(self.abstract_dict)
                    self.abstract_words.append(word)

                word_idx = self.abstract_dict[word]

                if word_idx in self.papers[index].abstract_freq:
                    self.papers[index].abstract_freq[word_idx] += self.title_search_weight
                else:
                    self.papers[index].abstract_freq[word_idx] = self.title_search_weight

            for word in row['abstract'].split():
                if word not in self.abstract_dict:
                    self.abstract_dict[word] = len(self.abstract_dict)
                    self.abstract_words.append(word)

                word_idx = self.abstract_dict[word]

                if word_idx in self.papers[index].abstract_freq:
                    self.papers[index].abstract_freq[word_idx] += 1
                else:
                    self.papers[index].abstract_freq[word_idx] = 1

        self.paper_vectors = np.zeros([self.n_papers, self.word_dim])
        tqdm.pandas(desc="Building papers' vectors", unit='abstract', ncols=TQDM_NCOLS)
        df.progress_apply(_build_paper_vector, axis=1)

        self.nearest_neighbours = KDTree(self.paper_vectors)

    def build_similar_dictionary(self, count: int = 5) -> None:
        similar_dictionary = {word: self.get_most_similar_words(word, count) for word in tqdm(
            self.words, desc='Creating dictionary of similar words', unit='word', ncols=TQDM_NCOLS)}

        self._save_object(self.model_dir / 'similar_dictionary', similar_dictionary)

    def clustering_papers(self, clusters: int = 10) -> None:
        if self.paper_vectors.shape[0] < clusters * 10:
            new_n_clusters = self.paper_vectors.shape[0] // 10
            self.logger.warning(
                f'Number of papers ({self.paper_vectors.shape[0]}) is less than 10 times the number of clusters ({clusters}).'
                f' Setting number of clusters to {new_n_clusters}')
            estimator = KMeans(init='k-means++', n_clusters=new_n_clusters, n_init=10)
        else:
            estimator = KMeans(init='k-means++', n_clusters=clusters, n_init=10)

        estimator.fit(self.paper_vectors)
        self.paper_cluster_ids = estimator.labels_
        self.cluster_abstract_freq = []

        for i in tqdm(range(clusters), desc='Generating clusters of frequent words', ncols=TQDM_NCOLS):
            cluster_papers_index = [j for j in range(
                self.n_papers) if self.paper_cluster_ids[j] == i]

            counter = Counter(
                self.papers[cluster_papers_index[0]].abstract_freq)
            for j in cluster_papers_index[1:]:
                counter += Counter(self.papers[j].abstract_freq)

            abstract = dict(counter)
            self.cluster_abstract_freq.append(
                sorted(abstract.items(), key=lambda x: x[1], reverse=True))

    def convert_text_with_phrases(self, src_file: Path, dest_file: Path, column: str = 'abstract') -> None:
        if 'csv' in src_file.suffix:
            df = pd.read_csv(src_file, sep='|', dtype=str, keep_default_na=False)
        else: # if 'feather' in src_file.suffix:
            df = pd.read_feather(src_file)

        self.logger.print(f'Building new text file on {dest_file}')
        tqdm.pandas(unit='word', desc=f'Replacind words by n-grams', ncols=TQDM_NCOLS)
        df[column] = df[column].progress_apply(self._replace_words_by_ngrams)

        if 'csv' in dest_file.suffix:
            df.to_csv(dest_file, sep='|', index=False)
        else: # if 'feather' in dest_file.suffix:
            df.to_feather(dest_file, compression='zstd')

    def create_corpus_with_phrases(self, file_path: Path) -> None:
        self.logger.print(f'Building new corpus on {file_path}')
        with open(file_path, 'w') as target:
            target.write(' '.join(self.words))

    def detect_ngrams(self, n: int, ngram_threshold: int = 1000) -> None:
        def _chunks(words, chunk_size):
            for i in range(0, len(words), chunk_size):
                yield words[i:i + chunk_size]

        ngrams_set = self._create_ngrams_set(self.words, n, ngram_threshold)

        if len(ngrams_set) > 0:
            self.logger.info(
                f'\nOnly {len(ngrams_set):n} {n}-grams occurs more than {ngram_threshold:n} times')

            chunk_size = 500_000 // n # words
            new_words = []

            with tqdm(total=math.ceil(len(self.words) / chunk_size) * len(ngrams_set), unit='chunk', desc=f'Replacing words by frequent {n}-grams', ncols=TQDM_NCOLS) as pbar:
                for chunk in _chunks(self.words, chunk_size):
                    words = f' {" ".join(chunk)} '
                    for ngram in ngrams_set:
                        regex = f'\b{" ".join(ngram.split("_"))}\b'
                        words = re.sub(regex, f'\b{ngram}\b', words)
                        pbar.update(1)

                    new_words += words.strip().split()

            self.words = new_words

            if n > self.max_ngram:
                self.max_ngram = n

            self.logger.debug(f'Max n-gram: {self.max_ngram}')
            self.build_dictionary(rebuild=True)

        else:
            self.logger.info(
                f'No {n}-grams occurs more than {ngram_threshold} times')

    def get_most_similar_words(self, target_word: str, count: int = 5) -> List[Tuple[float, str]]:
        return self.model.get_nearest_neighbors(target_word, k=count)

    def load_paper_info(self, paper_info_file: Path) -> None:
        def _add_paper_info(row, papers_info, accents):
            if 'conference' in row:
                conference = row['conference']
            else:
                conference = ''

            if 'year' in row:
                year = int(row['year'])
            else:
                year = 0

            # clean paper title
            paper_title = row['title'].lower()
            for k, v in accents.items():
                paper_title = re.sub(k, v, paper_title)
            paper_title = re.sub('([\w]+)[\-\−\–]([\w]+)', '\\1_\\2', paper_title)
            paper_title = re.sub('[^ \w_/]', '', paper_title)

            papers_info.append(PaperInfo(title=row['title'],
                                         clean_title=paper_title.strip(),
                                         abstract_url=str(row['abstract_url']),
                                         pdf_url=str(row['pdf_url']),
                                         conference=conference,
                                         year=year))

        extension = paper_info_file.suffix
        if 'csv' in extension:
            df = pd.read_csv(paper_info_file, sep=';', keep_default_na=False)
        elif 'feather' in extension:
            df = pd.read_feather(paper_info_file)
            df.dropna(inplace=True)


        # each paper contains 3 infos (title, abstract_url, paper_url)
        accents = {
            '[áàãâä]|´a|`a|\~a|\^a': 'a',
            'ç': 'c',
            '[éèêë]|´e|`e|\^e': 'e',
            '[íïì]|´i|`i': 'i',
            'ñ': 'n',
            '[óòôö]|´o|`o|\~o|\^o': 'o',
            '[úùü]|´u|`u': 'u',
        }

        self.papers: List[PaperInfo] = []
        tqdm.pandas(unit='paper', desc='Reading papers info', ncols=TQDM_NCOLS)
        df.progress_apply(_add_paper_info, axis=1, papers_info=self.papers, accents=accents)

        self.n_papers = len(self.papers)
        self.logger.print(f'Found info for {self.n_papers:n} papers')

    def load_words_model(self, model_filename: str) -> None:
        self.model = FastText.load_model(model_filename)
        self.word_dim = self.model.get_dimension()
        self.words = list(self.model.get_words())
        self.logger.print(f'Loaded. Dictionary size: {len(self.words):n}')

    def reduce_paper_vectors_dim(self, new_dim: int, perplexity: int = 5, n_iter: int = 2000) -> None:
        self.logger.print(f'Reducing paper vectors from {self.paper_vectors.shape[1]} to {new_dim} dims')
        with Timer(name=f'Reducing dimensions'):
            tsne = TSNE(perplexity=perplexity, n_components=new_dim, verbose=1,
                        init='pca', n_iter=n_iter, n_jobs=3*multiprocessing.cpu_count()//4)
            self.paper_vectors = tsne.fit_transform(self.paper_vectors)
        self.nearest_neighbours = KDTree(self.paper_vectors)

    def train_words_model(self, corpus_file: Path, n_words: int, model: str = 'skipgram', min_count: int = 5) -> None:
        model_file = self.model_dir / f'fasttext_{model}_{n_words}w.bin'
        self.logger.print(f'Training for {corpus_file} Model={model} Dim={self.word_dim} MinCount={min_count}')

        with Timer(name=f'Training {model} model'):
            self.model = FastText.train_unsupervised(input=str(corpus_file), model=model, dim=self.word_dim,
                                                    minCount=min_count, wordNgrams=self.max_ngram, thread=3*multiprocessing.cpu_count()//4)
        self.model.save_model(str(model_file))
        self.words = list(self.model.get_words())
        self.logger.print(f'Finished. Dictionary size: {len(self.words):n}')
