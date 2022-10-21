import argparse
import locale
import logging
from pathlib import Path

from paper_finder_trainer import PaperFinderTrainer
from utils import conferences_pdfs, setup_log, supported_conferences


_logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


def main(args):
    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'train_word_model.log'
    setup_log(args, log_file)

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()

    p2v = PaperFinderTrainer(word_dim=args.word_dim, data_dir=data_dir, model_dir=model_dir,
                             max_dictionary_words=args.max_dictionary_words)

    #####################################

    max_ngram = 5

    if not args.skip_training:
        _logger.print('\nStep 1: Removes rare words to build a suitable size of dictionary.')

        corpus_files = [data_dir / c / 'pdfs_clean.csv' for c in conferences_pdfs]
        corpus_files = [c for c in corpus_files if c.exists()]

        for corpus_file in corpus_files:
            p2v.add_dictionary_from_file(corpus_file)

        p2v.build_dictionary()

        # removing 1st common word, since it is UNK
        n_common = 50
        most_common_words = "\n".join(
            f'{k}: {v:n}' for k, v in p2v.count[1:n_common+1])
        _logger.print(f'\nCheck {n_common} most common words:\n{most_common_words}')

        least_common_words = "\n".join(
            f'{k}: {v:n}' for k, v in p2v.count[-n_common:])
        _logger.print(f'\nCheck {n_common} least common words:\n{least_common_words}')

        #####################################

        _logger.print('\nStep 2: Detect n-grams by their appearance frequency. Then re-build a new corpus.')
        for i, n in enumerate(reversed(range(2, max_ngram + 1))):
            p2v.detect_ngrams(n, args.ngram_threshold + i*args.ngram_threshold_step)

        corpus_ngram_file = data_dir / f'corpus_{args.max_dictionary_words}w.txt'
        p2v.create_corpus_with_phrases(corpus_ngram_file)
        p2v.convert_text_with_phrases(data_dir / 'abstracts_clean_pwc.feather', data_dir / f'abstracts_{max_ngram}gram.feather')

        abstract_files = [Path(c) / 'abstracts_clean.csv' for c in supported_conferences]
        abstract_files = [c for c in abstract_files if c.exists()]

        for abstract_file in abstract_files:
            p2v.convert_text_with_phrases(abstract_file, abstract_file.parent / f'abstracts_{max_ngram}gram.csv')

        #####################################

        _logger.print('\nStep 3: Train word representation with fasttext.')
        p2v.train_words_model(corpus_ngram_file, n_words=args.max_dictionary_words, model=args.train_model, min_count=args.min_count)
        p2v.build_similar_dictionary()

    else:  # if args.skip_training
        p2v.load_words_model(str(model_dir / f'fasttext_{args.train_model}_{args.max_dictionary_words}w.bin'))

    words_to_check = [
        'bert',
        'bert_based',
        'capsule',
        'catastrophic_forgetting',
        'continual_learning',
        'dataset',
        'explainability',
        'explanatory_interactive_learning',
        'incremental_learning',
        'interactive_learning',
        'interpretability',
        'large_scale pre_training',
        'model_editing',
        'multimodal_dataset',
        'multimodal_feature',
        'multimodal pre_training',
        'new_dataset',
        'new_multimodal_dataset',
        'question_answering',
        'pre_training',
        'rationale',
        'representation_learning',
        'scene_graph',
        'super_resolution',
        'survey',
        'transformer',
        'visual_dialog',
        'visual_dialog generative',
        'visual_dialog pre_training',
        'visual_dialog new_dataset',
        'visual_entailment',
        'visual_question_answering',
        'visual_question_answering new_dataset',
        'visual_reasoning',
        'vqa',
        ]

    _logger.print('\nChecking result. Finding similar words for:')
    for word in words_to_check:
        most_similar_words = p2v.get_most_similar_words(word, 10)
        most_similar_words = [f'{w}: {v:2.3f}' for v, w in most_similar_words]
        most_similar_words = "\n\t".join(most_similar_words)
        _logger.print(f'\n{word}\n\t{most_similar_words}')

    #####################################

    _logger.print('\nStep 4: Build paper representation vectors with fasttext.')

    p2v.build_paper_vectors(data_dir / f'abstracts_{max_ngram}gram.feather', suffix=f'_pwc')
    p2v.save_paper_vectors(f'_{args.max_dictionary_words}w_{args.word_dim}dims_pwc')

    #####################################

    _logger.print(
        '\nStep 5: Reduce dimensions and then apply k-means clustering.')

    p2v.reduce_paper_vectors_dim(args.paper_dim, perplexity=args.perplexity)
    p2v.save_paper_vectors(f'_{args.max_dictionary_words}w_{args.paper_dim}dims_pwc')

    p2v.clustering_papers(clusters=args.clusters)
    p2v.save_paper_vectors(f'_{args.max_dictionary_words}w_{args.clusters}_clusters_pwc')

    not_informative_words = [
        'data',
        'learning',
        'method',
        'model',
        'network',
        'problem',
        'result',
        'task',
        'training'
    ]
    n_keywords = 15
    for i in range(args.clusters):
        cluster_keywords = p2v.cluster_abstract_freq[i]
        cluster_keywords = [
            p2v.abstract_words[w] for w, _ in cluster_keywords if w not in not_informative_words][:n_keywords]
        _logger.print(f'cluster {i+1:02d} keywords: {", ".join(cluster_keywords)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory for the input data')
    parser.add_argument('--model_dir', type=str, default='model_data',
                        help='directory for saving the model')

    # args for building corpus
    parser.add_argument('-d', '--max_dictionary_words', type=int,
                        default=50_000, help='max words to save in dictionary')
    parser.add_argument('-t', '--ngram_threshold', type=int, default=1_000,
                        help='minimum number of occurrences of n-gram to consider as a new term')
    parser.add_argument('--ngram_threshold_step', type=int, default=0,
                        help='increase threshold as ngram size decreases')

    # args for training / clustering
    parser.add_argument('-m', '--train_model', type=str, default='skipgram',
                        choices=('skipgram', 'cbow'),
                        help='model for training word representation')
    parser.add_argument('-w', '--word_dim', type=int, default=150,
                        help='dimensions for word representation')
    parser.add_argument('-i', '--min_count', type=int, default=10,
                        help='minimal number of word occurences')
    parser.add_argument('-p', '--paper_dim', type=int, default=3,
                        help='dimensions for paper representation')
    parser.add_argument('-x', '--perplexity', type=int,
                        default=25, help='perplexity param for t-SNE')
    parser.add_argument('-c', '--clusters', type=int, default=26,
                        help='number of clusters to be divided')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-s', '--skip_training', action='store_true',
                        help='skip training and load trained model')

    args = parser.parse_args()

    main(args)
