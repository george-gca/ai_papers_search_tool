#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

paper_separator='<#sep#>'

train_paper_finder=1
create_for_app=1
# skip_train_paper_finder=1

n_clusters=150
ngram_threshold=500
ngram_threshold_step=250

# https://github.com/awslabs/autogluon/issues/1020#issuecomment-926089808
export OPENBLAS_NUM_THREADS=15
export GOTO_NUM_THREADS=15
export OMP_NUM_THREADS=15

if [ -n "$train_paper_finder" ]; then
    if [ -n "$create_for_app" ]; then
        # when building for web app, only consider the last 5 years of papers
        echo -e "\nBuilding word representation with fasttext for web app"
        $run_command python train_word_representation.py -l info -c $n_clusters -d 30000 -t $ngram_threshold --ngram_threshold_step $ngram_threshold_step -i --build_dictionary --detect_ngrams --train --min_year $(($(date +'%Y') - 6))
    fi

    echo -e "\nBuilding word representation with fasttext"
    $run_command python train_word_representation.py -l info -c $n_clusters -t $ngram_threshold --ngram_threshold_step $ngram_threshold_step --build_dictionary --detect_ngrams --train

elif [ -n "$skip_train_paper_finder" ]; then
    echo -e "\nBuilding paper vectors only"
    $run_command python train_word_representation.py -l info -c $n_clusters
fi
