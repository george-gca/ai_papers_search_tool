# AI Papers Search Tool

Automatic paper clustering and search tool by [fastText from Facebook Research](https://fasttext.cc/).

Based on [CVPR_paper_search_tool by Jin Yamanaka](https://github.com/jiny2001/CVPR_paper_search_tool). I decided to split the code into multiple projects:

- [AI Papers Scrapper](https://github.com/george-gca/ai_papers_scrapper) - Download papers pdfs and other information from main AI conferences
- [AI Papers Cleaner](https://github.com/george-gca/ai_papers_cleaner) - Extract text from papers PDFs and abstracts, and remove uninformative words
- this project - Automatic paper clustering
- [AI Papers Searcher](https://github.com/george-gca/ai_papers_searcher) - Web app to search papers by keywords or similar papers
- [AI Conferences Info](https://github.com/george-gca/ai_conferences_info) - Contains the titles, abstracts, urls, and authors names extracted from the papers

I also added support for more conferences in a single web app, customized it a little further, and hosted it on [PythonAnywhere](https://www.pythonanywhere.com/). You can see a running example of the web app [here](https://georgegca.pythonanywhere.com/).

## Requirements

[Docker](https://www.docker.com/) or, for local installation:

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/)

> Note: Poetry installation currently not working due to [a bug when installing fasttext](https://github.com/facebookresearch/fastText/pull/1292).

## Usage

To make it easier to run the code, with or without Docker, I created a few helpers. Both ways use `start_here.sh` as an entry point. Since there are a few quirks when calling the specific code, I created this file with all the necessary commands to run the code. All you need to do is to uncomment the relevant lines and run the script:

```bash
train_paper_finder=1
create_for_app=1
# skip_train_paper_finder=1
```

### Running without Docker

You first need to install [Python Poetry](https://python-poetry.org/docs/). Then, you can install the dependencies and run the code:

```bash
poetry install
bash start_here.sh
```

### Running with Docker

To help with the Docker setup, I created a `Dockerfile` and a `Makefile`. The `Dockerfile` contains all the instructions to create the Docker image. The `Makefile` contains the commands to build the image, run the container, and run the code inside the container. To build the image, simply run:

```bash
make
```

To call `start_here.sh` inside the container, run:

```bash
make run
```
