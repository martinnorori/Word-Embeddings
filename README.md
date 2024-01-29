# WordEmbeddings

## Synonym Test Solver

This project automates synonym testing using multiple pre-trained and custom word embedding models. For the custom models, It trains the system on provided text data acquired from books available on Project Gutenberg. All of the models' performances in identifying synonym words are analyzed and compared.

## Files Organization
|-- books/
||--book1.txt
|-- dataset/
||--synonym.txt
|-- task1.py
|-- task2.py
|-- task3.py

## Prerequisites

- Python 3.6+
- NLTK, Gensim, Matplotlib
- Modules: Numpy, Scipy, CSV, Pandas

Install dependencies with:
pip install nltk gensim matplotlib

Place synonym.txt and text data, so the different books (e.g., book1.txt, book2.txt, ..., book51.txt) in the project directory.

The synonym file that was provided for the course has to be in the .txt format. This is the format that was used for task 3 and the desired one for this code.

You can then run the code.

## Run

To run Task 1, execute the following command in the same directory as the file:
```bash
python3 task1.py
```

To run Task 2, execute the following command in the same directory as the file:
```bash
python3 task2.py
```

To run Task 3, execute the following command in the same directory as the file:
```bash
python3 task3.py
```

## Customization

Modify window_sizes and embedding_sizes in the script to experiment with different model configurations.

## Output

analysis.csv: Model performance data
model_performance_comparison.png: Model accuracy chart
model-details.csv: Each model's details
