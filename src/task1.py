import csv
import gensim.downloader as api
from gensim.models import KeyedVectors

'''
Note: If using a machine where gensim.downloader.load does not work, you have to use KeyedVectors.load_word2vec_format.
In this case, you need to manually download the model using this URL:https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
'''

# Function to read the dataset from a txt file and store it in a dictionary
def read_dataset(file_path):
    dataset = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process lines in pairs (question and choices)
    for i in range(0, len(lines), 6):
        question_number = int(lines[i].split('.')[0])
        question_word = lines[i].split('.')[1].strip()

        choices = [line.strip()[3:] for line in lines[i + 1:i + 5]]  # Remove prefix and '\t'
        correct_choice = ord(lines[i + 5].strip()) - ord('a')  # Convert the last entry to an index integer

        dataset[question_word] = choices + [correct_choice]

    return dataset

# Function to find the closest synonym using Word2Vec similarity
def find_closest_synonym(question_word, choices, model):
    # Ensure the question word is in the Word2Vec model's vocabulary
    if question_word in model.key_to_index:
        # Calculate the similarity between the question word and each choice
        similarity_scores = [model.similarity(question_word, choice) for choice in choices[:-1]]
        
        # Find the index of the choice with the highest similarity
        closest_synonym_index = similarity_scores.index(max(similarity_scores))
        
        # Return the closest synonym
        return closest_synonym_index
    else:
        print(f"Question word {question_word} is not in vocabulary.")
        return -1

# Specify the path to your dataset txt file
file_path = './dataset/synonym.txt'

# Read the dataset from the file
dataset = read_dataset(file_path)

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
# model = api.load('word2vec-google-news-300') # This only works on gensim installed on Windows
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

# Analysis details
vocab_size = len(model.key_to_index)
correct_labels = 0
questions_without_guessing = 0

# Output file details
details_file_path = 'word2vec-google-news-300-details.csv'
# Output file analysis
analysis_file_path = 'analysis.csv'

with open(details_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop through the entire dataset and find the closest synonym for each question word
    for question_word, choices in dataset.items():
        closest_synonym_index = find_closest_synonym(question_word, choices, model)
        
        # Determine if the system's guess was guess/correct/wrong
        if closest_synonym_index == -1 or question_word == choices[closest_synonym_index]:
            label = 'guess'
        elif closest_synonym_index == choices[-1]:
            label = 'correct'
            correct_labels += 1
            questions_without_guessing += 1
        else:
            label = 'wrong'
            questions_without_guessing += 1

        # Write row to CSV file
        writer.writerow([question_word, choices[choices[-1]], choices[closest_synonym_index], label])
        
# Calculate accuracy
accuracy = correct_labels / questions_without_guessing if questions_without_guessing > 0 else 0

# Write analysis details to CSV file
with open(analysis_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write analysis details
    writer.writerow([f'word2vec-google-news-300', vocab_size, correct_labels, questions_without_guessing, accuracy])