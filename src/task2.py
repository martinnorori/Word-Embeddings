import csv
import gensim.downloader as api
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt


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

def find_closest_synonym(question_word, choices, model):
    valid_choices = [choice for choice in choices if choice in model.key_to_index]
    
    if question_word in model.key_to_index and valid_choices:
        similarity_scores = [model.similarity(question_word, choice) for choice in valid_choices]
    

        closest_synonym_index = similarity_scores.index(max(similarity_scores))
        closest_synonym_word = valid_choices[closest_synonym_index]

        return closest_synonym_word

        
    else:
        return None

# Specify the path to your dataset txt file
file_path = './dataset/synonym.txt'

# Read the dataset from the file
dataset = read_dataset(file_path)


model_list = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-100", "glove-twitter-25"]

model_names = []
accuracies = []

for m in model_list:


    model = api.load(m) # This only works on gensim installed on Windows

    # Analysis details
    vocab_size = len(model.key_to_index)
    correct_labels = 0
    questions_without_guessing = 0

    # Output file details
    details_file_path = f"{m}-details.csv"
    # Output file analysis
    analysis_file_path = 'analysis.csv'

    with open(details_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Loop through the entire dataset and find the closest synonym for each question word
        for question_word, choices in dataset.items():
            closest_synonym_word = find_closest_synonym(question_word, choices[:-1], model)
        

            # Determine if the system's guess was guess/correct/wrong
            if closest_synonym_word is None:
                label = 'guess'
            elif question_word == closest_synonym_word:
                label = 'guess'
            elif closest_synonym_word == choices[choices[-1]]:
                label = 'correct'
                correct_labels += 1
                questions_without_guessing += 1
            else:
                label = 'wrong'
                questions_without_guessing += 1

            # Write row to CSV file
            writer.writerow([question_word, choices[choices[-1]], closest_synonym_word, label])
            
    # Calculate accuracy
    accuracy = correct_labels / questions_without_guessing if questions_without_guessing > 0 else 0

    model_names.append(m)
    accuracies.append(accuracy)

    with open(analysis_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write analysis details
        writer.writerow([f"{m}", vocab_size, correct_labels, questions_without_guessing, accuracy])

plt.figure(figsize=(10, 6))
plt.plot(model_names, accuracies, marker='o', color='orange', linestyle='-', linewidth=2, markersize=8)
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1) 
plt.xticks(rotation=45)  
plt.tight_layout()
plt.savefig('model_accuracies.png') 
plt.show()  
