import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import csv
import pandas as pd
import matplotlib.pyplot as plt


# Download necessary NLTK resources. 
nltk.download('punkt')  # This is a tokenizer model used by NLTK to split text into a list of sentences.
nltk.download('stopwords') # This is a set of common words like "the", "is", "in", which are often filtered out during text processing

# Load the set of stopwords
stop_words = set(stopwords.words('english'))

# List of books
books = [
    'book1.txt', 'book2.txt', 'book3.txt', 'book4.txt', 'book5.txt', 'book6.txt', 'book7.txt', 'book8.txt',
    'book9.txt', 'book10.txt', 'book11.txt', 'book12.txt', 'book13.txt', 'book14.txt', 'book15.txt',
    'book16.txt', 'book17.txt', 'book18.txt', 'book19.txt', 'book20.txt', 'book21.txt', 'book22.txt',
    'book23.txt', 'book24.txt', 'book25.txt', 'book26.txt', 'book27.txt', 'book28.txt', 'book29.txt',
    'book30.txt', 'book31.txt', 'book32.txt', 'book33.txt', 'book34.txt', 'book35.txt', 'book36.txt',
    'book37.txt', 'book38.txt', 'book39.txt', 'book40.txt', 'book41.txt', 'book42.txt', 'book43.txt',
    'book44.txt', 'book45.txt', 'book46.txt', 'book47.txt', 'book48.txt', 'book49.txt', 'book50.txt',
    'book51.txt'
]



# Dictionary to hold the preprocessed sentences of each book
preprocessed_books = {}

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    
    # Tokenize each sentence into words, remove non-alphabetic characters, convert to lowercase
    sentences = [[word.lower() for word in word_tokenize(sentence) if word.isalpha()] for sentence in sentences]


    # Remove stopwords
    sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]


    return sentences

for book in books:
    path = './books/' + book
    with open(path, 'r', encoding='utf-8') as file:
        book_text = file.read()
        preprocessed_books[book] = preprocess_text(book_text)
        

# Flattens the dictionary of lists into a single list containing all the sentences from all the books.
all_sentences = [sentence for book_sentences in preprocessed_books.values() for sentence in book_sentences]


# Define different values for window sizes and embedding sizes
window_sizes = [10, 20]  # Example window sizes
embedding_sizes = [100, 300]  # Example embedding sizes

# Train a model for each combination of window size and embedding size
epochs = 10  # Number of training iterations
for window in window_sizes:
    for size in embedding_sizes:
        model = Word2Vec(sentences=all_sentences, vector_size=size, window=window, min_count=1, workers=4)
        model.train(all_sentences, total_examples=len(all_sentences), epochs=epochs)  # Train model
        model_name = f"gutenberg_w{window}_size{size}.model"
        model.save(model_name)
        


# Function to evaluate the model and return details
def evaluate_model(model, test_data):
    correct_labels = 0
    questions_without_guess = 0
    model_details = []

    for word, actual_synonym in test_data:
        if word in model.wv:
            predicted_synonyms = model.wv.most_similar(word, topn=3)  # Define predicted_synonyms
            predicted_synonym = predicted_synonyms[0][0]
            is_correct = predicted_synonym == actual_synonym
            model_details.append([word, actual_synonym, predicted_synonym, 'correct' if is_correct else 'incorrect'])
            correct_labels += int(is_correct)
            questions_without_guess += 1
            
        else:
            model_details.append([word, actual_synonym, 'N/A', 'guess'])
            

    accuracy = correct_labels / questions_without_guess if questions_without_guess > 0 else 0
    return model_details, correct_labels, questions_without_guess, accuracy





# Parse the Synonym Test data from the text file
test_data = []
with open('./dataset/synonym.txt', 'r') as file:
    lines = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip whitespace
    i = 0  # Initialize a counter to keep track of the line number
    while i < len(lines):
        # Extract the question word and the correct answer letter
        question_part = lines[i].split()  # Split by whitespace
        question = question_part[1]  # The word is the second element
        correct_option_letter = lines[i+5]  # The correct answer letter is 5 lines down from the question

        # Extract the correct answer based on the letter
        if correct_option_letter in 'abcd':  # Check if the letter is valid
            answer_index = 'abcd'.index(correct_option_letter)  # Get the index of the correct letter
            answer = lines[i + 1 + answer_index].split()[1]  # The answer is on the line after the options start, offset by the answer index
            test_data.append((question, answer))
        else:
            print(f"Invalid correct answer letter found on line {i+6}: {correct_option_letter}")

        i += 6  # Move to the next question block, which starts 6 lines down




# Iterate through each trained model and evaluate it
results = []
first_model = True  # Flag to check if it's the first model

for window in window_sizes:
    for size in embedding_sizes:
        model_file_name = f"gutenberg_w{window}_size{size}"  # Correctly set up model file name
        model = Word2Vec.load(f"{model_file_name}.model")  # Load the model

        # Evaluate the model
        model_details, correct_labels, questions_without_guess, accuracy = evaluate_model(model, test_data)

        
        
        # Write the details to a csv file for each model
        details_file_name = f"{model_file_name}-details.csv"  # Correctly set up adetails file name
        with open(details_file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Word', 'Actual Synonym', 'Predicted Synonym', 'Correct/Incorrect/Guess'])
            for detail in model_details:
                writer.writerow(detail)

        # Append the results to the analysis.csv file
        with open('analysis.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([model_file_name, len(model.wv), correct_labels, questions_without_guess, accuracy])

# Create a list to store the performance data of each model
model_performance = []

for window in window_sizes:
    for size in embedding_sizes:
        model_file_name = f"gutenberg_w{window}_size{size}"
        model = Word2Vec.load(f"{model_file_name}.model")

        # Evaluate the model
        model_details, correct_labels, questions_without_guess, accuracy = evaluate_model(model, test_data)

        # Store the performance data
        performance_data = {
            "Model Name": model_file_name,
            "Vocabulary Size": len(model.wv),
            "Correct Labels": correct_labels,
            "Questions Answered Without Guessing": questions_without_guess,
            "Accuracy": accuracy
        }
        model_performance.append(performance_data)

# Convert the performance data to a DataFrame
performance_df = pd.DataFrame(model_performance)

# Set up the plotting environment
plt.figure(figsize=(15, 8))  # Adjust the figure size as necessary

# Plotting accuracy for each model
plt.bar(performance_df['Model Name'], performance_df['Accuracy'], color='grey')

# Adding titles and labels
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')

# Rotate and align the x-axis labels for better readability
plt.xticks(rotation=0, ha='center')  # Adjust 'ha' as 'center' to align labels in the center

# Automatically adjust subplot params for the plot to fit into the figure area
plt.tight_layout()

# Save the figure
plt.savefig('model_performance_comparison.png', bbox_inches='tight')  # Use bbox_inches='tight' to fit everything within the plot area

# Show the plot
plt.show()
