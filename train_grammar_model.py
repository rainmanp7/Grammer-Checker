import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle  # Add this import

# Step 1: Create a dataset.
dataset = [
    # Greetings and Salutations (Correct)
    ("Hello, how are you?", 1),
    ("Good morning, everyone.", 1),
    ("Hi, what's up?", 1),
    ("Good afternoon, sir.", 1),
    ("Hey there!", 1),
    ("How's it going?", 1),
    ("Nice to meet you.", 1),
    ("Greetings, my friend.", 1),
    ("What's new?", 1),
    ("How do you do?", 1),
    ("Good evening, ma'am.", 1),
    ("Hi, how have you been?", 1),
    ("It's great to see you.", 1),
    ("Hello, it's been a while.", 1),
    ("Good morning, how's your day?", 1),
    ("Nice seeing you again.", 1),
    ("What’s going on?", 1),
    ("Pleasure meeting you.", 1),
    ("Hey, long time no see.", 1),
    ("Good day, my friend.", 1),
    
    # Greetings and Salutations (Incorrect)
    ("Hello how is?", 0),
    ("Gud morning evryone.", 0),
    ("Hi what is up?", 0),
    ("Good afternoons sir.", 0),
    ("Hey their!", 0),
    ("How it going?", 0),
    ("Nice too meet you.", 0),
    ("Greetings my frend.", 0),
    ("What new?", 0),
    ("How does you do?", 0),
    ("Helo, how is been?", 0),
    ("Its great seeing you.", 0),
    ("Good morning, hows day?", 0),
    ("Nice see you again.", 0),
    ("What goin on?", 0),
    ("Pleasure met you.", 0),
    ("Hey, long time no seee.", 0),
    ("Good days my friends.", 0),
    
    # Farewells (Correct)
    ("Goodbye, see you later.", 1),
    ("Take care!", 1),
    ("Have a nice day!", 1),
    ("See you soon.", 1),
    ("Farewell, my friend.", 1),
    ("It was nice talking to you.", 1),
    ("Catch you later.", 1),
    ("Talk to you later.", 1),
    ("See you tomorrow.", 1),
    ("Have a great night.", 1),
    ("See you next week.", 1),
    ("Take it easy!", 1),
    ("Until next time.", 1),
    ("Talk soon!", 1),
    ("Good night, sleep well.", 1),
    ("Goodbye, and take care.", 1),
    ("I’ll see you around.", 1),
    ("It’s been a pleasure.", 1),
    ("Drive safe!", 1),
    ("See you in the morning.", 1),
    
    # Farewells (Incorrect)
    ("Goodby see later.", 0),
    ("Take car!", 0),
    ("Have nice day!", 0),
    ("See soon.", 0),
    ("Farwell my freind.", 0),
    ("It nice talking you.", 0),
    ("Catch later.", 0),
    ("Talk you later.", 0),
    ("See tomorrow.", 0),
    ("Have great night.", 0),
    ("See yous next week.", 0),
    ("Take its easy!", 0),
    ("Until nxt time.", 0),
    ("Talks soon!", 0),
    ("Good night, sleep wel.", 0),
    ("Goodby, and take cares.", 0),
    ("I’ll sees you around.", 0),
    ("It been a pleasure.", 0),
    ("Drive saf!", 0),
    ("Se you in morning.", 0),
    
    # Common Nouns and Adjectives (Correct)
    ("The sky is blue.", 1),
    ("I have a red car.", 1),
    ("The children are happy.", 1),
    ("The park is full of people.", 1),
    ("I bought a new phone.", 1),
    ("She owns a small business.", 1),
    ("The flowers are colorful.", 1),
    ("My dog is very friendly.", 1),
    ("This house is beautiful.", 1),
    ("We live in a big city.", 1),
    ("The cat is sleeping peacefully.", 1),
    ("The mountains are majestic.", 1),
    ("I love sunny days.", 1),
    ("Her dress is elegant.", 1),
    ("Our garden has green grass.", 1),
    ("This book is interesting.", 1),
    ("The lake is crystal clear.", 1),
    ("He is holding a shiny coin.", 1),
    ("The forest is quiet and serene.", 1),
    ("Their car is fast and reliable.", 1),
    
    # Common Nouns and Adjectives (Incorrect)
    ("The sky blue.", 0),
    ("I has a red cars.", 0),
    ("The childrens happy.", 0),
    ("The parks full of people.", 0),
    ("I buys new phone.", 0),
    ("She own a small bussiness.", 0),
    ("The flower are colorful.", 0),
    ("My dogs very friend.", 0),
    ("This houses is beautifull.", 0),
    ("We lives in big cities.", 0),
    ("The cats sleeping peaceful.", 0),
    ("The mountain is majestics.", 0),
    ("I loving sunny days.", 0),
    ("Her dress elegant.", 0),
    ("Our gardens has green grasss.", 0),
    ("This books is interestings.", 0),
    ("The lakes crystal clear.", 0),
    ("He holding shiny coins.", 0),
    ("The forests quiet serene.", 0),
    ("Their cars fast reliable.", 0),
    
    # Everyday Sentences (Correct)
    ("I am going to the store.", 1),
    ("She is cooking dinner.", 1),
    ("He likes to play guitar.", 1),
    ("We are watching a movie.", 1),
    ("They went to the park.", 1),
    ("The teacher is giving a lecture.", 1),
    ("The kids are having fun.", 1),
    ("I need some water.", 1),
    ("She writes in her journal.", 1),
    ("They enjoy playing soccer.", 1),
    ("We are cleaning the house.", 1),
    ("He wants to visit his parents.", 1),
    ("I love listening to music.", 1),
    ("She is reading a novel.", 1),
    ("He is learning to code.", 1),
    ("We like hiking in the mountains.", 1),
    ("They are taking a road trip.", 1),
    ("She paints beautiful landscapes.", 1),
    ("I often drink coffee in the morning.", 1),
    ("The team is practicing for the game.", 1),
]

# Step 2: Prepare the Dataset
sentences, labels = zip(*dataset)

# Convert sentences to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences).toarray()
y = torch.tensor(labels, dtype=torch.float32)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the Neural Network
class GrammarModel(nn.Module):
    def __init__(self, input_size):
        super(GrammarModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Step 4: Initialize the Model
input_size = X_train.shape[1]
model = GrammarModel(input_size)

# Step 5: Define Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 6: Train the Model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 7: Save the Trained Model and Vectorizer
torch.save(model.state_dict(), "grammar_model.pth")
with open("vectorizer.pkl", "wb") as f:  # Save the vectorizer
    pickle.dump(vectorizer, f)
print("Training complete. Model saved to 'grammar_model.pth' and vectorizer to 'vectorizer.pkl'.")