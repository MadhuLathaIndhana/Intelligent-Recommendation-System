import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the data from Excel
df = pd.read_excel("Traning_data.xlsx")

# Step 2: Combine 'Assessments' and 'URL' to create a feature vector
df['Assessments'] = df['Assessments'].fillna('')
df['URL'] = df['URL'].fillna('')
df['features'] = df['Assessments'] + ' ' + df['URL']

# Step 3: Use TfidfVectorizer to convert text data into vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Step 4: Compute cosine similarity between the rows
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Function to recommend similar assessments based on the query
def recommend_assessments(query, cosine_sim=cosine_sim):
    try:
        idx = df[df['Query'] == query].index[0]
    except IndexError:
        return "Query not found in the data. Please try another one."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return df['Assessments'].iloc[movie_indices]

# Step 6: Function to recommend URLs based on the query
def recommend_urls(query, cosine_sim=cosine_sim):
    try:
        idx = df[df['Query'] == query].index[0]
    except IndexError:
        return "Query not found in the data. Please try another one."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return df['URL'].iloc[movie_indices]

# Route to render the homepage and accept user input
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    urls = []
    if request.method == "POST":
        query = request.form["query"]
        recommended_assessments = recommend_assessments(query)
        recommended_urls = recommend_urls(query)
        recommendations = recommended_assessments.tolist()
        urls = recommended_urls.tolist()
    
    return render_template("index.html", recommendations=recommendations, urls=urls)

if __name__ == "__main__":
    app.run(debug=True)
































# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Dummy assessment data
# assessment_data = {
#     "Math": "Math Assessment: Solve algebraic equations and calculus problems.",
#     "Science": "Science Assessment: Answer questions on biology, chemistry, and physics.",
#     "English": "English Assessment: Read and answer questions on grammar, literature, and comprehension.",
# }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     category = request.form['category']
#     assessment = assessment_data.get(category, "Sorry, we don't have assessments for this category.")
#     return render_template('recommend.html', category=category, assessment=assessment)

# if __name__ == "__main__":
#     app.run(debug=True)
