from flask import Flask, render_template, request, jsonify 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Function to normalize Yes/No values
def normalize_yes_no(value):
    if isinstance(value, str) and value.strip().lower() == 'yes':
        return 'Yes'
    return 'No'

# Load Excel files
df1 = pd.read_excel('assessments.xlsx')   # Contains 'Assessment', 'URL'
df2 = pd.read_excel('metadata.xlsx')      # Contains metadata like Remote Testing, Duration, etc.

# Clean column names (remove extra spaces/newlines)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Rename columns if needed
df2.rename(columns={
    'Remote Testing': 'Remote Testing Support (Yes/No)',
    'Adaptive Support': 'Adaptive/IRT Support (Yes/No)'
}, inplace=True)

# Combine data side-by-side
combined_df = pd.concat([df1, df2], axis=1)
combined_df.columns = combined_df.columns.str.strip()

# Normalize Remote Testing and Adaptive Support
if 'Remote Testing Support (Yes/No)' in combined_df.columns:
    combined_df['Remote Testing Support (Yes/No)'] = combined_df['Remote Testing Support (Yes/No)'].apply(normalize_yes_no)

if 'Adaptive/IRT Support (Yes/No)' in combined_df.columns:
    combined_df['Adaptive/IRT Support (Yes/No)'] = combined_df['Adaptive/IRT Support (Yes/No)'].apply(normalize_yes_no)

# Fill missing values
combined_df['Assessments'] = combined_df['Assessments'].fillna('')
combined_df['URL'] = combined_df['URL'].fillna('')

# Create combined text for similarity matching
combined_df['combined'] = combined_df['Assessments'] + ' ' + combined_df['URL']

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(combined_df['combined'])

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        query = request.form['query']
        query_tfidf = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
        recommendations_idx = cosine_similarities[0].argsort()[-5:][::-1]

        for i in recommendations_idx:
            row = combined_df.iloc[i]
            recommendations.append({
                'assessment': row.get('Assessments', 'N/A'),
                'url': row.get('URL', '#'),
                'remote_testing': row.get('Remote Testing Support (Yes/No)', 'N/A'),
                'adaptive_support': row.get('Adaptive/IRT Support (Yes/No)', 'N/A'),
                'duration': row.get('Duration', 'N/A'),
                'test_type': row.get('Test Type', 'N/A')
            })

    return render_template('index.html', recommendations=recommendations)





if __name__ == "__main__":
    app.run(debug=True)






























# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# # Load Excel files
# df1 = pd.read_excel('assessments.xlsx')   # Contains 'Assessment', 'URL'
# df2 = pd.read_excel('metadata.xlsx')      # Contains metadata like duration, type, etc.

# # Combine the two DataFrames side-by-side (by row order)
# combined_df = pd.concat([df1, df2], axis=1)

# # Fill missing values
# combined_df['Assessments'] = combined_df['Assessments'].fillna('')
# combined_df['URL'] = combined_df['URL'].fillna('')

# # Combine columns for vectorization
# combined_df['combined'] = combined_df['Assessments'] + ' ' + combined_df['URL']

# # Vectorize using TF-IDF
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(combined_df['combined'])

# @app.route("/", methods=["GET", "POST"])
# def index():
#     recommendations = []

#     if request.method == "POST":
#         query = request.form['query']
#         query_tfidf = vectorizer.transform([query])
#         cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
#         recommendations_idx = cosine_similarities[0].argsort()[-5:][::-1]

#         for i in recommendations_idx:
#             row = combined_df.iloc[i]
#             recommendations.append({
#                 'assessment': row.get('Assessments', 'N/A'),
#                 'url': row.get('URL', '#'),
#                 'remote_testing': row.get('Remote Testing Support (Yes/No)', 'N/A'),
#                 'adaptive_support': row.get('Adaptive/IRT Support (Yes/No)', 'N/A'),
#                 'duration': row.get('Duration', 'N/A'),
#                 'test_type': row.get('Test Type', 'N/A')
#             })

#     return render_template('index.html', recommendations=recommendations)

# if __name__ == "__main__":
#     app.run(debug=True)























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
