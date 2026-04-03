from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["Jakarta Bekasi Jakarta", "Bekasi Bekasi Jakarta"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

similarity_scores = cosine_similarity(count_matrix)

print(similarity_scores)
