# import numpy as np

# # A helper function to calculate cosine similarity
# def cos_similarity(x, y, eps=1e-8):
#     nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
#     ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
#     return np.dot(nx, ny)

# def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
#     # ❶ Get the query vector
#     if query not in word_to_id:
#         print('%s is not found' % query)
#         return

#     print('\n[query] ' + query)
#     query_id = word_to_id[query]
#     query_vec = word_matrix[query_id]

#     # ❷ Calculate cosine similarity
#     vocab_size = len(id_to_word)
#     similarity = np.zeros(vocab_size)
#     for i in range(vocab_size):
#         similarity[i] = cos_similarity(word_matrix[i], query_vec)

#     # ❸ Sort and print top results
#     count = 0
#     # argsort() returns indices that would sort the array; 
#     # multiplying by -1 sorts them in descending order.
#     for i in (-1 * similarity).argsort():
#         if id_to_word[i] == query:
#             continue
#         print(' %s: %s' % (id_to_word[i], similarity[i]))
        
#         count += 1
#         if count >= top:
#             return

# # --- Setup for Running ---

# # 1. Define a small vocabulary
# words = ['you', 'say', 'goodbye', 'i', 'say', 'hello', '.']
# word_to_id = {}
# id_to_word = {}
# for i, word in enumerate(set(words)):
#     word_to_id[word] = i
#     id_to_word[i] = word

# # 2. Create a dummy word matrix (Co-occurrence matrix)
# # In a real scenario, these would be vectors representing word meanings.
# vocab_size = len(word_to_id)
# word_matrix = np.random.rand(vocab_size, vocab_size) 

# # 3. Run the function
# most_similar('you', word_to_id, id_to_word, word_matrix)
import numpy as np
import matplotlib.pyplot as plt

def preprocess(text):
    text = text.lower().replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
    return M

# --- Execution ---
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD (Singular Value Decomposition)
U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)
print('Co-occurrence Matrix:\n', C)
print('-'*50)
print('PPMI Matrix:\n', W)
print('-'*50)
print('SVD (Word Vectors in 2D):\n', U[:, :2]) # First two columns for 2D coords