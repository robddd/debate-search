# %%
import time

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import pickle
import pandas as pd
# %%
# Set n for max number of messages
n = float('inf')
# Set top K for max messages to be returned
top_k = 5
# %%
## Load models
bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# %%
# Load data4
df = pd.read_csv('data/debates_by_message.csv', low_memory=False)
print(df.shape)
df = df[ ~df['message'].isna() ]
print(df.shape)
n = min(n, df.shape[0])
# %%
def pre_embed_messages(n=float('inf')):
    """
    TODO: remove 'n' pre run this and save result
    """
    df = pd.read_csv('data/debates_by_message.csv', low_memory=False)
    df = df[ ~df['message'].isna() ]
    n = min(n, df.shape[0])
    df = df.iloc[:n, :]
    t0 = time.time()
    batch_size = 10000   
    embeddings_batched = {}
    print(f'Embedding {min(n, df.shape[0])} Messages ...')
    for i in range(batch_size, df.shape[0]+ batch_size, batch_size):
        batch_min = i - batch_size
        batch_max = min(df.shape[0], i)
        key = str(batch_min) + '-' + str(batch_max)
        print('Embedding batch', key)
        df_batch = df.iloc[batch_min:batch_max, :].reset_index(drop=True)
        batch_embs = bi_encoder.encode(df_batch['message'], convert_to_tensor=True)
        embeddings_batched[key] = batch_embs
    # parliament_embeddings = bi_encoder.encode(df['message'][:n], convert_to_tensor=True, show_progress_bar=True)
    t = time.time() - t0
    print(f'Embedded {n} messages in {t:.2f} seconds')
    return embeddings_batched

pe_df = pd.read_parquet('parliament_embs.parquet')
parliament_embeddings = torch.tensor(pe_df.values)
# parliament_embeddings = pre_embed_messages()
# %%
pe_keys = ['0-10000', '10000-20000', '20000-30000', '30000-40000', '40000-50000', '50000-60000', '60000-70000', '70000-80000', '80000-90000', '90000-100000', '100000-110000', '110000-116373']
p_embs = parliament_embeddings[pe_keys[0]]
for i, key in enumerate(pe_keys):
    if i > 0:
        p_embs = torch.cat((p_embs, parliament_embeddings[key]))
print(p_embs.shape)
# %%
def filter_by_speaker(speaker, df, parliament_embeddings):
    """
    Args:
        speaker (str): speaker of interest
        df (pd.DataFrame): table of messages that make up debate
        corpus_embeddings (torch.tensor): embedded messages
    
    Return:
        (list): messages filtered by speaker
        (torch.tensor): embedded messages filtered by speaker
    """
    if speaker is not None:
        print(f'Filtering for messages from: {speaker}')
        sub_df = df.iloc[:n, :].loc[ df['speaker'] == speaker ].copy()
        sub_embeddings = parliament_embeddings[sub_df.index, :]
        sub_corpus = sub_df['message'].fillna('').tolist()
    else:
        print('Filtering for messages from anyone')
        sub_df = df.iloc[:n, :].copy()
        sub_embeddings = parliament_embeddings
        sub_corpus = sub_df['message'].fillna('').tolist() 
    return sub_df, sub_embeddings, sub_corpus


test_speaker = 'Rt Hon JACINDA ARDERN'
sub_df, sub_embeddings, sub_corpus = filter_by_speaker(test_speaker, df, parliament_embeddings)
print(sub_df.shape, sub_embeddings.shape, len(sub_corpus))
# %%
def find_closest_messages(query, sub_embeddings, sub_corpus, sub_df, top_k=5):
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, sub_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query
    # score all retrieved passages with the cross_encoder
    cross_inp = [[query, sub_corpus[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    # cross_scores[:5]
    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

    ans = []
    sub_df = sub_df.reset_index(drop=True)
    for idx, hit in enumerate(hits[0:5]):
        ci = hit['corpus_id']
        ans.append(sub_corpus[ci])
        print(f"\nOn {sub_df['date'][ci]}:\n{sub_df['speaker'][ci]}:\n{sub_corpus[ci]}")


test_query = 'house prices for first home buyers'
test_speaker = 'Rt Hon JACINDA ARDERN'
sub_df, sub_embeddings, sub_corpus = filter_by_speaker(test_speaker, df, parliament_embeddings)
print(sub_df.shape, sub_embeddings.shape, len(sub_corpus))
find_closest_messages(test_query, sub_embeddings, sub_corpus, sub_df, top_k=5)
# %%
parliament_embeddings.shape
# %%
pe_df = pd.DataFrame(p_embs.numpy())
pe_df.columns = [str(c) for c in pe_df.columns]
# %%
pe_df.shape
# %%
pe_df.to_csv('temp_5000_embs.csv', index=False, header=False)
# %%
pe_df.to_parquet('parliament_embs.parquet', index=False)

# %%
pe_tt = torch.tensor(pe_df.values)
# %%
pe_pq = 