import networkx as nx
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from networkx.algorithms.link_prediction import (preferential_attachment, jaccard_coefficient, adamic_adar_index, resource_allocation_index)
from sklearn.metrics import roc_auc_score, average_precision_score

# Read csv & prep data
df = pd.read_csv("Final/Ratings.csv")
df = df.dropna()

books = sorted(df["Title"].unique().tolist())
users = sorted(df["User_id"].unique().tolist())
edges = df[["Title","User_id","review/score"]].values.tolist()

G = nx.Graph()
G.add_nodes_from(books, bipartite=0)
G.add_nodes_from(users, bipartite=1)
G.add_weighted_edges_from(edges)

# Function to compute Katz centrality
def katz_centrality_fn(G, test_edges, alpha=0.005):
    katz_centrality = nx.katz_centrality(G, alpha=alpha)
    scores = []
    for u, b in test_edges:
        # For bipartite graphs, compute the Katz centrality of the user and book
        user_score = katz_centrality.get(u, 0)  # Default to 0 if user has no Katz centrality value
        book_score = katz_centrality.get(b, 0)  # Default to 0 if book has no Katz centrality value
        scores.append((u, b, user_score + book_score))  # Add the scores of the user and book
    return scores

def evaluate_link_prediction(
    G, users, books, edges, heuristic_fn, test_size=0.2, seed=25, verbose=True
):

    # Train/test split
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)

    # Build training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(users, bipartite=0)
    G_train.add_nodes_from(books, bipartite=1)
    G_train.add_edges_from(train_edges)

    # Positive test examples
    test_pos = set(test_edges)

    # Negative sampling
    all_edges = set(edges)
    test_neg = set()
    random.seed(seed)
    while len(test_neg) < len(test_pos):
        u = random.choice(users)
        b = random.choice(books)
        if (u, b) not in all_edges and (u, b) not in test_neg:
            test_neg.add((u, b))

    # Combine for testing
    test_all = list(test_pos) + list(test_neg)

    # Compute heuristic scores
    scores = list(heuristic_fn(G_train, test_all))

    # Format results
    score_name = heuristic_fn.__name__
    pred_df = pd.DataFrame(scores, columns=["user", "book", score_name])
    pred_df["label"] = [1] * len(test_pos) + [0] * len(test_neg)

    # Evaluate
    auc = roc_auc_score(pred_df["label"], pred_df[score_name])
    ap = average_precision_score(pred_df["label"], pred_df[score_name])

    if verbose:
        print(f"Evaluation using {score_name}:")
        print(f"  AUC: {auc:.4f}")
        print(f"  AP:  {ap:.4f}")

    return pred_df, auc, ap

def compare_methods(G, users, books, edges, heuristics, test_size=0.2, seed=25):
    results = []

    for heuristic in heuristics:
        pred_df, auc, ap = evaluate_link_prediction(
            G, users, books, edges, heuristic, test_size=test_size, seed=seed, verbose=False
        )
        results.append({
            "Method": heuristic.__name__,
            "AUC": round(auc, 4),
            "Average Precision": round(ap, 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False).reset_index(drop=True)
    print(results_df)
    return results_df

edges_simple = [(user, book) for book, user, _ in edges]

# List of heuristics to test, including Katz centrality
heuristics = [
    preferential_attachment,
    jaccard_coefficient,
    adamic_adar_index,
    resource_allocation_index,
    katz_centrality_fn  # Add Katz centrality as a heuristic
]

# Run comparison
results_df = compare_methods(G, users, books, edges_simple, heuristics)
