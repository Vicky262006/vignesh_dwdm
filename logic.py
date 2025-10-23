import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient

def load_orders_from_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["grocerydb"]
    collection = db["orders"]
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    return df

def get_missed_products(customer_name, current_order):
    df = load_orders_from_mongo()
    
    product_cols = ['product1','product2','product3']
    df['all_products'] = df[product_cols].apply(lambda row: [p for p in row if pd.notna(p)], axis=1)
    
    all_products = sorted(list(set([p for sublist in df['all_products'] for p in sublist])))
    model_df = pd.DataFrame({p:[1 if p in row else 0 for row in df['all_products']] for p in all_products})
    model_df['customer_name'] = df['customer_name']
    
    # Aggregate per customer
    customer_history = model_df.groupby('customer_name').max()
    
    # KNN model
    X = customer_history.drop(columns=['customer_name'], errors='ignore')
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X)
    
    if customer_name not in customer_history.index:
        return []

    customer_vec = X.loc[[customer_name]]
    distances, indices = knn.kneighbors(customer_vec)
    neighbors = X.iloc[indices[0]]
    recommended = neighbors.sum(axis=0)
    recommended_products = recommended[recommended >= 1].index.tolist()
    
    missed = [p for p in recommended_products if p not in current_order]
    return missed
