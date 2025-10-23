import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient

def load_orders_from_mongo():
    """Load grocery orders dataset from MongoDB into a Pandas DataFrame"""
    client = MongoClient("mongodb://localhost:27017/")  # change if using cloud MongoDB
    db = client["grocerydb"]
    collection = db["orders"]

    # Fetch all documents, excluding the default MongoDB _id
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    return df

def get_missed_product_recommendations_ai(customer_name, current_order):
    """
    Recommends products a customer has bought previously but is missing from
    their current order, using a K-Nearest Neighbors AI model.
    """
    try:
        df = load_orders_from_mongo()
    except Exception as e:
        return f"Error: Could not load data from MongoDB. Details: {e}"

    # Step 1: Data Preparation
    product_columns = ['product1', 'product2', 'product3']
    df['all_products'] = df[product_columns].apply(
        lambda row: [item for item in row if pd.notna(item)], axis=1
    )

    all_products = sorted(list(set([prod for sublist in df['all_products'] for prod in sublist])))

    model_df = pd.DataFrame(
        {product: [1 if product in row else 0 for row in df['all_products']] for product in all_products}
    )
    model_df['customer_name'] = df['customer_name']

    customer_purchase_history = model_df.groupby('customer_name').max()

    # Step 2: Train KNN Model
    X = customer_purchase_history.drop(columns=['customer_name'], errors='ignore')
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    model.fit(X)

    # Step 3: Get Recommendations
    if customer_name not in customer_purchase_history.index:
        return f"No previous order history found for customer: {customer_name}"

    customer_vector = X.loc[[customer_name]]
    distances, indices = model.kneighbors(customer_vector)
    neighbor_indices = indices[0]
    neighbor_data = X.iloc[neighbor_indices]
    neighbor_purchases = neighbor_data.sum(axis=0)

    threshold = 1
    recommended_products = neighbor_purchases[neighbor_purchases >= threshold].index.tolist()

    current_order_set = set(current_order)
    missed_products = [prod for prod in recommended_products if prod not in current_order_set]

    if not missed_products:
        return "No missed products to recommend. You have all your usual items!"
    else:
        return f"Based on your habits and similar customers, you may have missed: {', '.join(missed_products)}"

# Example Usage:
customer_to_check = 'Hussain'
new_order = ['Fish', 'Salt', 'Butter']
recommendations = get_missed_product_recommendations_ai(customer_to_check, new_order)
print(recommendations)
