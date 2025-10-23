import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

def get_missed_product_recommendations_ai(customer_name, current_order):
    """
    Recommends products a customer has bought previously but is missing from
    their current order, using a K-Nearest Neighbors AI model.
    """
    try:
        # Load the dataset
        df = pd.read_csv('final.csv')
    except FileNotFoundError:
        return "Error: The dataset 'final.csv' was not found."

    # Step 1: Data Preparation for the AI Model
    # Combine all product columns into a single list per row
    product_columns = ['product1', 'product2', 'product3']
    df['all_products'] = df[product_columns].apply(
        lambda row: [item for item in row if pd.notna(item)], axis=1
    )

    # Create a unique list of all products
    all_products = sorted(list(set([prod for sublist in df['all_products'] for prod in sublist])))
    
    # Create a DataFrame for model training with one-hot encoding
    # A '1' indicates a product was purchased, a '0' indicates it was not
    model_df = pd.DataFrame(
        {product: [1 if product in row else 0 for row in df['all_products']] for product in all_products}
    )
    model_df['customer_name'] = df['customer_name']
    
    # Aggregate data by customer to get a single row for each customer
    customer_purchase_history = model_df.groupby('customer_name').max()

    # Step 2: Train the K-Nearest Neighbors (KNN) Model
    X = customer_purchase_history.drop(columns=['customer_name'], errors='ignore')
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    model.fit(X)

    # Step 3: Get Recommendations
    if customer_name not in customer_purchase_history.index:
        return f"No previous order history found for customer: {customer_name}"

    # Get the purchase history of the customer to check
    customer_vector = X.loc[[customer_name]]
    
    # Find the 5 nearest customers (neighbors)
    distances, indices = model.kneighbors(customer_vector)
    
    # Get the combined purchase history of the neighboring customers
    neighbor_indices = indices[0]
    neighbor_data = X.iloc[neighbor_indices]
    
    # Calculate the average purchase history of the neighbors
    neighbor_purchases = neighbor_data.sum(axis=0)
    
    # Identify highly-purchased products among neighbors
    threshold = 1 # A product bought by at least one neighbor
    recommended_products = neighbor_purchases[neighbor_purchases >= threshold].index.tolist()
    
    # Filter out products already in the current order
    current_order_set = set(current_order)
    missed_products = [prod for prod in recommended_products if prod not in current_order_set]

    if not missed_products:
        return "No missed products to recommend. You have all your usual items!"
    else:
        return f"Based on your purchasing habits and similar customers, you may have missed: {', '.join(missed_products)}"

# Example Usage:
# Imagine 'Hussain' is starting a new order with 'Fish' and 'Salt'.
customer_to_check = 'Hussain'
new_order = ['Fish', 'Biscuit','Boost','Bread', 'Butter', 'Chicken', 'Coke', 'Eggs', 'Horlicks', 'Lays', 'Milk', 'Oil', 'Paneer', 'Pepsi', 'Red Chili Powder', 'Rice', 'Salt' ]
recommendations = get_missed_product_recommendations_ai(customer_to_check, new_order)
print(recommendations)

'''# Another example for 'Ram'
customer_to_check_2 = 'Ram'
new_order_2 = ['Lays']
recommendations_2 = get_missed_product_recommendations_ai(customer_to_check_2, new_order_2)
print(recommendations_2)
'''