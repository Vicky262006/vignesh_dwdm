from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import fpgrowth
from prophet import Prophet
import io
from functools import wraps

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "smartcart_secret"

# ---------------- MongoDB connection ----------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["grocerydb"]
users_collection = db["users"]
orders_collection = db["orders"]
neworders_collection = db["neworders"]
admin_collection = db["admins"]

# ---------------- Token verification ----------------
def admin_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get("admin_token")
        if not token:
            return redirect(url_for("admin_login_page"))
        try:
            decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            admin_name = decoded.get("admin_name")
            current_admin = admin_collection.find_one({"name": admin_name})
            if not current_admin:
                return redirect(url_for("admin_login_page"))
        except:
            return redirect(url_for("admin_login_page"))
        return f(current_admin, *args, **kwargs)
    return decorated

def user_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get("user_token")
        if not token:
            return redirect(url_for("login_page"))
        try:
            decoded = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            username = decoded.get("name")
            current_user = users_collection.find_one({"name": username})
            if not current_user:
                return redirect(url_for("login_page"))
        except:
            return redirect(url_for("login_page"))
        return f(current_user, *args, **kwargs)
    return decorated

# ---------------- Serve HTML pages ----------------
@app.route("/", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")

@app.route("/index_page")
@user_token_required
def index_page(current_user):
    return render_template("index.html", user_name=current_user["name"])

@app.route("/admin_login_page")
def admin_login_page():
    return render_template("admin_login.html")

@app.route("/admin_dashboard")
@admin_token_required
def admin_dashboard(current_admin):
    return render_template("admin_dashboard.html", admin_name=current_admin["name"])

@app.route("/add_admin_page")
@admin_token_required
def add_admin_page(current_admin):
    return render_template("add_admin.html", admin_name=current_admin["name"])

# ---------------- Signup API ----------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already registered"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({"name": name, "email": email, "password": hashed_pw})
    return jsonify({"message": "Signup successful"}), 200

# ---------------- User Login API ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name")
    password = data.get("password")

    user = users_collection.find_one({"name": name})
    if not user:
        return jsonify({"message": "User not found"}), 400

    if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        token = jwt.encode({
            "id": str(user["_id"]),
            "name": user["name"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        response = make_response(jsonify({"message": "Login successful", "redirect": "/index_page"}))
        response.set_cookie("user_token", token, httponly=True)
        return response
    else:
        return jsonify({"message": "Invalid credentials"}), 400

# ---------------- Admin Login API ----------------
@app.route("/admin_login", methods=["POST"])
def admin_login():
    data = request.json
    name = data.get("name")
    password = data.get("password")

    admin = admin_collection.find_one({"name": name})
    if not admin:
        return jsonify({"message": "Admin not found"}), 400

    if bcrypt.checkpw(password.encode("utf-8"), admin["password"]):
        token = jwt.encode({
            "admin_name": name,
            "role": "admin",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        response = make_response(jsonify({"message": "Login successful", "redirect": "/admin_dashboard"}))
        response.set_cookie("admin_token", token, httponly=True)
        return response
    else:
        return jsonify({"message": "Invalid credentials"}), 400

# ---------------- Logout ----------------
@app.route("/logout")
def logout():
    response = redirect(url_for("login_page"))
    response.delete_cookie("user_token")
    response.delete_cookie("admin_token")
    return response

# ---------------- Add New Admin API ----------------
@app.route("/add_admin", methods=["POST"])
@admin_token_required
def add_admin(current_admin):
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"message": "All fields required"}), 400

    if admin_collection.find_one({"email": email}):
        return jsonify({"message": "Admin already exists"}), 409

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    admin_collection.insert_one({"name": name, "email": email, "password": hashed_pw})
    return jsonify({"message": "New admin added successfully"}), 201

# ---------------- Import Dataset API ----------------
@app.route("/import_dataset", methods=["POST"])
@admin_token_required
def import_dataset(current_admin):
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded"}), 400
    try:
        filename = file.filename.lower()
        content = file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return jsonify({"message": "Unsupported file type"}), 400

        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        if records:
            orders_collection.insert_many(records)

        return jsonify({"message": f"Inserted {len(records)} records into database"}), 200
    except Exception as e:
        return jsonify({"message": f"Error importing dataset: {str(e)}"}), 500

# ---------------- Sales Graph Route ----------------
@app.route("/sales_graph")
@admin_token_required
def sales_graph(current_admin):
    try:
        df = pd.read_csv("main_cleaned.csv")
        if 'order_date' not in df.columns or 'Sales' not in df.columns:
            return "Dataset must contain 'order_date' and 'Sales' columns.", 400

        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
        df = df.dropna(subset=['order_date'])
        df = df.sort_values('order_date')

        # Historical sales
        sales_data = df.groupby('order_date')['Sales'].sum().reset_index()
        graph_data = {
            "dates": sales_data['order_date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": sales_data['Sales'].tolist()
        }

        # Prophet forecast next 30 days
        prophet_df = sales_data.rename(columns={'order_date':'ds', 'Sales':'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast_graph = {
            "dates": forecast['ds'].dt.strftime('%Y-%m-%d')[-30:].tolist(),
            "predicted_sales": forecast['yhat'][-30:].fillna(0).round(2).astype(float).tolist()
        }

        return render_template(
            "sales_graph.html",
            graph_data=graph_data,
            forecast_graph=forecast_graph,
            admin_name=current_admin["name"]
        )

    except Exception as e:
        return f"Error generating sales graph: {str(e)}", 500

# ---------------- Build Customer-Product Matrix ----------------
def build_customer_product_matrix():
    orders = list(orders_collection.find({}, {"_id": 0}))
    if not orders:
        return None, [], None

    df = pd.DataFrame(orders)
    product_cols = [col for col in df.columns if col.lower().startswith("product") or col.lower().startswith("item")]
    if not product_cols:
        return None, [], df

    df['all_products'] = df[product_cols].apply(
        lambda r: [str(x).strip() for x in r if x and str(x).upper() not in ("NULL","NAN") and not str(x).isdigit()],
        axis=1
    )
    all_products = sorted({p for sublist in df['all_products'] for p in sublist})
    if 'customer_name' not in df.columns:
        candidate = next((c for c in df.columns if c.lower() in ('customer', 'name', 'customer_name')), None)
        df['customer_name'] = df[candidate] if candidate else None

    product_matrix = pd.DataFrame({p: [1 if p in row else 0 for row in df['all_products']] for p in all_products})
    product_matrix['customer_name'] = df['customer_name']
    customer_history = product_matrix.groupby('customer_name').max()
    return customer_history, all_products, df

# ---------------- FP-Growth Recommendations ----------------
def fp_growth_recommend(df_orders, current_order, min_support=0.05):
    all_products = sorted({p for sublist in df_orders['all_products'] for p in sublist})
    ohe_df = pd.DataFrame([{p: 1 if p in row else 0 for p in all_products} for row in df_orders['all_products']])
    freq_items = fpgrowth(ohe_df, min_support=min_support, use_colnames=True)
    freq_items = freq_items.sort_values('support', ascending=False)

    candidates = set()
    for products in freq_items['itemsets']:
        if any(p in products for p in current_order):
            candidates.update(products)
    candidates -= set(current_order)
    candidates = [p for p in candidates if p and str(p).upper() not in ("NULL","NAN") and not str(p).isdigit()]
    return list(candidates)

# ---------------- Recommendations API ----------------
@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        data = request.json or {}
        customer_name = str(data.get("customer_name", "")).strip()
        current_order = [str(p).strip() for p in data.get("current_order", []) if p]

        customer_history, all_products, df_orders = build_customer_product_matrix()
        if customer_history is None or len(all_products) == 0:
            return jsonify({"missed_products": []}), 200

        product_prices = {}
        if df_orders is not None:
            for p in all_products:
                row = df_orders[df_orders['all_products'].apply(lambda lst: p in lst)]
                if not row.empty and 'price' in row.columns:
                    product_prices[p] = float(row['price'].iloc[0])
                else:
                    product_prices[p] = 50.0
        else:
            product_prices = {p: 50.0 for p in all_products}

        fp_candidates = fp_growth_recommend(df_orders, current_order)

        if customer_name in customer_history.index:
            X = customer_history.fillna(0).astype(float)
            if len(X) > 1:
                n_neighbors = min(5, len(X))
                knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
                knn.fit(X)
                customer_vector = X.loc[[customer_name]]
                distances, indices = knn.kneighbors(customer_vector)
                distances = distances.flatten()
                indices = indices.flatten()
                eps = 1e-6
                weights = 1.0 / (distances + eps)
                neighbor_rows = X.iloc[indices]
                weighted_scores = (neighbor_rows.T * weights).T.sum(axis=0)

                user_history = X.loc[customer_name]
                user_products = [p for p in X.columns if int(user_history[p]) == 1]
                candidate_products = [p for p in user_products if p not in current_order]
                candidate_products = list(set(candidate_products) | set(fp_candidates))
                candidate_scores = {p: float(weighted_scores.get(p, 0.0)) for p in candidate_products}
                ranked = [p for p, s in sorted(candidate_scores.items(), key=lambda x: -x[1])]

                cleaned_products = [p for p in ranked if p and str(p).upper() not in ("NULL","NAN") and not str(p).isdigit()]
                return jsonify({"missed_products": [{"name": p, "price": product_prices.get(p,50)} for p in cleaned_products]}), 200

        popular_scores = {p: df_orders['all_products'].apply(lambda lst: 1 if p in lst else 0).sum() for p in all_products}
        popular_sorted = [p for p, s in sorted(popular_scores.items(), key=lambda x: -x[1]) if p not in current_order]
        candidates = list(dict.fromkeys(fp_candidates + popular_sorted))
        cleaned_candidates = [p for p in candidates if p and str(p).upper() not in ("NULL","NAN") and not str(p).isdigit()]
        return jsonify({"missed_products": [{"name": p, "price": product_prices.get(p,50)} for p in cleaned_candidates]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Place Order API ----------------
@app.route("/place_order", methods=["POST"])
@user_token_required
def place_order(current_user):
    try:
        data = request.json or {}
        products = data.get("items")
        if isinstance(products, dict):
            products = [products]

        if not products or len(products) == 0:
            return jsonify({"message": "No products to place order"}), 400

        order = {
            "customer_name": current_user["name"],
            "products": products,
            "order_date": datetime.datetime.utcnow()
        }

        neworders_collection.insert_one(order)
        return jsonify({"message": "Order placed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
