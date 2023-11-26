import streamlit as st
import pandas as pd
import numpy as np 
import base64
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from surprise import Dataset, Reader, KNNBasic, SVD, NMF, dump
from surprise.model_selection import train_test_split

#Load product data
df_product_info = pd.read_csv("https://raw.githubusercontent.com/manaswitha1001/beautyrecommendationapp/main/datasets/product_info.csv", header = 0)
df_product = df_product_info[['product_id', 'product_name', 'brand_name','ingredients','highlights', 'primary_category', 'secondary_category', 'tertiary_category']]
df_product['tertiary_category'].fillna(df_product['secondary_category'], inplace=True)
df_product = df_product.dropna()

#Load reviews data
df_reviews = pd.read_csv('https://raw.githubusercontent.com/manaswitha1001/beautyrecommendationapp/main/datasets/reviews.csv', header = 0, low_memory=False)
df_review = df_reviews[['author_id', 'product_id', 'rating']]
df_review = df_review.drop_duplicates()
reader = Reader(rating_scale=(0, 5))
review_data = Dataset.load_from_df(df_review, reader)
trainset, testset = train_test_split(review_data, test_size=.2, random_state=42)
trainset = review_data.build_full_trainset()

#utility functions 

def get_rated_products_by_user(user_id, df_review, df_product_info):
    # Filter reviews by the user and high ratings
    user_ratings = df_review[(df_review['author_id'] == user_id) & (df_review['rating'] >= 4.0)]

    # Merge with product information
    merged_data = pd.merge(user_ratings, df_product_info, on='product_id')

    print("\nProducts with High Ratings by User", user_id, ":")
    
    # Extract relevant columns
    product_details = merged_data[['product_id', 'product_name', 'primary_category', 'rating_x']]

    return product_details.reset_index(drop=True)



# The method get_user_recommendations(user_id, model, df_review, df_product_info) takes the input user_id, model type review info and produict info and returns the list of products which can be a potential recommendations for the user. 

def get_user_recommendations(user_id, model, df_review, df_product_info, n=10):
    # Get a list of all unique product IDs
    all_product_ids = df_review['product_id'].unique()

    # Remove products that the user has already rated
    products_rated_by_user = df_review[df_review['author_id'] == user_id]['product_id'].values
    products_to_predict = np.setdiff1d(all_product_ids, products_rated_by_user)

    # Make predictions for the products to predict
    predictions = [model.predict(user_id, product_id) for product_id in products_to_predict]

    # Sort the predictions by estimated ratings (in descending order)
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top-N recommended products
    top_n_recommendations = [prediction.iid for prediction in predictions[:n]]

    # Populate the recommendations list
    recommendations = []
    for i, product_id in enumerate(top_n_recommendations, start=1):
        # Find the product details in df_product_info
        product_details = df_product_info[df_product_info['product_id'] == product_id]

        if len(product_details) > 0:
            product_name = product_details['product_name'].values[0]
            primary_category = product_details['primary_category'].values[0]

            # Append the recommendation to the list
            recommendations.append([i, product_id, product_name, primary_category])

    columns = ['Rank', 'Product_ID', 'Product_Name', 'Primary_Category']
    recommendations_df = pd.DataFrame(recommendations, columns=columns)

    return recommendations_df

def get_recommendations(product_id, top_n):
    product_index = df_product[df_product['product_id'] == product_id].index[0]
    similarities = similarity_matrix[product_index]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]  # Exclude the product itself
    recommendations = df_product.loc[similar_indices, ['product_id','product_name', 'brand_name', 'primary_category', 'tertiary_category']]
    recommendations_reset = recommendations.reset_index(drop=True)
    recommendations_reset.insert(0, 'rank', range(1, top_n+1))  # Add a rank column at the beginning
    recommendations_reset = recommendations_reset.iloc[:, 1:]
    return recommendations_reset

#TFIDF model 
tfidf = TfidfVectorizer()
ingredient_vector = tfidf.fit_transform(df_product['ingredients'])
highlights_vector = tfidf.fit_transform(df_product['highlights'])
prim_category_vector = tfidf.fit_transform(df_product['primary_category'])
sec_category_vector = tfidf.fit_transform(df_product['secondary_category'])
tert_category_vector = tfidf.fit_transform(df_product['tertiary_category'])


# Combine ingredient , highlights and category vectors
feature_vectors = hstack((ingredient_vector,highlights_vector, prim_category_vector,sec_category_vector, tert_category_vector))
similarity_matrix = cosine_similarity(feature_vectors)
df_product = df_product.reset_index(drop=True)


##SVD MODEL
best_params = {'n_factors': 100, 'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.1}
svd_best_model = SVD(**best_params)
svd_best_model.fit(review_data.build_full_trainset())

#KNN MODEL
# best_params = {'k':10}
# knn_best_model = KNNBasic(**best_params)
# knn_best_model.fit(review_data.build_full_trainset())


# Add background image 
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
background_image_path = 'https://github.com/manaswitha1001/beautyrecommendationapp/blob/main/images/makeup.jpg'

add_bg_from_local(background_image_path) 

# Side bar 
st.sidebar.header('User Input')
preference = st.sidebar.slider('Choose number of recommendations', 0, 10, 5)
page_options = ['Trendspotter','TraitPicker','SquadSuggest']
choice = st.sidebar.selectbox('Select Page', page_options)


if choice == 'Trendspotter':
    st.write('Trending Now')
    df_product_info['product_brand'] = df_product_info['brand_name'] + ' - ' + df_product_info['product_name']
    df_sorted = df_product_info.sort_values(by='loves_count', ascending=False)
    top_products = df_sorted['product_brand'].head(n= preference)
    st.dataframe(top_products)

elif choice == 'TraitPicker':
    st.title('Sephora Content-Based Recommendation System')
    col1, col2 = st.columns(2)
    # Dropdown for selecting brand
    with col1:
        selected_brand = st.selectbox('Select Brand', df_product['brand_name'].unique())
    # Filter products based on the selected brand
    filtered_products = df_product[df_product['brand_name'] == selected_brand]['product_name']
    # Dropdown for selecting product
    with col2:
        selected_product = st.selectbox('Select Product', filtered_products)
    # Display the selected brand and product
    st.write(f'Selected Brand: {selected_brand}')
    st.write(f'Selected Product: {selected_product}')

    info_product = df_product[(df_product['product_name'] == selected_product) & (df_product['brand_name'] == selected_brand)][['product_id','product_name', 'brand_name', 'primary_category','secondary_category' ,'tertiary_category']]
    product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )

    # Button to get recommendations
    if st.button('Get Recommendations'):
        info_product = df_product[df_product['product_name'] == selected_product][['product_id','product_name', 'brand_name', 'primary_category','secondary_category' ,'tertiary_category']]
        product_table = tabulate(info_product, headers='keys', tablefmt='psql',showindex=False )
        st.write(info_product)
        #recommendations = get_similar_recommendations(info_product['product_id'].values[0], df_product,similarity_matrix, top_n = preference)
        recommendations = get_recommendations(info_product['product_id'].values[0], top_n=preference)
        st.write('Similar products')
        st.write(recommendations)


elif choice == 'SquadSuggest':
    st.header('Squad Recommender system')
    user_input = st.text_input('Enter the user id here:', '8447021668')
    if st.button("Get recommendations for user"):
        rated_products = get_rated_products_by_user(user_input, df_review, df_product_info)
        st.write("The products rated by userid ", user_input)
        st.write(rated_products)
        recs = get_user_recommendations(user_input, svd_best_model, df_review, df_product_info, n=preference)
        st.write("The products recommended for  userid ", user_input)
        st.write(recs)



