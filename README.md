# Sephora Beauty Recommender 

Sephora stands out as a leading e-commerce platform, globally renowned for its vast selection of high-quality cosmetics and beauty products. The platform offers a diverse range of items, including skincare, makeup, hair care, and perfumes from reputable brands. Navigating through numerous options, product complexity, limited information, and slow search speed poses challenges for customers. To address these issues, the development of a product recommendation system becomes crucial. Such a system aims to enhance the shopping experience, provide tailored suggestions, and boost overall sales.

The primary objective is to offer customers personalized product recommendations based on their preferences and characteristics.
The project employs three distinct approaches: 

**Fav Finder - Popularity Based Algorithm:**
Fav Finder employs a popularity-based algorithm to recommend products. It prioritizes items based on their overall popularity, often measured by factors such as user ratings, likes, or sales counts. This approach suggests products that have gained significant attention or positive feedback from a broad user base.

**Trait Picker - Content-Based Filtering Algorithm:**
Trait Picker utilizes a content-based filtering algorithm for product recommendations. This approach involves analyzing the inherent characteristics and features of products. In Trait Picker, the system recommends items by considering specific traits, such as brand, category, or ingredients, aligning with user preferences derived from their previous choices or stated traits.

**Squad Suggester - Collaborative Filtering Algorithm:**
Squad Suggester employs a collaborative filtering algorithm to provide personalized recommendations. This method identifies patterns of similarity between users based on their past behavior, preferences, or ratings. Squad Suggester recommends products that align with the preferences of users who share similar tastes, enhancing the chances of user satisfaction.


# Dataset Overview
The dataset used in this project was obtained from the Kaggle dataset entitled [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews). This dataset provides comprehensive information on more than 8,000 beauty products available in Sephora's online store. This dataset includes various attributes for each product, such as product name, product brand, price, ingredients, ratings, and other relevant features.

The datasets used in this project are: "product_info.csv" and "reviews.csv".

* The file "product_info.csv" has 8,494 rows and 27 columns. Each row represents a beauty product available in the Sephora online store, and the columns contain the following information:

  * `product_id`: Unique ID for the product on the website
  * `product_name`: Full name of the product
  * `brand_id`: Unique identifier for the product brand from the website
  * `brand_name`: Product brand name
  * `loves_count` The number of people who have marked this product as a favorite
  * `rating`: Average rating of the product based on user reviews
  * `reviews`: Number of user reviews for the product
  * `size`: Product size, can be in oz, ml, g, packets, or other units depending on the product type
  * `variation_type`: Variation parameter type for the product
  * `variation_value`: The specific value of the variation parameter for the product
  * `variation_desc`: Description of variation parameters for the product
  * `ingredients`: List of ingredients contained in the product.
  * `price_usd`: Product price in United States dollars
  * `value_price_usd`: Product cost savings potential, displayed on the website next to the regular price
  * `sale_price_usd`: Product sale price in United States dollars
  * `limited_edition`: Indicates whether this product is limited edition or not
  * `new`: Indicates whether this product is new or not
  * `online_only`: Indicates whether this product is only sold online or not
  * `out_of_stock`: Indicates whether this product is currently out of stock or not
  * `sephora_exclusive`: Indicates whether this product is exclusive to Sephora or not
  * `highlights`: A list of tags or features that highlight product attributes
  * `primary_category`: The primary category of the product
  * `secondary_category`: Second category
  * `tertiary_category`: The tertiary category of the product
  * `child_count`: Number of product variations available
  * `child_max_price`: Highest price among product variations
  * `child_min_price`: Lowest price among product variations
 
An overview of the dataframe from product_info.csv can be seen in Table 1. which contains several data related to product details.

Table 1. Product dataframe
| product_id |       product_name       | brand_id | brand_name | loves_count | rating | reviews |     size      |     variation_type      |   variation_value   | ... | online_only | out_of_stock | sephora_exclusive |                     highlights                    | primary_category | secondary_category |  tertiary_category  | child_count | child_max_price | child_min_price |
|------------|--------------------------|----------|------------|-------------|--------|---------|---------------|-------------------------|---------------------|-----|-------------|--------------|-------------------|---------------------------------------------------|------------------|---------------------|---------------------|-------------|-----------------|-----------------|
|  P473671   | Fragrance Discovery Set  |   6342   |   19-69    |     6320    | 3.6364 |   11.0  |     NaN       |          NaN            |        NaN          | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Warm & Spicy Scen... |     Fragrance     |  Value & Gift Sets |  Perfume Gift Sets |      0      |       NaN       |       NaN       |
|  P473668   |  La Habana Eau de Parfum |   6342   |   19-69    |     3827    | 4.1538 |   13.0  | 3.4 oz/ 100 mL| Size + Concentration + Formulation | 3.4 oz/ 100 mL | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Layerable Scent'... |     Fragrance     |       Women       |      Perfume       |      2      |      85.0       |      30.0       |
|  P473662   | Rainbow Bar Eau de Parfum|   6342   |   19-69    |     3253    |  4.25  |   16.0  | 3.4 oz/ 100 mL| Size + Concentration + Formulation | 3.4 oz/ 100 mL | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Layerable Scent'... |     Fragrance     |       Women       |      Perfume       |      2      |      75.0       |      30.0       |


* The file "reviews.csv" consists of 49,977 rows. Each row represents a user review for a specific product, and the columns include:

  * `author_id`: Unique ID for the review author on the website
  * `rating`: The rating given by the review author to the product on a scale of 1 to 5
  * `is_recommended`: Indicates whether the author recommends the product or not
  * `helpfulness`: Ratio between number of positive reviews and total reviews: helpfulness = total_pos_feedback_count / total_feedback_count
  * `total_feedback_count`: Total number of feedback (positive and negative ratings) given by users for reviews
  * `total_neg_feedback_count`: Number of users who gave a negative rating to a review
  * `total_pos_feedback_count`: Number of users who gave a positive rating to a review
  * `submission_time`: Date the review was posted on the website in 'yyyy-mm-dd' format
  * `review_text`: The main text of the review written by the author
  * `review_title`: Title of the review written by the author
  * `skin_tone`: Author's skin tone
  * `eye_color`: Author's eye color
  * `skin_type`: Author's skin type
  * `hair_color`: Author's hair color
  * `product_id`: Unique ID for the product

An overview of the dataframe from reviews.csv can be seen in Table 2. which contains some data related to user reviews.

Tabel 2. Dataframe review
| Unnamed: 0 | author_id    | rating | is_recommended | helpfulness | total_feedback_count | total_neg_feedback_count | total_pos_feedback_count | submission_time | review_text                                          | review_title                          | skin_tone     | eye_color | skin_type   | hair_color | product_id | product_name                                     | brand_name | price_usd |
|------------|--------------|--------|----------------|-------------|----------------------|--------------------------|--------------------------|-----------------|------------------------------------------------------|---------------------------------------|---------------|-----------|-------------|------------|------------|--------------------------------------------------|------------|-----------|
| 0          | 1945004256   | 5      | 1              | 1.0         | 0.000000             | 2                        | 2                        | 2022-12-10      | I absolutely L-O-V-E this oil. I have acne pro...    | A must have!                          | lightMedium   | green     | combination | NaN        | P379064    | Lotus Balancing & Hydrating Natural Face Treat... | Clarins    | 65.0      |
| 1          | 5478482359   | 3      | 1              | 1.0         | 0.333333             | 3                        | 2                        | 2021-12-17      | I gave this 3 stars because it give me tiny li...    | it keeps oily skin under control      | mediumTan    | brown     | oily        | black      | P379064    | Lotus Balancing & Hydrating Natural Face Treat... | Clarins    | 65.0      |
| 2          | 29002209922  | 5      | 1              | 1.0         | 1.000000             | 2                        | 0                        | 2021-06-07      | Works well as soon as I wash my face and pat d...    | Worth the money!                      | lightMedium   | brown     | dry         | black      | P379064    | Lotus Balancing & Hydrating Natural Face Treat... | Clarins    | 65.0      |



## Installation

To run the streamlit app locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/beauty-recommender-app.git
   cd beauty-recommender-app
   ```

2. **Install Dependencies**
Make sure you have the necessary dependencies installed. Use the package manager of your choice (e.g., pip for Python).
 ```bash
    pip install -r requirements.txt
```

3.**Run the App**
Execute the app script to start the local development server.
 ```bash
    streamlit run app.py
```

The app is dpeloyed in the Streamlit community cloud
App Link - https://beautyrecommender.streamlit.app/

