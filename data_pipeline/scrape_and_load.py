import os
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def scrape_category(start_url, category):
    """Scrape all pages of one book category."""

    current_url = start_url
    books = []

    while current_url:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for book in soup.select("article.product_pod"):
            title = book.h3.a["title"]

            price = book.select_one(
                ".price_color"
            ).get_text(strip=True)

            rating_classes = book.select_one(
                "p.star-rating"
            )["class"]

            star_rating = next(
                rating
                for rating in rating_classes
                if rating != "star-rating"
            )

            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category,
                }
            )

        # Find the next page
        next_link = soup.select_one("li.next a")

        if next_link:
            next_url = next_link["href"]

            if current_url.endswith("/"):
                current_url = current_url + next_url
            else:
                current_url = (
                    current_url.rsplit("/", 1)[0]
                    + "/"
                    + next_url
                )
        else:
            current_url = None

    return books


def clean_data(df):
    """Clean and enrich scraped book data."""

    # Price: remove currency symbols and convert to float
    df["price_gbp"] = (
        df["price"]
        .str.replace(r"[^\d.]", "", regex=True)
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce",
    )

    # Rating: convert text to integer
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # Availability: convert text to boolean
    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False,
    )

    # Median imputation for unexpected numeric parsing failures
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)

    median_rating = df["rating"].median()
    df["rating"] = (
        df["rating"]
        .fillna(median_rating)
        .round()
        .astype(int)
    )

    # Required fixed project conversion rate
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df


def create_database(df, db_path):
    """Create normalized SQLite database and load cleaned data."""

    # Remove old database so every run starts from scratch
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    # Create categories table
    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    # Create books table
    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # Insert unique categories
    categories = df["category"].unique()

    for category in categories:
        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,),
        )

    # Create category lookup
    cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    )

    category_lookup = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    # Insert books
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                int(row["rating"]),
                int(row["in_stock"]),
                category_lookup[row["category"]],
            ),
        )

    conn.commit()

    return conn


if __name__ == "__main__":

    categories = {
        "Travel": (
            "https://books.toscrape.com/"
            "catalogue/category/books/travel_2/index.html"
        ),
        "Mystery": (
            "https://books.toscrape.com/"
            "catalogue/category/books/mystery_3/index.html"
        ),
        "Historical Fiction": (
            "https://books.toscrape.com/"
            "catalogue/category/books/"
            "historical-fiction_4/index.html"
        ),
    }

    all_books = []

    # Scrape all selected categories
    for category, url in categories.items():
        books = scrape_category(url, category)

        print(
            f"{category}: {len(books)} books scraped"
        )

        all_books.extend(books)

    # Create DataFrame
    df = pd.DataFrame(all_books)

    print()
    print(f"Raw books scraped: {len(df)}")

    # Clean and enrich
    df = clean_data(df)

    print()
    print("Cleaned DataFrame:")
    print(
        df[
            [
                "title",
                "price_gbp",
                "rating",
                "in_stock",
                "price_inr",
                "category",
            ]
        ].head()
    )

    print()
    print("Data types:")
    print(df.dtypes)

    print()
    print("Missing values:")
    print(df.isnull().sum())

    print()
    print(
        f"GBP → INR rate used: {GBP_TO_INR}"
    )

    # Create SQLite database
    db_path = "data_pipeline/books.db"

    conn = create_database(
        df,
        db_path,
    )

    print()
    print(
        f"Database created: {db_path}"
    )

    conn.close()