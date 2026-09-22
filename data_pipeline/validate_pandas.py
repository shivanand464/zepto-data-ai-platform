import sqlite3

import pandas as pd 

DB_PATH = "data_pipeline/books.db"

if __name__ == "__main__":

    conn = sqlite3.connect(DB_PATH)

    # 1. Read SQL Query 1 using pd.read_sql

    query_1 = """
        SELECT 
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
    """

    sql_result_1 = pd.read_sql(
        query_1,
        conn,
    )

    print("=" * 70)
    print("pd.read_sql() - Query 1")
    print("=" * 70)
    print(sql_result_1.head(10).to_string(index=False))


    # 2. Read JOIN query using pd.read_sql

    join_query = """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.price_inr
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC
        LIMIT 10
    """

    sql_join_result = pd.read_sql(
        join_query,
        conn,
    )

    print()
    print("=" * 70)
    print("SQL JOIN RESULT - pd.read_sql()")
    print("=" * 70)
    print(sql_join_result.to_string(index=False))

    # 3. Read raw tables into pandas 

    books_df = pd.read_sql(
        "SELECT * FROM books",
        conn,
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        conn,
    )

    conn.close()

    # 4. Reproduce the JOIN using pd.merge()

    merged_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
    )

    # Select the same columns as SQL JOIN 
    merged_result = merged_result[
        [
            "title",
            "category_name",
            "rating",
            "price_gbp",
            "price_inr",
        ]
    ]


    # Apply the same ordering and LIMIT 

    merged_result = (
        merged_result.sort_values(by=["rating", "price_gbp"],ascending=[False, False],).head(10).reset_index(drop=True))

    sql_join_result = (
        sql_join_result.reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("PANDAS JOIN RESULT - pd.merge()")
    print("=" * 70)
    print(merged_result.to_string(index=False))

    # Compare both results


equivalent = sql_join_result.equals(
    merged_result
)

print()
print("=" * 70)
print("JOIN EQUIVALENCE CHECK")
print("=" * 70)

print(
    "SQL JOIN and pd.merge() equivalent:",equivalent,
)
