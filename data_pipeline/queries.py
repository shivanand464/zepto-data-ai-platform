import sqlite3
from contextlib import redirect_stdout
import pandas as pd

DB_PATH = "data_pipeline/books.db"
OUTPUT_PATH = "data_pipeline/query_results.txt"

def run_query(conn, query_number, query):
    """Execute a SQL query and print its query and output."""

    print("=" * 70)
    print(f"QUERY {query_number}")
    print("=" * 70)

    print("\nSQL:")
    print(query)

    df = pd.read_sql_query(query,conn)

    print("\nOUTPUT:")
    print(df.to_string(index=False))

    print()

    return df

if __name__ == "__main__":

    conn = sqlite3.connect(DB_PATH) 

    # Query 1: SELECT + WHERE

    query_1 = """
        SELECT
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4 
    """

    # Query 2: ORDER BY

    query_2 = """
        SELECT 
            title,
            price_gbp,
            price_inr
        FROM books
        ORDER BY price_gbp DESC
    """

    # Query 3: LIMIT 

    query_3 = """
        SELECT 
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY rating DESC, price_gbp DESC
        LIMIT 10
    """

    # Query 4: DISTINCT

    query_4 = """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """

    # Query 5: BETWEEN

    query_5 = """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """

    # Query 6: JOIN

    query_6 = """
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

    queries = [
        (1, query_1),
        (2, query_2),
        (3, query_3),
        (4, query_4),
        (5, query_5),
        (6, query_6),
    ]

    # Print to terminal and save identical output to a file 

    with open(OUTPUT_PATH,"w",
              encoding="utf-8",
    ) as output_file:

        with redirect_stdout(output_file):
            for query_number, query in queries:
                run_query(
                    conn,
                    query_number,
                    query,
                )

    # Also display a confirmation in terminal 

    print(
        f"Query results saved to: {OUTPUT_PATH}"
    )

        
    conn.close()
    








