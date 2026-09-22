import sqlite3

DB_PATH = "data_pipeline/books.db"

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

# Check tables 
cursor.execute("""
    SELECT name 
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""")

tables = cursor.fetchall()

print("Tables:")
for table in tables:
    print("-", table[0])


# Check category count 

cursor.execute("""
    SELECT COUNT(*)
    FROM categories
""")

category_count = cursor.fetchone()[0]

print()
print("Number of categories:", category_count)

# Check book count

cursor.execute("""
    SELECT COUNT(*)
    FROM books
""")

book_count = cursor.fetchone()[0]

print("Number of books:", book_count)

# Check books per category
cursor.execute("""
    SELECT 
        c.category_name,
        COUNT(b.book_id) AS book_count
    FROM categories AS c
    JOIN books AS b
        ON c.category_id = b.category_id
    GROUP BY c.category_name
    ORDER BY book_count DESC
""")

print()
print("Books by category:")

for category, count in cursor.fetchall():
    print(f"{category}: {count}")

conn.close()