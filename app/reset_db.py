"""
Clears all data from both Neo4j Aura and Weaviate Cloud.
Run: python -m app.reset_db

Options:
    --neo4j-only     clear only Neo4j
    --weaviate-only  clear only Weaviate
    --pinecone-only  clear only Pinecone
    --confirm        skip confirmation prompt
"""
import sys
import weaviate
from neo4j import GraphDatabase
from weaviate.classes.init import Auth
from pinecone import Pinecone

from app.config import settings


def clear_neo4j(confirm: bool = False):
    print("\n---------------- Neo4j Aura ----------------")
    driver = GraphDatabase.driver(
        uri=settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    with driver.session() as session:
        count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f"Nodes before : {count}")

        if count == 0:
            print(" - Already empty, skipping.")
            driver.close()
            return

        if not confirm:
            ans = input(f" - Delete ALL {count} nodes and relationships? [y/N]: ").strip().lower()
            if ans != "y":
                print(" - Skipped.")
                driver.close()
                return

        print("  Deleting in batches...")
        deleted = 0
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c"
            )
            batch = result.single()["c"]
            deleted += batch
            if batch == 0:
                break
            print(f" - Deleted {deleted} nodes so far...")

        count_after = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f" [SUCCESS] Neo4j cleared. Nodes remaining: {count_after}")

    driver.close()


def clear_weaviate(confirm: bool = False):
    print("\n---------------- Weaviate Cloud ----------------")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
    )

    collections = ["Chunk", "Entity", "Section", "Document"]  # reverse depth order
    totals = {}
    for name in collections:
        if client.collections.exists(name):
            col = client.collections.get(name)
            totals[name] = col.aggregate.over_all(total_count=True).total_count
        else:
            totals[name] = 0

    total = sum(totals.values())
    for name, cnt in totals.items():
        print(f" - {name:<12} : {cnt} objects")

    if total == 0:
        print(" - Already empty, skipping.")
        client.close()
        return

    if not confirm:
        ans = input(f"\n - Delete ALL {total} objects and recreate schema? [y/N]: ").strip().lower()
        if ans != "y":
            print(" - Skipped.")
            client.close()
            return

    print(" - Dropping collections...")
    from app.data.storage.weaviate_schema import WeaviateSchema
    schema = WeaviateSchema(client)
    schema.drop_schema()
    print(" - Recreating schema...")
    schema.create_schema()

    # Verify
    for name in collections:
        if client.collections.exists(name):
            cnt = client.collections.get(name).aggregate.over_all(total_count=True).total_count
            print(f" [SUCCESS] {name:<12} : {cnt} objects (fresh)")

    client.close()


def clear_pinecone(confirm: bool = False):
    print("\n---------------- Pinecone ----------------")
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME

    if index_name not in pc.list_indexes().names():
        print(f" - Index {index_name} not found, skipping.")
        return

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f" - Total vectors: {total}")

    if total == 0:
        print(" - Already empty, skipping.")
        return

    if not confirm:
        ans = input(f"\n - Delete ALL {total} vectors in {index_name}? [y/N]: ").strip().lower()
        if ans != "y":
            print(" - Skipped.")
            return

    print(f" - Deleting all vectors in {index_name}...")
    index.delete(delete_all=True)
    print(" [SUCCESS] Pinecone cleared.")


def main():
    args        = sys.argv[1:]
    neo4j_only  = "--neo4j-only"    in args
    weaviate_only = "--weaviate-only" in args
    pinecone_only = "--pinecone-only" in args
    confirm     = "--confirm"       in args

    if not any([neo4j_only, weaviate_only, pinecone_only]):
        # Clear all
        clear_neo4j(confirm)
        try:
            clear_weaviate(confirm)
        except Exception as e:
            print(f" [!] Skipping Weaviate: {e}")
        clear_pinecone(confirm)
    elif neo4j_only:
        clear_neo4j(confirm)
    elif weaviate_only:
        clear_weaviate(confirm)
    elif pinecone_only:
        clear_pinecone(confirm)


if __name__ == "__main__":
    main()