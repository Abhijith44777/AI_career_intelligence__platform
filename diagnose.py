from database import Database
from vector_store import VectorStore
from knowledge_repository import reindex_all_meetings

db = Database()
store = VectorStore()

meetings = db.list_meetings()
print(f"Meetings in database: {len(meetings)}")
for m in meetings:
    print(f"  id={m['id']}  {m['filename']}  {m['created_at']}")
print(f"Vectors in store BEFORE: {store.count()}  (backend: {store.backend_name})")

print("\nRe-indexing all meetings now...")
results = reindex_all_meetings(db)
for meeting_id, chunk_count in results.items():
    print(f"  meeting {meeting_id}: indexed {chunk_count} chunks")

print(f"Vectors in store AFTER: {store.count()}")