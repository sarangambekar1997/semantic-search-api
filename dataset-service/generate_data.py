from faker import Faker
import json
from datetime import datetime, timedelta
fake = Faker()

data = []
for i in range(1, 51):
    created = datetime.now() - timedelta(days=fake.pyint(1, 30))
    data.append({
        "id": i,
        "title": fake.sentence(nb_words=4),
        "description": fake.text(max_nb_chars=200),
        "category": fake.random_element(["Payment", "Login", "Order", "Shipping"]),
        "priority": fake.random_element(["High", "Medium", "Low"]),
        "status": fake.random_element(["Open", "Closed", "Resolved"]),
        "created_at": created.isoformat()
    })

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
print("50 records generated in data.json")
