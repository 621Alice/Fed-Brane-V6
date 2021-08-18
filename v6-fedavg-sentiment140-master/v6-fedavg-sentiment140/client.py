import time
import pickle
from vantage6.client import Client

client = Client(
    host="http://localhost",
    port=5001,
    path="/api"
)

client.authenticate("admin", "password")
client.setup_encryption(None)

# # 2. Prepare input for the summary Docker image (algorithm)
input_ = {
        "master": "true",
        "method": "master",
        "kwargs":{
            'ids':2
        }
        }
# # 3. post the task to the server post_task
task = client.post_task(
    name="sentiment3",
    image="fedavg-sentiment-2",
    collaboration_id=1,
    organization_ids=[2],  # specify where the central container should run! # 4 is the newly created node with the api key that the node config uses
    input_=input_
)

# # 4. poll if central container is finished
task_id = task.get("id") #"id"
print(f"task id={task_id}")

task = client.request(f"task/{task_id}")
while not task.get("complete"):
    task = client.request(f"task/{task_id}")
    print("Waiting for results...")
    time.sleep(1)

# # 5. obtain the finished results
results = client.get_results(task_id=task.get("id"))

# e.g. print the results per node
for result in results:
    print("-"*80)
    print(pickle.loads(result.get("result")))