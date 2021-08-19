from vantage6.tools.mock_client import ClientMockProtocol



client = ClientMockProtocol(
    datasets=["./local/sentiment_train_test_header_1.csv","./local/sentiment_train_test_header_2.csv"],
    module="v6-fedavg-sentiment140"
)

master_task = client.create_new_task(
    input_={
                "master": 1,
                "method":"master",
                'kwargs': {
                    'ids':[0],
                    'epoch_per_round':1
                }


    }, organization_ids=[0,1])

results = client.get_results(master_task.get("id"))
print(results)
r = 0
for result in results:
    print(result)
    r =r + result

print('averaging result: ', r/len(results))