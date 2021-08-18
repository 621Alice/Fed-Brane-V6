from vantage6.tools.mock_client import ClientMockProtocol



client = ClientMockProtocol(
    datasets=["./local/sentiment_train_test_header_1.csv"],
    module="v6-fedavg-sentiment140"
)

master_task = client.create_new_task(
    input_={
                "master": 1,
                "method":"master",
                'kwargs': {
                    'ids':0
                }


    }, organization_ids=[0])

results = client.get_results(master_task.get("id"))
print(results)
