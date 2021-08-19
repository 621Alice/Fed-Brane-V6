from vantage6.tools.mock_client import ClientMockProtocol



client = ClientMockProtocol(
    datasets=["./local/mnist_train_test_header_1.csv","./local/mnist_train_test_header_2.csv"],
    # datasets=["./local/training.pt"],
    module="v6-fedavg-mnist"
)

# organizations = client.get_organizations_in_my_collaboration()
# org_ids = ids = [organization["id"] for organization in organizations]
# print(org_ids)
# print(type(org_ids))
master_task = client.create_new_task(
    input_={
                "master": 1,
                "method":"master",
                'kwargs': {
                    'ids':[0,1],
                    'epoch_per_round':1
                }


    }, organization_ids=[0,1])
results = client.get_results(master_task.get("id"))
print(results)
