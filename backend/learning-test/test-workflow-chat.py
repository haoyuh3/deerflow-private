from deerflow.client import DeerFlowClient

client = DeerFlowClient()
response = client.chat("介绍一下自己")
print(response)