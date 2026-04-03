import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from deerflow.client import DeerFlowClient

client = DeerFlowClient(agent_name="backend-programmer")
response = client.chat("介绍一下自己")
print(response)
