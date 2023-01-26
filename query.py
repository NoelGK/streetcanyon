import os
from datetime import timedelta
import pandas as pd
from azure.kusto.data import KustoClient, ClientRequestProperties, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table


# ADX cluster
cluster = "https://libdataexplorer.eastus.kusto.windows.net"

# AAD keys
client_id = "3dc8b1e3-bbc5-4d2b-b4e5-d754ccf6ee5b"
tenant_id = "6e9a6bfa-6787-47bc-8b83-45132ee8296d"
secret = "W9j8Q~d8DaWjj6-.pLwp4PfwW-YKcOCPIsqxTbt5"

# Create connection string
conn_str = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, client_id, secret, tenant_id)

# Connect to DataBase and make querys
database = "spldata"
query = "one_spl " +\
"| where timestamp >= ago(7d) and timestamp <= now() and deviceid == 'ONE20383534533250190029003F' and isnotnull(LAeq) " +\
"| project timestamp, LAeq " +\
"| summarize LAeq=avg(LAeq) by bin(timestamp, 1h) " +\
"| order by timestamp asc"

with KustoClient(conn_str) as client:
    response = client.execute(database, query)

dframe = dataframe_from_result_table(response.primary_results[0])
save_path = "/home/ngomariz/Escritorio/kusto-querys/dataframe.csv"
dframe.to_csv(save_path, index=False)
