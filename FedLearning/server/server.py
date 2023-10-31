import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:5010", 
    config=fl.server.ServerConfig(num_rounds=4)
)