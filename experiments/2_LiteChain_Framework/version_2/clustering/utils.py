# utils.py

def get_client_id():
    """
    Each Docker container should set CLIENT_ID env variable
    """
    import os
    return os.getenv("CLIENT_ID", "client_unknown")
