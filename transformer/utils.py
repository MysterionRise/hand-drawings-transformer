import numpy as np
import requests


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return nodes[np.argmin(dist_2)]


def file_name(original, suffix):
    return f"{original.rsplit('.', 1)[0]}-{suffix}.jpeg"


def save_file(filename, data):
    with open(filename, "wb") as f:
        f.write(data)


def upload_file(file, record, token, url):
    data = {
        "record": record,
        "token": token,
    }
    with open(file, "rb") as f:
        files = {"file": f}
        r = requests.post(url, data=data, files=files)
        print(r)
