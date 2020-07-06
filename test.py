import json
import requests
from tensorflow import keras


def test_serving():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) \
        = fashion_mnist.load_data()

    test_images = test_images / 255.0

    data = json.dumps({"signature_names": "serving_default",
                       "instances": test_images.tolist()})

    headers = {"content-type": "application/json"}
    url = 'http://serving:8501/v1/models/mnist_model:predict'
    response = requests.post(url,
                             data=data,
                             headers=headers)
    assert response.status_code == 200
    predictions = response.json()['predictions']
    assert len(predictions) == len(test_images)
    assert [len(p) == t.size for p, t in zip(predictions, test_labels)]
