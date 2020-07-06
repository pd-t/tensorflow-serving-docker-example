FROM tensorflow/tensorflow:latest AS interpreter
RUN pip install pytest requests

FROM interpreter as training
COPY ./train.py /app/
WORKDIR /app
RUN python train.py

FROM tensorflow/serving:latest AS tensorflow-serving
COPY --from=training /app/mnist_model /models/mnist_model/1
ENV MODEL_NAME=mnist_model
EXPOSE 8501
