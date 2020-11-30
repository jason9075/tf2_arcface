FROM tensorflow/tensorflow:2.3.1-gpu

ENV PATH="/opt/ml/code:${PATH}"

RUN pip3 install tensorflow_addons==0.11.2 python-dotenv

COPY backend /opt/ml/code/backend
COPY loss_func /opt/ml/code/loss_func
COPY model /opt/ml/code/model
COPY saved_model /opt/ml/code/saved_model
COPY train /opt/ml/code/train
COPY .env /opt/ml/code/.env
COPY convert_tensorflow.py /opt/ml/code/convert_tensorflow.py

WORKDIR /opt/ml/code

ENV SAGEMAKER_PROGRAM train

ENTRYPOINT ["/usr/local/bin/python"]

