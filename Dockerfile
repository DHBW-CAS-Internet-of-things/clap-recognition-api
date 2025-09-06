FROM python:3.12

COPY requirements.txt .
COPY app.py .
COPY cnn.py .
COPY config.py .
COPY best_clap_cnn.pt .

RUN pip install -r ./requirements.txt

CMD python app.py