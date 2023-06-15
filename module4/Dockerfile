FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["score.py", "data/yellow_tripdata_2022-04.parquet", "./"]

CMD ["python", "score.py", "2022", "4"]
