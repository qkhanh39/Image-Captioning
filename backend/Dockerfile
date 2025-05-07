FROM python:3.9.20-slim

WORKDIR /image-captioning/backend

COPY ./requirements.txt /image-captioning/backend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /image-captioning/backend/requirements.txt

COPY . .

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]