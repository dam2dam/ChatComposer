FROM python:3.9-slim-buster
WORKDIR /app

RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 3000
COPY . .
CMD [ "python", "gateway.py" ]
# CMD ["gunicorn", "--bind", "0.0.0.0:3000", "gateway:app"]