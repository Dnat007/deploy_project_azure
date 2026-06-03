FROM python:3.10-slim-buster
WORKDIR /service 
COPY requirements.txt .
COPY . ./                 
RUN pip install -r requirements.txt
EXPOSE  5000
# ENTRYPOINT ["python3","app.py"]
CMD ["gunicorn","--bind","0.0.0.0:5000","app:app"]