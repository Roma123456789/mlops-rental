FROM python:3
# working directory
WORKDIR /mlapp
# copy requirements.txt, app.py and data/rental_1000.csv
COPY . .
# install libraries
RUN pip install --no-cache-dir -r requirements.txt
# Default command to run and start container
CMD [ "python", "./app.py" ]