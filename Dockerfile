FROM tensorflow/tensorflow:2.4.1

RUN pip3 install --upgrade pip
RUN pip3 install pandas
RUN pip3 install psycopg2
RUN pip3 install tensorflowjs