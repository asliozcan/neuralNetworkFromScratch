FROM python:2.7.16-stretch
# install dependencies
RUN apt update && apt install python-pip -y

# install pip requirements
COPY requirements.txt /
RUN pip install -r requirements.txt

# install pip dev requirements
COPY requirements.dev.txt /
RUN pip install -r requirements.dev.txt

# log volumes
VOLUME ["/log"]
# copy app and run
COPY . /myModule
WORKDIR /myModule

CMD ["pytest" ]