FROM python:2.7

WORKDIR /stars

ADD . /stars

RUN pip install -r requirements.txt
# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

ENV NAME World

CMD ["python", "stars.py"]
