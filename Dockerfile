FROM jupyter/datascience-notebook:latest

# Install extra Python packages
RUN pip install \
    pandas \
    scikit-learn \
    seaborn \
    matplotlib \
    duckdb \
    xgboost \
    lightgbm

# Install extra R packages
RUN R -e "install.packages(c('tidyverse', 'data.table', 'lubridate', 'caret', 'randomForest'), repos='https://cloud.r-project.org/')"