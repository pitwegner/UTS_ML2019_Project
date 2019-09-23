FROM jupyter/tensorflow-notebook

RUN conda install --quiet --yes \
    pytorch-cpu torchvision-cpu -c pytorch && \
    conda install --quiet --yes \
    -c plotly plotly && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

