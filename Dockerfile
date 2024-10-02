FROM python:3.9

# Instalar MLflow e dependências
RUN pip install mlflow

# Expor a porta padrão do MLflow
EXPOSE 5000

# Definir o comando padrão ao iniciar o container
CMD ["mlflow", "server", "--host", "0.0.0.0"]

