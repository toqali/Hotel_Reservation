# 1. Base Image
FROM python : latest

# 2. Create work directory inside contained
WORKDIR /app 

# 3. create copy of each files into container
COPY . .

# 4️. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️. Flask application port
EXPOSE 5000

# 6️. Commands to run the application
CMD ["python", "app.py"]