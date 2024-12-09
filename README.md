### Machine Learning Deployment Pipeline**  

This project demonstrates the deployment of a Machine Learning (ML) model pipeline with functionalities for model prediction, retraining, bulk data uploads, and more. The deployment includes dockerized web applications for both the frontend and backend, hosted on cloud platforms. The dockerized application was deployed on google cloup platform and it uses the the google cloud run application. 

## Features  
1. **Model Prediction**  
   - Predict on a single data point (e.g., an image or selected CSV features).  

2. **Data Visualization**  
   - Interpret and visualize at least three features in the dataset with meaningful stories.  

3. **Data Upload**  
   - Bulk upload data (CSV, images, or other formats) for retraining.  

4. **Model Retraining**  
   - Trigger a model retraining process via a user-friendly interface.  

5. **Flood Simulation**  
   - Simulate requests using Locust to evaluate response time and latency under different loads.  


## Deployment Details  

- **Frontend**: [Deployed Frontend URL](https://edupred-fn.onrender.com)  
- **Backend**: [Deployed Backend URL](https://new-flask-app-592896761758.us-east4.run.app/)  
- **Docker Image**: [DockerHub Link](https://hub.docker.com/r/kennyg37/edupred)  
- **Video Folder**: [Google Drive Link](https://drive.google.com/drive/folders/1cQYNznaFOkISp3LeiaqEULL9giL77I-9?usp=drive_link)  


## Steps to Run the Application  

### Using the Docker Image  

1. **Install Docker**  
   - Ensure Docker is installed on your machine. If not, download it from [Docker Official Website](https://www.docker.com).  

2. **Pull the Docker Image**  
   ```bash
   docker pull your-docker-image-name
   ```  

3. **Run the Container**  
   ```bash
   docker run -d -p 80:80 your-docker-image-name
   ```  

4. **Access the Application**  
   - Open your browser and navigate to `http://localhost` to interact with the app.  


## **Steps to Build Locally**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/edupred.git
cd project_name
```

### **2. Set Up a Python Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Run the Application Locally**  
```bash
python app.py
```
Navigate to `http://localhost:5000` to access the application.

---

## **Directory Structure**  

```
Project_name/
│
├── README.md              # Project description and setup instructions
│
├── notebook/
│   ├── project_name.ipynb # Preprocessing, model training, and evaluation
│
├── src/
│   ├── preprocessing.py   # Preprocessing logic
│   ├── model.py           # Model training and evaluation
│   └── prediction.py      # Model prediction logic
│
├── data/
│   ├── train/             # Training dataset
│   └── test/              # Testing dataset
│
└── models/
    ├── model_name.pkl     # Saved model in Pickle format
    └── model_name.tf      # Saved model in TensorFlow format
```

---

## Flood Request Simulation Results  

The application was tested under high traffic using Locust to simulate floods of requests.  



## **Requirements**  
- Python 3.10 or later.  
- Docker installed.  
- Dependencies specified in `requirements.txt`.  

---

## **Results**  
### **Model Evaluation**  
- Metrics: Accuracy, Precision, Recall, F1-Score.  
