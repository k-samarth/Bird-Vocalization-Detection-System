# Bird-Vocalization-Detection-System

### Project prerequisites
- **Python:**  3.12.5 [Used by me while pushing the code]

## Steps to Run the Project
### Create Virtual Env
- ```
  python -m venv env
  ```
- ```
  pip install -r requirements.txt
  ```

### Run the following to directly run the web app
1. Run the following:
    ```
    cd Django\bird_voice
    ```
2. Run in command prompt
    ```
    python manage.py runserver
    ```

### Run the following to run training code and web project
1. Run in command prompt
    ```
    python file_preparator.py
    ```

2. Run all the cells in svm_model.ipynb
3. Run all the cells in Predict.ipynb
4. Run the following:
    ```
    cd Django\bird_voice
    ```
5. Run in command prompt
    ```
    python manage.py runserver
    ```
