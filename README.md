# Facial-Emotion-Recognition

This repo is for a facial emotion recognition system. The project is divided in two main packages, `Frontend` and `Backend` packages. Read the section [Project Structure](#project-structure) for more details.

## Project Structure
We have two main packages here, the `Frontend` represent the user interface and the `backend` is our API and Model.

```
-  main.py       // Entry point of the app
-  packages\
    - frontend\
        - index.html   // The entry point of the user interface 
    - backend\
        - api\
            __init__.py // Entry point of api 
        - model
            __init__.py // Entry point of the model
- README.md 
```

## How to install dependencies
You should activate the `virtualenv` first by running the activate script in the `Scripts` folder by this:
```
    $ .\Script\activate.bat
```
> Note that we are using python 3.9 for this project

After activate the virtualenv, you should install project dependencies with
```
    $ pip install -r requirements.txt
```

## How to run the app ?

### What to check before ?
Before running the app, ensure you have the pretrained model at `packages/backend/model/data/facial_emotion_recognition_model.h5` location.

You can get it at the data location, or you can run the script `train_model.py` to train the model with your own `fer2013` dataset.

### Run the app
To run it, you should just need to go to the root directory of your project (directory that contains your main.py file and your packages). Then use this command:

```
    $ fastapi dev main.py
```
Ensure the development server is running at the `port 8000`, and then go to `http://localhost:8000/app` in your web browser.