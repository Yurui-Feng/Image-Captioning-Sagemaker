# Image Captioning Sagemaker App 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CD Pipeline](https://github.com/Yurui-Feng/Image-Captioning-Sagemaker/actions/workflows/deploy.yml/badge.svg)](https://github.com/Yurui-Feng/Image-Captioning-Sagemaker/actions/workflows/deploy.yml) [![Flask CI](https://github.com/Yurui-Feng/Image-Captioning-Sagemaker/actions/workflows/ci.yml/badge.svg)](https://github.com/Yurui-Feng/Image-Captioning-Sagemaker/actions/workflows/ci.yml)

This application deploys a Huggingface Image-to-text pretrained model on AWS SageMaker and provides a Flask-based web interface to caption images. The web application is capable of handling both URL and uploaded images.

## Prerequisites
- AWS Account with SageMaker and Elastic Beanstalk permissions
- AWS CLI installed and configured with user credentials
- Git LFS: On Mac, you can install it using `brew install git-lfs`
- Python3 with Flask, Boto3, SageMaker, Pillow and Requests installed

## Deployment Steps
1. Clone the repository and navigate to the project directory.

2. Prepare the model for deployment:
    - Create a folder called `code` in the project directory.
    - Add `inference.py` and `requirements.txt` to the `code` folder.
    - Create a `model.tar.gz` file with the layout specified in the [Huggingface SageMaker inference documentation](https://huggingface.co/docs/sagemaker/inference#user-defined-code-and-modules).
```
    model.tar.gz/
    |- pytorch_model.bin
    |- ....
    |- code/
      |- inference.py
      |- requirements.txt 
```

3. Upload the model to S3 using AWS CLI: `aws s3 cp model.tar.gz s3://<your-bucket-name>`

4. Deploy the model on SageMaker. You can follow the instructions in the SageMaker notebook provided in this repository. Alternatively, you can load the deployed model using the following Python code:

    ```python
    from sagemaker import Session
    from sagemaker.huggingface.model import HuggingFacePredictor
    
    sagemaker_session = Session()
    predictor = HuggingFacePredictor(
        endpoint_name="<endpoint-name>",
        sagemaker_session=sagemaker_session
    )
    ```

5. Set up the Flask application:
    - Create `application.py` and `index.html` in the `templates` folder.
    - Test the application locally by running `flask run` and visiting `http://127.0.0.1:5000/`.

6. Prepare the application for Elastic Beanstalk:
    - Rename `app.py` to `application.py`.
    - Create a `requirements.txt` file listing all the necessary Python packages.
    - Create a `.ebextensions` directory and a `01_flask.config` file inside it with the following content:

        ```python
        option_settings:
          aws:elasticbeanstalk:application:environment:
            PYTHONPATH: "/var/app/current:$PYTHONPATH"
          aws:elasticbeanstalk:container:python:
            WSGIPath: "application:application"
        ```

    - Zip all the application files at the root level of the `flask_app` folder.

Please note that the file structure of your application should look like this:

```
flask_app/
├── application.py
├── requirements.txt
├── templates/
│   └── index.html
└── .ebextensions/
    └── 01_flask.config
``` 

7. Deploy the application on Elastic Beanstalk:
    - Go to the AWS Management Console and select Elastic Beanstalk.
    - Add SageMaker Full Access to `aws-elasticbeanstalk-ec2-role`.
    - Create a new application and select the default VPC.
    - Upload the zip archive created in the previous step and launch the application.
  
8. Setting Environment Variables in Elastic Beanstalk

Once your application is deployed on Elastic Beanstalk, you need to set some environment variables for your application to function properly. 

1. In the Elastic Beanstalk dashboard, navigate to your application.
2. Under the "Software" configuration, click on "Modify".
3. Scroll down to the "Environment properties" section. 

Here, you will need to add the following variables:

1. `AWS_REGION`: The AWS region where your resources are located.
2. `PYTHONPATH`: Should be set automatically
3. `AWS_ACCESS_KEY_ID`: Your AWS Access Key ID for programmatic access.
4. `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Access Key corresponding to the Access Key ID.
5. `SAGEMAKER_ENDPOINT_NAME`: The endpoint name of your deployed SageMaker model.

Your environment variables section should look like this:

![Environment Variables](imgs/envs.png)

Make sure to replace the placeholders with your actual values and click "Apply" to save the changes. 

