import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-02-14-16-43-57-027" ## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    ## TODO: fill in
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    ## TODO: fill in
    predictor = sagemaker.Predictor("image-classification-2023-02-14-16-43-57-027")

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    ## TODO: fill in
    inferences = predictor.predict(image, initial_args={'ContentType': 'application/x-image'})
    
    # We return the data back to the Step Function    
    event["inferences"] = [float(i) for i in inferences.decode('utf-8').strip('][').split(', ')]
    return {
        'statusCode': 200,
        'body': event
    }