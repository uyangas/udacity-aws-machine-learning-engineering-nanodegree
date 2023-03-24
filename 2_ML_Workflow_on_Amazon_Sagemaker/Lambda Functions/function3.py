import json
import numpy as np

THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    ## TODO: fill in
    inferences = event['body']['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    ## TODO: fill in
    meets_threshold = np.any([inferences[0]>=THRESHOLD, inferences[1]>=THRESHOLD])
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
