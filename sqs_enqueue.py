import json
import os

import boto3
from botocore.exceptions import ClientError

boto3.setup_default_session(profile_name='qa')
s3_bucket = 'astra-face-recognition-dataset'
s3_folder = 'divide_tiny/divide/'
total_dataset = 'dataset/cele_cc_2'
current_dataset = 'dataset/cele_cc_2_aug'

def main():
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName='cv-aug-image')

    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(s3_bucket)

    total_dirs = os.listdir(total_dataset)
    total_dirs = [d for d in total_dirs if not d.startswith('.')]
    total_dirs = set(total_dirs)
    print(f'len of total {len(total_dirs)}')

    current_dirs = os.listdir(current_dataset)
    current_dirs = [d for d in current_dirs if not d.startswith('.')]
    current_dirs = set(current_dirs)
    print(f'len of total {len(current_dirs)}')

    unupload_id_list=set()
    for d in total_dirs:
        if d not in current_dirs:
            unupload_id_list.add(d)

    print(f'len of unupload {len(unupload_id_list)}')
    print(unupload_id_list)

    for unupload_id in unupload_id_list:

        message_body = {'bucket': s3_bucket,
                        'folder': f'celebrity/{unupload_id}/'}

        try:
            response = queue.send_message(MessageBody=json.dumps(message_body))
        except ClientError as error:
            raise error


if __name__ == '__main__':
    main()
