import pika
from flask import Flask, request, jsonify
import threading
import json
import time
import logging
import sys
import requests 

# Configure the logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MUSIC_SERVER = "http://222.107.238.124:8503"
ALBUM_SERVER = "https://k8a504.store/album"

app = Flask(__name__)

def handle_message(ch, method, properties, body):
    logging.info("handle message called!")
    sys.stdout.flush()
    # Check if the GPU server is available to process new requests
    if is_gpu_available():
        # Process the message using the GPU server
        process_message(method,properties, body,ch)
        # Acknowledge the message

def start_consuming():
    logging.info("Starting to consume messages from request queue...")
    sys.stdout.flush()
    credentials = pika.PlainCredentials(username="admin", password="cotzjavhwj504")
    parameters = pika.ConnectionParameters("k8a504.p.ssafy.io", 4002, '/', credentials=credentials, heartbeat=0)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Declare the main queue that the GPU server will consume messages from
    channel.queue_declare(queue='request.queue', durable=True)
    channel.queue_declare(queue='response.queue', durable=True)

    # Set the prefetch count to limit the number of unacknowledged messages
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(queue='request.queue', auto_ack= False, on_message_callback=handle_message)
    channel.start_consuming()

# def stop_consuming():
#     logging.info("Stopping to consume messages from request queue...")
#     channel.stop_consuming()

def is_gpu_available():
    # Your logic to check if the GPU server is available goes here
    # Return True if the GPU server is available, False otherwise
    return True

def process_message(method,props,message,ch):
    data = {}
    data['correlation_id'] = props.correlation_id
    data['message'] = message.decode('utf8')
    logging.info(data)
    sys.stdout.flush()
    # data = json.dumps(data)

    # Your logic to process the message using the GPU server goes here
    if method.routing_key == "diffusion.key":
        logging.info("Diffusion message consumed!")
        sys.stdout.flush()
        response = requests.get(f"{ALBUM_SERVER}/api/generate?prompt={data['message']}", timeout=500)
        album_data = response.content
        logging.info("publish response message")
        ch.basic_publish(properties=pika.BasicProperties(correlation_id = props.correlation_id),  exchange='music.fanout', routing_key= props.reply_to, body=album_data)
        ch.basic_ack(delivery_tag = method.delivery_tag)
        # start_consuming()
    else:
        logging.info("Riffusion message consumed!")
        sys.stdout.flush()
        logging.info(message)
        # time.sleep(10)
        response = requests.post(f"{MUSIC_SERVER}/riff", data = data['message'], timeout=500)
        music_data = response.content
        logging.info("publish response message")
        ch.basic_publish(properties=pika.BasicProperties(correlation_id = props.correlation_id),  exchange='music.fanout', routing_key= props.reply_to, body=music_data)
        ch.basic_ack(delivery_tag = method.delivery_tag)
        # start_consuming()
    

@app.route("/response", methods=['Post'])
def handle_response():
    data = request.get_json()
    print(data)
    message = json.dumps(data)
    channel.basic_publish(exchange='music.fanout', routing_key="", body=message)
    return "response handled"

# @channel.basic_consume(queue='availability_queue')
# def handle_availability_message(ch, method, properties, body):
#     if is_gpu_available():
#         start_consuming() 

@app.route("/")
def hello():
    return jsonify({"Message":"This is your flask app with docker"})

if __name__ == "__main__":
    print("start gateway")
    t = threading.Thread(target=start_consuming)
    t.start()
    app.run(host="0.0.0.0", port=8503, debug=False)