import time
import json
import stomp
import queue
import threading
import configparser


class TaskIDListener:
    def __init__(self, config_file="config.ini"):
        # Load configuration from config.ini
        config = configparser.ConfigParser()
        config.read(config_file)

        # Extract credentials and other settings
        self.mb_server_host_port = (config["LPORT"]["host"], int(config["LPORT"]["port"]))
        self.queue_name = config["LPORT"]["queue_name"]
        self.vhost = config["LPORT"]["vhost"]
        self.username = config["credentials"]["username"]
        self.passcode = config["credentials"]["passcode"]

        self.listener_lifetime_sec = 900  # Default lifetime
        self.task_id_queue = queue.Queue()

    def _parse_args(self, args):
        if len(args) == 1:
            frame = args[0]
            return (frame.cmd, frame.headers, frame.body)
        elif len(args) == 2:
            headers, message = args
            return (None, headers, message)

    class MyListener(stomp.ConnectionListener):
        def __init__(self, task_id_callback):
            self.task_id_callback = task_id_callback

        def on_connected(self, frame):
            print('received connected "%s" "%s"' % (frame.headers, frame.body))

        def on_error(self, frame):
            print('received an error "%s"' % frame.body)

        def on_message(self, *args):
            cmd, headers, body = self.task_id_callback._parse_args(args)
            if body:
                self.task_id_callback.read_id(body)

    def read_id(self, msg_json):
        try:
            msg = json.loads(msg_json)
            if msg["msg_type"] == "task_status" and msg["status"] == "defined":
                task_id = msg["taskid"]
                print(f"got task {task_id}")
                self.task_id_queue.put(task_id)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    def listen_for_tasks(self):
        conn = stomp.Connection12([self.mb_server_host_port], vhost=self.vhost)
        conn.set_listener("", self.MyListener(self))
        conn.connect(self.username, self.passcode, wait=True)
        print("connected")
        conn.subscribe(destination=self.queue_name, id=1, ack="auto")
        print("subscribed")

        while True:
            try:
                time.sleep(1)  # Keep listening
            except KeyboardInterrupt:
                break

        conn.disconnect()

    def start_listening(self):
        threading.Thread(target=self.listen_for_tasks).start()

    def get_task_id(self):
        try:
            return self.task_id_queue.get(timeout=1)
        except queue.Empty:
            return None


class ResponseSender:
    def __init__(self, mb_server_host_port, queue_name, vhost, username, passcode):
        self.mb_server_host_port = mb_server_host_port
        self.queue_name = queue_name
        self.vhost = vhost
        self.username = username
        self.passcode = passcode
        self.conn = None

    def connect(self):
        self.conn = stomp.Connection12([self.mb_server_host_port], vhost=self.vhost)
        self.conn.connect(self.username, self.passcode, wait=True)
        print("Connected to the message broker")

    def send_message(self, message):
        if self.conn is None:
            self.connect()

        message_json = json.dumps(message)
        self.conn.send(destination=self.queue_name, body=message_json)
        print(f"Message sent to {self.queue_name}")

    def disconnect(self):
        if self.conn is not None:
            self.conn.disconnect()
            print("Disconnected from the message broker")


# # Example usage:
# if __name__ == "__main__":
#     sender = ResponseSender(
#         mb_server_host_port=('aipanda100.cern.ch', 61613),
#         queue_name='/queue/new_task_notif',
#         vhost='/',
#         username='panda',
#         passcode='panda'
#     )
#
#     message = {
#         'msg_type': 'task_status',
#         'taskid': 123456,  # Example task ID
#         'status': 'defined',
#         'timestamp': int(time.time())
#     }
#
#     sender.send_message(message)
#     sender.disconnect()
#
# # Example usage:
# if __name__ == "__main__":
#     listener = TaskIDListener(
#         mb_server_host_port=('aipanda100.cern.ch', 61613),
#         queue_name='/queue/new_task_notif',
#         vhost='/',
#         username='panda',
#         passcode='panda'
#     )
#     listener.start_listening()
#
#     while True:
#         task_id = listener.get_task_id()
#         if task_id is not None:
#             print(f"Received JEDITASKID: {task_id}")
#             # Process the task ID here...
#         else:
#             pass
