import nnpy
import struct
import ipaddress
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI
import threading

class DigestController():

    def __init__(self, sw_name):
        self.topo = load_topo('topology.json')
        self.sw_name = sw_name
        self.thrift_port = self.topo.get_thrift_port(sw_name)
        self.controller = SimpleSwitchThriftAPI(self.thrift_port)

    def recv_msg_digest(self, msg):
        topic, device_id, ctx_id, list_id, buffer_id, num = struct.unpack("<iQiiQi",
                                                                     msg[:32])
        print(topic, device_id, ctx_id, list_id, buffer_id)
        print(num, len(msg))
        offset = 9
        msg = msg[32:]
        for sub_message in range(num):
            random_num, src, dst = struct.unpack("!BII", msg[0:offset])
            print("random number:", random_num, "src ip:", str(ipaddress.IPv4Address(src)),
                  "dst ip:", str(ipaddress.IPv4Address(dst)))
            msg = msg[offset:]

        self.controller.client.bm_learning_ack_buffer(ctx_id, list_id, buffer_id)

    def run_digest_loop(self):
        sub = nnpy.Socket(nnpy.AF_SP, nnpy.SUB)
        notifications_socket = self.controller.client.bm_mgmt_get_info().notifications_socket
        print("connecting to notification sub %s" % notifications_socket)
        sub.connect(notifications_socket)
        sub.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, '')

        while True:
            msg = sub.recv()
            self.recv_msg_digest(msg)

def run_digest_controller(sw_name):
    controller = DigestController(sw_name)
    controller.run_digest_loop()

def main():
    switch_names = ["s1", "s2", "s3"]  # Add all switch names here
    threads = []

    for switch_name in switch_names:
        thread = threading.Thread(target=run_digest_controller, args=(switch_name,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()