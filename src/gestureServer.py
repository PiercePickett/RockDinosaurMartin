import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 8000))

print("Listening for Hand States / Commands...")

while True:
    data, addr = sock.recvfrom(1024)
    msg = data.decode()

    # If it's a 4-bit command
    if len(msg) == 4 and all(c in "01" for c in msg):
        print(f"COMMAND RECEIVED: {msg}")
    else:
        print(f"Gesture: {msg}")