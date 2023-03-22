from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
from melodygenerator import MelodyGenerator
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import asyncio

IP_SC = "127.0.0.1"
PORT_SC = 6448
PORT_RNN = 6449
OUTPUT_LEN = 64
TEMPERATURE = 1.0

client = SimpleUDPClient(IP_SC, PORT_SC)
mg = MelodyGenerator()

def generate_melody(address: str, *args: List[Any]) -> None:
    # 
    print(f"Adress {address} values: {args}")
    seed = ''.join(args[0])
    print(seed)
    melody = mg.generate_melody(seed, OUTPUT_LEN, SEQUENCE_LENGTH, TEMPERATURE)
    print(melody)
    client.send_message("/sc/input", melody)

dispatcher = Dispatcher()
dispatcher.map("/rnn/input", generate_melody)


async def loop():
    """keeps the listening thread alive."""
    count = 0
    while True:
        print(f'{count} sleep(1)')
        count += 1
        await asyncio.sleep(1)
        
            
async def main():
    server = AsyncIOOSCUDPServer(
        (IP_SC, PORT_RNN), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()
    
    await loop()
    transport.close()

asyncio.run(main())