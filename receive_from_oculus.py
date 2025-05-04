import os  # create folder.
import asyncio
import struct  # parse bytes.
import numpy as np
import cv2
import csv
import uuid  # generate random name.
from datetime import datetime  # generate timestamp.
from collections import namedtuple  # define packet structure.
import itertools  # dict to list for csv writing.


'''
Global Configuration
'''
config = {
    'save_csv': True,
    'save_raw_image': True,
    'save_rect_image': True,
}


'''
Provide CSV filename and path.
'''
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
random_code = uuid.uuid4().hex[:6]
csv_filename = f"{timestamp}_{random_code}.csv"
os.makedirs(os.path.splitext(csv_filename)[0], exist_ok=True)


'''
Define several lists according to official doc.

status_message: received from oculus via UDP broadcast.
simple_fire_V2_request:
'''
FieldDef = namedtuple(
    'FieldDef', ['name', 'fmt', 'offset', 'size', 'units', 'desc'])

status_message = [
    FieldDef('oculusId',      '<H', 0,   2, 0x4F53, 'Identifier for messages'),
    FieldDef('sourceId',      '<H', 2,   2,
             None,   'Message source identifier'),
    FieldDef('destinationId', '<H', 4,   2, None,
             'Message destination identifier'),
    FieldDef('messaged',      '<H', 6,   2, None,
             'Message ID (section 2.1.1)'),
    FieldDef('version',       '<H', 8,   2, None,   'Message version'),
    FieldDef('payloadSize',   '<I', 10,  4, None,   'Payload size in bytes'),
    FieldDef('partNumber',    '<H', 14,  2, None,   'Subsea part number'),
    FieldDef('deviceId',      '<I',  16,  4,  None, 'Oculus serial number'),
    FieldDef('deviceType',    '<H',  20,  2,  None, 'Device type'),
    FieldDef('partNumber',    '<H',  22,  2,  None, 'Part number'),
    FieldDef('status',        '<I',  24,  4,  None, 'Status flags'),
    FieldDef('version0',      '<I',  28,  4,  None, 'ARM0 version'),
    FieldDef('date0',         '<I',  32,  4,  None, 'ARM0 build date'),
    FieldDef('version1',      '<I',  36,  4,  None, 'ARM1 version'),
    FieldDef('date1',         '<I',  40,  4,  None, 'ARM1 build date'),
    FieldDef('version2',      '<I',  44,  4,  None, 'Core version'),
    FieldDef('date2',         '<I',  48,  4,  None, 'Core build date'),
    FieldDef('ipAddr',        '<I',  52,  4,  None, 'Network IP'),
    FieldDef('ipMask',        '<I',  56,  4,  None, 'Network mask'),
    FieldDef('clientAddr',    '<I',  60,  4,  None, 'Client IP'),
    FieldDef('macAddr',       '6B', 64,  6,  None, 'MAC address'),
    FieldDef('temperatures',  '8d', 70,  64, '°C', 'Temperatures'),
    FieldDef('pressure',      'd', 134, 8,  'bar', 'Pressure rating')
]

def get_bit_flags():
    flags = [
        1,  # RangeInMetres: 1: range in metres, 0: range in percent.
        1,  # 16BitImg: 1: 16 bit image, 0: 8 bit image.
        0,  # GainSend: 1: send gain value, 0: do not send gain value.
        1,  # SimpleReturn: 1: simple return, 0: full return.
        0,  # GainAssist: 1: gain assist, 0: no gain assist.
        0,  # LowPower: 1: low power mode, 0: normal mode.
        1,  # FullBeams: 1: 512 beams, 0: 256 beams.
        0,  # NetworkTrigger: 1: fires when instructed, 0: fires automatically.
    ]

    return sum(bit << i for i, bit in enumerate(flags))

def get_simple_fire_V2_request_bytes():
    data = [
        # 16 bytes, message header.
        0x4F53,  # 2 bytes, default.
        40474,  # 2 bytes, serial number.
        52100,  # 2 bytes, destination port.
        21,  # 2 bytes, message identifier, 21 means simple fire request.
        2,  # 2 bytes, version of message.
        73,  # 4 bytes, payload size.
        1032,  # 2 bytes, part number.

        2,   # 1 byte, current run mode 1: LowFrequency, 2: HighFrequency, 3:PingerLocator.
        2,  # 1 byte, maximum ping rate 0: 10Hz Max, 1: 15Hz Max, 2: 40Hz Max, 3: 5Hz Max, 4: 2Hz Max, 5: Pinging disabled.
        100, # 1 byte, maximum network speed.
        127, # 1 byte, gamma correction value.
        get_bit_flags(), # 1 byte, bit flags.
        1,     # 8 bytes, demanded range in metres or percent.
        75.0,   # 8 bytes, demanded gain value in percent.
        1533.0, # 8 bytes, speed-of-sound used.
        35.0,   # 8 bytes, salinity of the environment.
        0xCAFEBABE, # 4 bytes, extended flags (section 0).
        0, 0,   # 8 bytes, reserved for future use.
        0,      # 4 bytes, the frequency of the pinger beacon.
        0, 0, 0, 0, 0  # 20 bytes, reserved for future use.
    ]

    packed = struct.pack(
        '<5H1I1H5B4d9I',
        *data[0:5],
        data[5],
        data[6],
        *data[7:12],
        *data[12:16],
        *data[16:25],
    )

    return packed

simple_ping_V2_result = [
    FieldDef('fireMessage',    '89s',   0,   89,  '-',   'Simple fire message struct'),
    FieldDef('pingId',         '<I',   89,   4,   '-',   'Ping sequence number'),
    FieldDef('reserved',       '<I',   93,   4,   '-',   'Reserved for future use'),
    FieldDef('frequency',      '<d',   97,   8,   'Hz',  'Current acoustic frequency'),
    FieldDef('temperature',    '<d',   105,  8,   '°C',  'External water temperature'),
    FieldDef('pressure',       '<d',   113,  8,   'bar', 'External environment pressure'),
    FieldDef('heading',        '<d',   121,  8,   '°',   'Heading of the sonar'),
    FieldDef('pitch',          '<d',   129,  8,   '°',   'Pitch of the sonar'),
    FieldDef('roll',           '<d',   137,  8,   '°',   'Roll of the sonar'),
    FieldDef('speedOfSound',   '<d',   145,  8,   'm/s', 'Speed-of-sound used'),
    FieldDef('pingStartTime',  '<d',   153,  8,   '-',   'Timestamp of the ping'),
    FieldDef('dataSize',       'B',    161,  1,   '-',   'Data item size'),
    FieldDef('rangeResolution','<d',   162,  8,   'M',   'Resolution of a single range line'),
    FieldDef('rangeCount',     '<H',   170,  2,   '-',   'Number of range lines in the image'),
    FieldDef('bearingCount',   '<H',   172,  2,   '-',   'Number of bearings in the image'),
    FieldDef('reserved[4]',    '4I',   174,  16,  '-',   'Reserved for future use'),
    FieldDef('imageOffset',    '<I',   190,  4,   '-',   'Offset to the image data'),
    FieldDef('imageSize',      '<I',   194,  4,   'bytes', 'Total size of the image'),
    FieldDef('messageSize',    '<I',   198,  4,   'bytes', 'Total size of the network payload'),
    FieldDef('bearings',       '{}h'.format('bearingCount'), 202, 'm*2', '-', 'Array of actual bearings used'),
    FieldDef('payload',        '{}B'.format('imageSize'),    'imageOffset', 'n', '-', 'Image data payload')
]


class PacketParser:
    '''
    PacketParser: a class to parse the received UDP broadcast and TCP data.
    '''
    @staticmethod
    def status_message(data: bytes) -> dict:
        result = {}
        for field in status_message:
            try:
                unpacked = struct.unpack_from(field.fmt, data, field.offset)

                # post process unpacked data.
                if field.name == 'macAddr':
                    value = ":".join(f"{x:02x}" for x in unpacked)  # MAC to string.
                elif 'ipAddr' in field.name:
                    value = PacketParser._int_to_ip(unpacked[0])     # IP to decimal.
                elif len(unpacked) == 1:
                    value = unpacked[0]
                else:
                    value = list(unpacked)  # array to list.

                result[field.name] = {
                    'value': value,
                    'units': field.units,
                    'desc': field.desc
                }
            except struct.error as e:
                print(f"fail to parse: {field.name} - {str(e)}")
                result[field.name] = None
        return result

    @staticmethod
    def _int_to_ip(ip_int: int) -> str:
        return ".".join(str((ip_int >> (8 * i)) & 0xFF) for i in range(0, 4))

    @staticmethod
    def parse_simple_ping_V2_result(data: bytes) -> dict:
        result = {}
        for field in simple_ping_V2_result:
            try:
                if field.name != 'bearings' and field.name != 'payload':
                    unpacked = struct.unpack_from(field.fmt, data, field.offset)
                else:
                    continue

                # post process unpacked data.
                if field.name == 'fireMessage':
                    value = "fire message"
                elif len(unpacked) == 1:
                    value = unpacked[0]
                else:
                    value = list(unpacked)

                result[field.name] = {
                    'value': value,
                    'units': field.units,
                    'desc': field.desc
                }
            except struct.error as e:
                print(f"fail to parse: {field.name} - {str(e)}")
                result[field.name] = None
        return result


class UDPBroadcastProtocol(asyncio.DatagramProtocol):
    def __init__(self, udp_callback=None):
        super().__init__()
        self.udp_callback = udp_callback
        self.connected = False

    def datagram_received(self, data, addr):
        parsed = PacketParser.status_message(data)
        # self.print_packet(parsed, addr)
        if not self.connected and self.udp_callback:
            ip_address = addr[0]
            asyncio.create_task(self.udp_callback(ip_address))
            self.connected = True

    def print_packet(self, parsed: dict, addr: tuple):
        print(f"\nReceived from {addr}:")
        for name, info in parsed.items():
            if info is None:
                continue
            print(f"{name:15} ({info['units'] or 'N/A'}): {info['value']}")


class TCPClientProtocol:
    def __init__(self):
        self.reader = None
        self.writer = None
        self.is_connected = False
        self.header = b'\x53\x4F'  # oculus return little endian.
        self.buffer = bytearray()

    async def send_data(self, ip):
        try:
            if not self.is_connected:
                self.reader, self.writer = await asyncio.open_connection(
                    ip, 52100
                )
                self.is_connected = True
                print(f"TCP connected to {ip}:52100")
                asyncio.create_task(self.receive_loop())
        except Exception as e:
            print(f"TCP connection failed: {e}")

        await self._send_data(get_simple_fire_V2_request_bytes())

    async def _send_data(self, data):
        if self.is_connected:
            self.writer.write(data)
            await self.writer.drain()

    async def receive_loop(self):
        try:
            while self.is_connected:
                data = await self.reader.read(1024)
                if not data:
                    break
                self.buffer.extend(data)
                header_pos = self.buffer.find(self.header)
                if header_pos != -1:
                    if len(self.buffer) >= header_pos + 16:
                        if self.buffer[header_pos + 6] == 0x23:
                            payload_size = int.from_bytes(self.buffer[header_pos + 10:header_pos + 14], 'little')
                            if len(self.buffer) >= 16 + payload_size:
                                asyncio.create_task(DataProcessor.process(self.buffer[header_pos:header_pos + 16 + payload_size]))
                                del self.buffer[:header_pos + 16 + payload_size]
                            del self.buffer[:header_pos]
                        else:
                            self.buffer.clear()
        except ConnectionResetError:
            print("TCP connection closed by peer")
        finally:
            self.close()

    async def close(self):
        if self.is_connected:
            self.writer.close()
            await self.writer.wait_closed()
            self.is_connected = False
            print("TCP connection closed")


class DataProcessor:
    _LUT = None

    def __init__(self):
        pass

    @staticmethod
    def render_opencv(window_name, image):
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    @staticmethod
    def calc_LUT(image, metadata):
        bearings = metadata['bearings']
        res = metadata['rangeResolution']
        nbins = metadata['rangeCount']
        nbeams = metadata['bearingCount']
        max_range = nbins * res
        leftmost = bearings[0]
        rightmost = bearings[-1]

        nrows = nbins
        y_res = res
        x_meters = max_range * (np.sin(np.deg2rad(rightmost)) - np.sin(np.deg2rad(leftmost)))
        ncols = np.round(x_meters/res).astype(np.uint16)
        x_res = x_meters / ncols

        LUT = np.zeros((nrows, ncols, 2), dtype=np.uint16)

        for row in range(1,nrows+1):
            for col in range(ncols):
                x = (col - (ncols-1)/2) * x_res
                y = row * y_res
                R = np.sqrt(x**2 + y**2)
                angle = np.arctan2(x, y)
                angle = np.degrees(angle)  # oculus uses degrees.

                if R > max_range or angle < leftmost or angle > rightmost:
                    continue

                R_idx = int(np.round(R / res)) - 1  # R_idx meets the requirments: >= 1 and <= nbins, by definition.
                angle_idx = int(np.argmin(np.abs(bearings - angle)))  # TODO: boundary angles should explicitly consider beam witdh.

                LUT[row-1, col, :] = [R_idx, angle_idx]

        DataProcessor._LUT = np.flipud(LUT)


    @staticmethod
    def get_LUT(image, metadata):
        if DataProcessor._LUT is None:
            DataProcessor.calc_LUT(image, metadata)
        return DataProcessor._LUT

    @staticmethod
    def map_by_LUT(image, LUT, background=0):
        if DataProcessor._LUT is None:
            DataProcessor.calc_LUT()
        image[0,0] = background
        return image[LUT[:, :, 0], LUT[:, :, 1]]

    @staticmethod
    async def process(data):
        parsed = PacketParser.parse_simple_ping_V2_result(data)

        # construct metadata.
        metadata = [
            [name, info['value']] for name, info in parsed.items() if info is not None
        ]
        metadata = dict(metadata)

        # process payload.
        payload_pos = metadata['imageOffset']
        payload = data[payload_pos:]
        bin_num = metadata['rangeCount']
        beam_num = metadata['bearingCount']
        dataSize = metadata['dataSize']
        bearing_item_size = 2

        # get bearings for each beam.
        bearings_size = beam_num * bearing_item_size
        bearings = data[202:202 + bearings_size]  # 202 is the packet size of simplefireV2.
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        bearings = np.frombuffer(bearings, dtype=dt)/100
        metadata['bearings'] = bearings

        # get image. (no gain value appended at each beam end.)
        image_payload = payload
        if dataSize == 0:
            dt = np.dtype(np.uint8)
        elif dataSize == 1:
            dt = np.dtype(np.uint16)
        elif dataSize == 2:
            dt = np.dtype(np.uint24)
        else:
            dt = np.dtype(np.uint32)
        dt = dt.newbyteorder('<')
        image_payload = np.frombuffer(image_payload, dtype=dt)
        image = image_payload.reshape(bin_num, beam_num)
        normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        LUT = DataProcessor.get_LUT(normalized, metadata)
        rect = DataProcessor.map_by_LUT(normalized, LUT)
        DataProcessor.render_opencv('normalized raw data', normalized)
        DataProcessor.render_opencv('raw data mapped to Cartesian coordinate', rect)

        # save metadata.
        if config['save_raw_image']:
            raw_image_filename = os.path.join(os.path.splitext(csv_filename)[0], 'raw' + str(metadata['pingId']).zfill(10) + '.png')
            cv2.imwrite(raw_image_filename, normalized)
            metadata['raw_image_path'] = raw_image_filename
        if config['save_rect_image']:
            rect_image_filename = os.path.join(os.path.splitext(csv_filename)[0], 'rect' + str(metadata['pingId']).zfill(10) + '.png')
            cv2.imwrite(rect_image_filename, rect)
            metadata['rect_image_path'] = rect_image_filename
        if config['save_csv']:
            metadata = list(itertools.chain.from_iterable(metadata.items()))
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(metadata)


async def main():
    tcp_client = TCPClientProtocol()
    def udp_callback(ip):
        return tcp_client.send_data(ip)

    udp_transport, udp_protocol = await asyncio.get_running_loop().create_datagram_endpoint(
        lambda: UDPBroadcastProtocol(udp_callback=udp_callback),
        local_addr=('0.0.0.0', 52102)  # 52102 is the default port for Oculus UDP broadcast.
    )

    print("Listening for UDP broadcasts on port 52102...")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        udp_transport.close()


if __name__ == "__main__":
    asyncio.run(main())
