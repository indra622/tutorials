import sounddevice as sd
import websockets
import asyncio
import numpy as np

# 설정
SERVER_URI = "ws://localhost:16399"
SAMPLERATE = 16000  # 16kHz 샘플레이트
BUFFER_DURATION = 2  # 2초 버퍼
CHUNK_DURATION = 1  # 1초마다 서버로 전송
BUFFER_SIZE = int(SAMPLERATE * BUFFER_DURATION)  # 2초 버퍼 크기
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)  # 1초마다 전송할 크기

buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)  # 2초 버퍼

async def send_audio(websocket):
    """마이크 입력을 받아 1초마다 서버로 전송."""
    stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype=np.float32)
    stream.start()
    print("마이크 입력 스트리밍 시작...")

    while True:
        # 1초 분량의 오디오 데이터를 읽어 버퍼에 추가
        audio_chunk = stream.read(CHUNK_SIZE)[0].flatten()
        buffer[:-CHUNK_SIZE] = buffer[CHUNK_SIZE:]  # 앞쪽 데이터 밀기
        buffer[-CHUNK_SIZE:] = audio_chunk  # 새 데이터 추가

        # 서버로 전송
        await websocket.send(buffer.tobytes())
        #print("1초 분량의 오디오 데이터 전송됨.")

        # 1초 대기
        await asyncio.sleep(CHUNK_DURATION)

async def receive_result(websocket):
    """서버에서 인식 결과를 수신."""
    while True:
        try:
            result = await websocket.recv()  # 서버에서 결과 수신
            print(f"서버로부터 받은 인식 결과: {result}")
        except websockets.ConnectionClosed:
            print("서버와의 연결이 종료되었습니다.")
            break

async def main():
    """WebSocket 연결 설정 및 송수신 작업 수행."""
    async with websockets.connect(SERVER_URI) as websocket:
        # 오디오 전송과 결과 수신을 동시에 실행
        send_task = asyncio.create_task(send_audio(websocket))
        receive_task = asyncio.create_task(receive_result(websocket))

        # 두 작업이 끝날 때까지 대기
        await asyncio.gather(send_task, receive_task)

if __name__ == "__main__":
    asyncio.run(main())