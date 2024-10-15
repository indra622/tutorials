import asyncio
import numpy as np
import torch
import websockets
import nemo.collections.asr as nemo_asr

# NeMo 모델 로드 및 CPU 설정
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_medium")
#asr_model.to(torch.device('cpu'))

async def transcribe_audio(websocket, path):
    """WebSocket으로 받은 오디오 데이터를 텍스트로 변환."""
    buffer = np.array([], dtype=np.float32)  # 데이터를 쌓을 버퍼

    try:
        print("클라이언트 연결됨")
        async for message in websocket:
            # 수신된 데이터를 numpy 배열로 변환하고 버퍼에 추가
            audio_data = np.frombuffer(message, dtype=np.float32)
            buffer = np.concatenate((buffer, audio_data))

            # 버퍼가 2초 분량(16000 * 2 샘플)을 넘으면 인식 수행
            if len(buffer) >= 32000:
                print("2초 분량의 데이터 인식 중...")
                transcription = asr_model.transcribe([buffer[:32000]])[0]
                print(f"인식 결과: {transcription}")

                # 인식된 텍스트를 클라이언트로 전송
                await websocket.send(transcription)

                # 버퍼에서 사용한 부분 제거
                buffer = buffer[16000:]

    except websockets.ConnectionClosed:
        print("클라이언트 연결이 종료되었습니다.")

async def main():
    async with websockets.serve(transcribe_audio, "0.0.0.0", 16399):
        print("WebSocket 서버가 시작되었습니다...")
        await asyncio.Future()  # 서버가 종료되지 않도록 대기

if __name__ == "__main__":
    asyncio.run(main())