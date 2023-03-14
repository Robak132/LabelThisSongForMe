import argparse
import torchopenl3
from models.common import convert_mp3_to_npy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create embeddings from selected audio file')
    parser.add_argument('filename', type=str, help='Path to music file (MP3)')
    args = parser.parse_args()

    audio = convert_mp3_to_npy(args.filename, sr=16000)
    emb, ts = torchopenl3.get_audio_embedding(audio, 16000, content_type="music", input_repr="mel256",
                                              embedding_size=512, hop_size=1, batch_size=10, sampler="julian")
    print(emb.detach().cpu().numpy())
    print(ts.detach().cpu().numpy())
