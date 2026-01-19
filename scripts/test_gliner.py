#!/usr/bin/env python3
"""Test GLiNER for entity extraction on torrent names."""

import time
from gliner import GLiNER

# Test samples - mix of audio and video
AUDIO_SAMPLES = [
    "Pink Floyd - The Dark Side of the Moon (1973) [FLAC]",
    "Led Zeppelin - IV (1971) [24-96 Vinyl Rip]",
    "The Beatles - Abbey Road [320kbps MP3]",
    "Radiohead - OK Computer 1997 FLAC",
    "Miles Davis - Kind of Blue (1959) [DSD256]",
    "Various Artists - Now That's What I Call Music 100 (2018)",
    "Queen - Greatest Hits I II & III The Platinum Collection",
    "Nirvana - Nevermind (20th Anniversary Super Deluxe Edition) [FLAC]",
    "Bob Dylan Discography 1962-2020 [MP3 320]",
    "Daft Punk - Random Access Memories 2013 [24bit Hi-Res]",
]

VIDEO_SAMPLES = [
    "The Shawshank Redemption 1994 1080p BluRay x264",
    "Game of Thrones S01E01 Winter Is Coming 1080p",
    "Breaking Bad S05E16 Felina 720p BluRay x264",
    "Inception 2010 2160p UHD BluRay REMUX HDR HEVC",
    "The Office US S01-S09 Complete 720p WEB-DL",
    "Stranger Things S04 Complete 1080p NF WEB-DL",
    "Pulp Fiction 1994 REMASTERED 1080p BluRay x265",
    "Avatar The Way of Water 2022 2160p WEB-DL DDP5.1 Atmos",
    "The Mandalorian S03E08 Chapter 24 The Return 2160p DSNP WEB-DL",
    "Oppenheimer 2023 1080p WEBRip x264 AAC",
]

def test_audio():
    print("=" * 60)
    print("AUDIO ENTITY EXTRACTION")
    print("=" * 60)

    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    labels = ["artist", "album", "year", "audio format", "bitrate"]

    total_time = 0
    for sample in AUDIO_SAMPLES:
        start = time.perf_counter()
        entities = model.predict_entities(sample, labels, threshold=0.3)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        print(f"\nInput: {sample}")
        print(f"Time: {elapsed*1000:.1f}ms")
        for ent in entities:
            print(f"  {ent['label']:12} → {ent['text']!r} (score: {ent['score']:.2f})")

    print(f"\nAvg time: {total_time/len(AUDIO_SAMPLES)*1000:.1f}ms")

def test_video():
    print("\n" + "=" * 60)
    print("VIDEO ENTITY EXTRACTION")
    print("=" * 60)

    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    labels = ["movie title", "tv show", "season", "episode", "year", "resolution", "video codec"]

    total_time = 0
    for sample in VIDEO_SAMPLES:
        start = time.perf_counter()
        entities = model.predict_entities(sample, labels, threshold=0.3)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        print(f"\nInput: {sample}")
        print(f"Time: {elapsed*1000:.1f}ms")
        for ent in entities:
            print(f"  {ent['label']:12} → {ent['text']!r} (score: {ent['score']:.2f})")

    print(f"\nAvg time: {total_time/len(VIDEO_SAMPLES)*1000:.1f}ms")

if __name__ == "__main__":
    print("Loading GLiNER large model...")
    test_audio()
    test_video()
