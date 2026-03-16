"""
VITS data_utils.py for Nepali TTS
Adapted for multi-speaker with phoneme-based training

Key Adaptations:
1. Handles filelist format: audio_path|speaker_id|phoneme_text
2. Uses multi-character phoneme tokenization
3. Speaker-aware data loading
"""

import os
import random
import numpy as np
import torch
import torch.utils.data
from scipy.io.wavfile import read

import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    Multi-speaker dataset loader for VITS with phoneme-based text
    
    Filelist format: audio_path|speaker_id|phoneme_text
    Example: data/processed/Voice001.wav|1|nəməskaːr
    """
    
    def __init__(self, audiopaths_sid_text, hparams):
        """
        Args:
            audiopaths_sid_text: List of [audiopath, speaker_id, text]
            hparams: Hyperparameters from config
        """
        self.audiopaths_sid_text = audiopaths_sid_text
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        
        self._filter()
    
    
    def _filter(self):
        """
        Filter items based on text and audio length
        """
        audiopaths_sid_text_new = []
        lengths = []
        
        for audiopath, sid, text in self.audiopaths_sid_text:
            # Filter by text length
            if self.min_text_len <= len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
    
    
    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        """
        Load and process a single sample
        
        Returns:
            text: Phoneme sequence (torch.LongTensor)
            spec: Mel-spectrogram (torch.FloatTensor)
            wav: Audio waveform (torch.FloatTensor)
            sid: Speaker ID (int)
        """
        # Unpack
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        
        # Process text
        text = self.get_text(text)
        
        # Load audio
        spec, wav = self.get_audio(audiopath)
        
        # Speaker ID
        sid = self.get_sid(sid)
        
        return (text, spec, wav, sid)
    
    
    def get_audio(self, filename):
        """
        Load audio and compute mel-spectrogram
        
        Args:
            filename: Path to audio file
            
        Returns:
            spec: Mel-spectrogram
            audio_norm: Normalized audio waveform
        """
        audio, sampling_rate = load_wav_to_torch(filename)
        
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename}: Sample rate {sampling_rate} doesn't match "
                f"config {self.sampling_rate}"
            )
        
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        
        spec_filename = filename.replace(".wav", ".spec.pt")
        
        # Try to load cached spectrogram
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename)
            except:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        
        return spec, audio_norm
    
    
    def get_text(self, text):
        """
        Convert phoneme text to sequence of IDs
        
        Args:
            text: Phoneme text (IPA)
            
        Returns:
            Phoneme sequence (torch.LongTensor)
        """
        if self.cleaned_text:
            # Text is already cleaned (our case)
            text_norm = cleaned_text_to_sequence(text, add_blank=self.add_blank)
        else:
            # Apply cleaners
            text_norm = text_to_sequence(text, self.text_cleaners, add_blank=self.add_blank)
        
        return torch.LongTensor(text_norm)
    
    
    def get_sid(self, sid):
        """
        Convert speaker ID to tensor
        
        Args:
            sid: Speaker ID (int or str)
            
        Returns:
            Speaker ID tensor
        """
        sid = torch.LongTensor([int(sid)])
        return sid
    
    
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])
    
    
    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """
    Collate function for batching
    Zero-pads model inputs and targets
    """
    
    def __init__(self, return_ids=False):
        self.return_ids = return_ids
    
    
    def __call__(self, batch):
        """
        Collate batch of samples
        
        Args:
            batch: List of (text, spec, wav, sid)
            
        Returns:
            Batched and padded tensors
        """
        # Right zero-pad all one-hot text sequences to max length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0,
            descending=True
        )
        
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            
            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            
            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            sid[i] = row[3]
        
        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                ids_sorted_decreasing
            )
        
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch for efficiency
    Adapted for multi-speaker setup
    """
    
    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
    
    
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
        
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        
        return buckets, num_samples_per_bucket
    
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
        
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            
            # Add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )
            
            # Subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]
            
            # Batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)
        
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        
        self.batches = batches
        
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
    
    
    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1
    
    
    def __len__(self):
        return self.num_samples // self.batch_size


def load_filepaths_and_text_nepali(filename):
    """
    Load filelist for Nepali multi-speaker format
    
    Format: audio_path|speaker_id|phoneme_text
    
    Args:
        filename: Path to filelist
        
    Returns:
        List of [audiopath, speaker_id, text]
    """
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                audiopath, sid, text = parts
                audiopath = f"data/{audiopath}"
                filepaths_and_text.append([audiopath, sid, text])
            else:
                print(f"Warning: Skipping invalid line: {line.strip()}")
        
    print(f"Loaded {len(filepaths_and_text)} samples from {filename}")
    return filepaths_and_text


# Export main functions
__all__ = [
    'TextAudioSpeakerLoader',
    'TextAudioSpeakerCollate',
    'DistributedBucketSampler',
    'load_filepaths_and_text_nepali'
]