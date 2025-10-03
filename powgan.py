#################################################
# Python module for the PowGAN model            #
#                                               #
# Author: Miguel Marcos                         #
# (Based on the original MuseGAN notebook)      #
#################################################

from os import remove
import numpy as np
import torch
import pypianoroll
from pretty_midi import PrettyMIDI
import scipy.io.wavfile
from pypianoroll import Multitrack, Track, BinaryTrack

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Config

n_tracks = 3  # number of tracks
n_pitches = 72  # number of pitches
lowest_pitch = 24  # MIDI note number of the lowest pitch
programs = [0, 25, 33]  # program number for each track
is_drums = [True, False, False]  # drum indicator for each track
track_names = ['Drums', 'Guitar', 'Bass']  # name of each track

tempo = 100
n_measures = 4  # number of measures per sample
beat_resolution = 4  # temporal resolution of a beat (in timestep)
n_beats_per_measure = 4
measure_resolution = 4 * beat_resolution
tempo_array = np.full((n_measures * n_beats_per_measure * beat_resolution, 1), tempo)
note_thresholds = [0.60828567, 0.55597573, 0.54794814] 
# Values lower than this are considered silence. Note that this value
# directly influences metrics. These three values were obtained via BO
th_tensor = torch.tensor(note_thresholds).view(1,3,1,1)
if torch.cuda.is_available():
  th_tensor = th_tensor.cuda()

latent_dim = 256

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Generator

class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super(GeneratorBlock,self).__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
        self.silu = torch.nn.SiLU()
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        x = self.silu(x)
        return x

class OutputBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super(OutputBlock,self).__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.sig(x)
        return x

class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector for all tracks and a per-track latent vector,
    then outputs a fake sample."""
    def __init__(self):
        super().__init__()        

        # Shared network
        self.transconv0 = GeneratorBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1)) # 4, 1, 1
        self.transconv1 = GeneratorBlock(256, 128, (1, 4, 3), (1, 4, 3)) # 4, 4, 3
        self.transconv2 = GeneratorBlock(128, 64, (1, 2, 2), (1, 2, 2)) # 4, 8, 6

        # Private pitch-time network
        self.pt_transconv3 = torch.nn.ModuleList([
            GeneratorBlock(64, 16, (1, 1, 12), (1, 1, 12)) # 4, 8, 72
            for _ in range(n_tracks)
        ])
        self.pt_transconv4 = torch.nn.ModuleList([
            GeneratorBlock(16, 1, (1, 2, 1), (1, 2, 1)) # 4, 16, 72
            for _ in range(n_tracks)
        ])

        # Private time-pitch network
        self.tp_transconv3 = torch.nn.ModuleList([
            GeneratorBlock(64, 16, (1, 2, 1), (1, 2, 1)) # 4, 16, 6
            for _ in range(n_tracks)
        ])
        self.tp_transconv4 = torch.nn.ModuleList([
            GeneratorBlock(16, 1, (1, 1, 12), (1, 1, 12)) # 4, 16, 72
            for _ in range(n_tracks)
        ])

        # Merge and output
        self.transconv5 = torch.nn.ModuleList([
            OutputBlock(2,1,(1,1,1),(1,1,1)) # 4, 16, 72
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        # Shared network
        x = x.view(-1, 256, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        # Private pitch-time network
        x_pt = [transconv(x) for transconv in self.pt_transconv3]
        x_pt = torch.cat([transconv(x_) for x_, transconv in zip(x_pt, self.pt_transconv4)], 1)
        # Private time-pitch network
        x_tp = [transconv(x) for transconv in self.tp_transconv3]
        x_tp = torch.cat([transconv(x_) for x_, transconv in zip(x_tp, self.tp_transconv4)], 1)
        # Merge and output
        # Probably not the most elegant solution, but it's the only one
        # I could figure out. Tensors and convolutions are really annoying
        x = [torch.cat([torch.unsqueeze(x_pt[:,i],1),
                        torch.unsqueeze(x_tp[:,i],1)],1)
                        for i in range(n_tracks)]

        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)],1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x


# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Critic

class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
        self.silu = torch.nn.SiLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        return self.silu(x)

class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Comments next to layers indicate the shape of the output,
        # in the form <M,T,P> (Measure,Time,Pitch)

        # Mirror the private time-pitch network (so now it's pitch-time)
        self.pt_conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks) # 4, 16, 6
        ])
        self.pt_conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 64, (1, 2, 1), (1, 2, 1)) for _ in range(n_tracks) # 4, 8, 6
        ])
        # Mirror the private pitch-time network (so now it's time-pitch)
        self.tp_conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 2, 1), (1, 2, 1)) for _ in range(n_tracks) # 4, 8, 72
        ])
        self.tp_conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 64, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks) # 4, 8, 6
        ])
        # Mirror the shared network
        self.conv2 = DiscriminatorBlock(64 * n_tracks * 2, 128, (1, 2, 2), (1, 2, 2)) # 4, 4, 3
        self.conv3 = DiscriminatorBlock(128, 256, (1, 4, 3), (1, 4, 3)) # 4, 1, 1

        # Chroma stream
        self.chroma_conv0 = DiscriminatorBlock(n_tracks,64,(1,1,12),(1,1,12)) # 4, 4, 1
        self.chroma_conv1 = DiscriminatorBlock(64,128,(1,4,1),(1,4,1)) # 4, 1, 1

        # Onset/Offset stream
        self.onoff_conv0 = DiscriminatorBlock(n_tracks,64,(1,4,1),(1,4,1)) # 4, 4, 1
        self.onoff_conv1 = DiscriminatorBlock(64,128,(1,4,1),(1,4,1)) # 4, 1, 1

        # Merge streams
        self.conv4 = DiscriminatorBlock(512, 256, (2, 1, 1), (1, 1, 1)) # 3, 1, 1
        self.conv5 = DiscriminatorBlock(256, 256, (3, 1, 1), (3, 1, 1)) # 1, 1, 1

        # Final output
        self.dense1 = torch.nn.Linear(256, 1)

    def forward(self, x):

        # x has shape <B,I,T,P>
        # (Batch, Instrument, Time and Pitch)
        # Instruments are considered channels

        # Extract chroma feature
        chroma = x.view(-1, n_tracks, n_measures, n_beats_per_measure, beat_resolution, n_pitches)
        chroma = torch.sum(chroma,4) # 4, 4, 72
        chroma = chroma.view(-1, n_tracks, n_measures, n_beats_per_measure, n_pitches//12, 12)
        chroma = torch.sum(chroma,4) # 4, 4, 6

        # Extract onset/offset feature
        # Heads-up: PyTorch's padding starts from the last dimension.
        # We want to pad the Time dimension (second to last)
        # (0,0,1,0) means "Don't pad the Pitch dimension, add 1 padding
        # at the top of the Time dimension".
        onoff = torch.nn.functional.pad(x[:,:,:-1],(0,0,1,0))
        onoff = x - onoff
        onoff = onoff.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        onoff = torch.sum(onoff,4,keepdim=True) # 4, 16, 1

        # Compute the private instrument networks
        x = x.view(-1, n_tracks, 1, n_measures, measure_resolution, n_pitches)
        # Pitch-time
        x_pt = [conv(x[:,i]) for i, conv in enumerate(self.pt_conv0)]
        x_pt = torch.cat([conv(x_) for x_, conv in zip(x_pt, self.pt_conv1)], 1)
        # Time-pitch
        x_tp = [conv(x[:,i]) for i, conv in enumerate(self.tp_conv0)]
        x_tp = torch.cat([conv(x_) for x_, conv in zip(x_tp, self.tp_conv1)], 1)
        # Shared network
        x = torch.cat([x_pt,x_tp],1)
        x = self.conv2(x)
        x = self.conv3(x)

        # Chroma stream
        c = self.chroma_conv0(chroma)
        c = self.chroma_conv1(c)

        # Osset/offset stream
        o = self.onoff_conv0(onoff)
        o = self.onoff_conv1(o)

        # Merge streams
        x = torch.cat([x,c,o],1)
        x = self.conv4(x)
        x_f = self.conv5.layernorm(self.conv5.conv(x))
        x = self.conv5.silu(x_f)
        x = x.view(-1, 256)
        x_out = self.dense1(x)
        return x_out, x_f


# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Autoencoder

class Encoder(torch.nn.Module):
      
    def __init__(self):
        super().__init__()

        # Comments next to layers indicate the shape of the output,
        # in the form <M,T,P> (Measure,Time,Pitch)

        # Mirror the private time-pitch network (so now it's pitch-time)
        self.pt_conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks) # 4, 16, 6
        ])
        self.pt_conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 64, (1, 2, 1), (1, 2, 1)) for _ in range(n_tracks) # 4, 8, 6
        ])
        # Mirror the private pitch-time network (so now it's time-pitch)
        self.tp_conv0 = torch.nn.ModuleList([
            DiscriminatorBlock(1, 16, (1, 2, 1), (1, 2, 1)) for _ in range(n_tracks) # 4, 8, 72
        ])
        self.tp_conv1 = torch.nn.ModuleList([
            DiscriminatorBlock(16, 64, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks) # 4, 8, 6
        ])
        # Mirror the shared network
        self.conv2 = DiscriminatorBlock(64 * n_tracks * 2, 128, (1, 2, 2), (1, 2, 2)) # 4, 4, 3
        self.conv3 = DiscriminatorBlock(128, 256, (1, 4, 3), (1, 4, 3)) # 4, 1, 1

        # Chroma stream
        self.chroma_conv0 = DiscriminatorBlock(n_tracks,64,(1,1,12),(1,1,12)) # 4, 4, 1
        self.chroma_conv1 = DiscriminatorBlock(64,128,(1,4,1),(1,4,1)) # 4, 1, 1

        # Onset/Offset stream
        self.onoff_conv0 = DiscriminatorBlock(n_tracks,64,(1,4,1),(1,4,1)) # 4, 4, 1
        self.onoff_conv1 = DiscriminatorBlock(64,128,(1,4,1),(1,4,1)) # 4, 1, 1

        # Merge streams
        self.conv4 = DiscriminatorBlock(512, latent_dim, (2, 1, 1), (1, 1, 1)) # 3, 1, 1
        self.conv5 = DiscriminatorBlock(latent_dim, latent_dim, (3, 1, 1), (3, 1, 1)) # 1, 1, 1

    def forward(self, x):

        # x has shape <B,I,T,P>
        # (Batch, Instrument, Time and Pitch)
        # Instruments are considered channels

        # Extract chroma feature
        chroma = x.view(-1, n_tracks, n_measures, n_beats_per_measure, beat_resolution, n_pitches)
        chroma = torch.sum(chroma,4) # 4, 4, 72
        chroma = chroma.view(-1, n_tracks, n_measures, n_beats_per_measure, n_pitches//12, 12)
        chroma = torch.sum(chroma,4) # 4, 4, 6

        # Extract onset/offset feature
        # Heads-up: PyTorch's padding starts from the last dimension.
        # We want to pad the Time dimension (second to last)
        # (0,0,1,0) means "Don't pad the Pitch dimension, add 1 padding
        # at the top of the Time dimension".
        onoff = torch.nn.functional.pad(x[:,:,:-1],(0,0,1,0))
        onoff = x - onoff
        onoff = onoff.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        onoff = torch.sum(onoff,4,keepdim=True) # 4, 16, 1

        # Compute the private instrument networks
        x = x.view(-1, n_tracks, 1, n_measures, measure_resolution, n_pitches)
        # Pitch-time
        x_pt = [conv(x[:,i]) for i, conv in enumerate(self.pt_conv0)]
        x_pt = torch.cat([conv(x_) for x_, conv in zip(x_pt, self.pt_conv1)], 1)
        # Time-pitch
        x_tp = [conv(x[:,i]) for i, conv in enumerate(self.tp_conv0)]
        x_tp = torch.cat([conv(x_) for x_, conv in zip(x_tp, self.tp_conv1)], 1)
        # Shared network
        x = torch.cat([x_pt,x_tp],1)
        x = self.conv2(x)
        x = self.conv3(x)

        # Chroma stream
        c = self.chroma_conv0(chroma)
        c = self.chroma_conv1(c)

        # Osset/offset stream
        o = self.onoff_conv0(onoff)
        o = self.onoff_conv1(o)

        # Merge streams
        x = torch.cat([x,c,o],1)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, latent_dim)

        return x

class LinearLReLUBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim,output_dim)
        self.batchnorm = torch.nn.BatchNorm1d(output_dim)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.lrelu(self.batchnorm(self.dense(x)))
        return x
    
class VarAutoencoder(torch.nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        self.dense_enc1 =  LinearLReLUBlock(latent_dim,128)
        self.dense_enc2 =  LinearLReLUBlock(128,64)
        self.dense_enc3 =  LinearLReLUBlock(64,32)
        self.dense_enc4 =  LinearLReLUBlock(32,10)
        self.mus = torch.nn.Linear(10,10)
        self.sigmas = torch.nn.Linear(10,10)
        self.dense_dec1 =  LinearLReLUBlock(10,32)
        self.dense_dec2 =  LinearLReLUBlock(32,64)
        self.dense_dec3 =  LinearLReLUBlock(64,128)
        self.dense_dec4 =  torch.nn.Linear(128,latent_dim)

        self.N = torch.distributions.Normal(0,1)
        if torch.cuda.is_available():
          self.N.loc = self.N.loc.cuda()
          self.N.scale = self.N.scale.cuda()

        self.kl = 0

    def forward(self, x):

        #Encode the input
        x = self.get_encoding(x)

        #Extract the distributions
        mu = self.mus(x)
        mu2 = torch.pow(mu,2)
        sigma = self.sigmas(x)
        sigma2 = torch.pow(sigma,2)
        #Sanity check, in case some wild zero appears in sigma2
        sigma2+=1e-6
        #Sample the distributions
        z = mu + sigma*self.N.sample(mu.shape)
        #Compute KL divergence
        self.kl = (mu2 + sigma2 - torch.log(sigma2) - 0.5).mean()

        # Decode the latent space
        x = self.get_decoded_sample(z)
        # Smooth thresholding
        x = torch.sigmoid(10*(x-th_tensor))

        return x

    def get_encoding(self, x):

        x = self.dense_enc1(self.enc(x))
        x = self.dense_enc2(x)
        x = self.dense_enc3(x)
        x = self.dense_enc4(x)

    def get_highdim(self,x):
       
        x = self.dense_dec1(x)
        x = self.dense_dec2(x)
        x = self.dense_dec3(x)
        x = self.dense_dec4(x)

        return x
    
    def get_decoded_sample(self,x):
       
        x = self.dec(self.get_highdim(x))

        return x


# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Sample management

def clip_samples(samples,thresholds):
    samples = torch.transpose(samples,0,1)
    for i in range(n_tracks):
        samples[i,samples[i]<thresholds[i]]=0.0
        samples[i,samples[i]>thresholds[i]]=1.0
    samples = torch.transpose(samples,0,1)
    return samples

def samples_to_multitrack(samples):
  
  samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
  tracks = []
  for idx, (program, is_drum, track_name) in enumerate(
      zip(programs, is_drums, track_names)
  ):
      pianoroll = np.pad(
          samples[idx] > 0.1,
          ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
      )
      tracks.append(
          Track(
              name=track_name,
              program=program,
              is_drum=is_drum,
              pianoroll=pianoroll
          )
      )
  m = Multitrack(
      tracks=tracks,
      tempo=tempo_array,
      resolution=beat_resolution
  )

  return m

def write_sample(m, sf2_path, file_name = "sample", write_wav = False):
    """
    Save the Multitrack object m as midi and optionally wav
    """

    npz_name = file_name+".npz"
    midi_name = file_name+".mid"
    wav_name = file_name+".wav"

    m.save(npz_name)
    m = pypianoroll.load(npz_name)
    remove(npz_name)
    m.write(midi_name)
    if write_wav:
        music = PrettyMIDI(midi_file=midi_name)
        waveform = music.fluidsynth(sf2_path=sf2_path,fs=44100.0) # fluidsynth needs the sample frequency as a float,
        scipy.io.wavfile.write(wav_name,44100,waveform) # but scipy needs an integer. Care.


# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Metric extraction functions

def empty_bar_ratio(samples_np,verbose=True,std=False):
  """
  Samples are supposed to be a numpy array of shape [n,i,t,p] where
  · n is the number of samples
  · i is the number of instruments (tracks)
  · t is the timestep resolution (beat_resolution*beats_per_measure*measures)
  · p is the pitch dimension
  """

  torch.cuda.empty_cache()

  s = samples_np.shape
  samples = torch.from_numpy(samples_np)
  if torch.cuda.is_available():
      samples = samples.cuda()
  sample_bars = samples.view(s[0],s[1],s[2]//measure_resolution,measure_resolution,s[3])
  s = sample_bars.size()
  total_bars_per_track = s[0]*s[2]

  """
  sample_bars is now of shape [n,i,m,b,p] where
  · m is the number of measures per sample
  · b is the timesteps in a measure (beat_resolution*beats_per_measure)
  """

  """
  # Turn binary representation of pitch into pitch count
  non_zero_pitches = torch.count_nonzero(sample_bars,dim=4)
  # If atleast one pitch is present, the beat is not empty
  non_empty_beats = torch.count_nonzero(non_zero_pitches,dim=3)
  # If atleast one beat is not empty, the bar is not empty
  non_empty_bars = torch.count_nonzero(non_empty_beats,dim=2)
  """
  non_empty_bars = torch.count_nonzero(
                      torch.count_nonzero(
                          torch.count_nonzero(sample_bars, dim = 4),
                          dim = 3
                      ),
                      dim = 2
                   )

  ebrs=[]

  for i in range(s[1]):
    empty_bars_per_track = total_bars_per_track
    for n in range(s[0]):
      empty_bars_per_track-=non_empty_bars[n,i]
    EBr = empty_bars_per_track/total_bars_per_track
    EBr = EBr.item()
    if verbose:
      print("Empty Bar ratio for",track_names[i],":")
      print(EBr)
    ebrs.append(EBr)

  del non_empty_bars
  del sample_bars
  del samples
  torch.cuda.empty_cache()

  return np.array(ebrs)

def used_pitch_classes(samples_np,verbose=True,std=False):
    """
    Samples are supposed to be a numpy array of shape [n,i,t,p] where
    · n is the number of samples
    · i is the number of instruments (tracks)
    · t is the timestep resolution (beat_resolution*beats_per_measure*measures)
    · p is the pitch dimension
    """

    torch.cuda.empty_cache()

    s = samples_np.shape
    samples = torch.from_numpy(samples_np)
    if torch.cuda.is_available():
        samples = samples.cuda()
    sample_bars = samples.view(s[0],s[1],s[2]//measure_resolution,measure_resolution,s[3]//12,12)
    s = sample_bars.size()

    """
    sample_bars is now of shape [n,i,m,b,s,p] where
    · m is the number of measures per sample
    · b is the timesteps in a measure (beat_resolution*beats_per_measure)
    · s is the scale of the pitch
    """

    """
    # In the original paper, same pitch at different scales is counted only once
    # (This is inferred from the presented metrics but not actually discussed in the text)
    scaled_pitch_hist = torch.sum(sample_bars,dim=4)
    # Sum all pitch appearances over the beats of each measure
    pitch_hist = torch.sum(scaled_pitch_hist,dim=3)
    # If a pitch appears atleast once in a bar, it is counted
    pitches_per_bar = torch.count_nonzero(pitch_hist,dim=3).type(torch.float)
    # Average over all measure of each sample
    avg_p_per_sample = torch.mean(pitches_per_bar,dim=2)
    # Average over all samples
    avg_p_per_track = torch.mean(avg_p_per_sample,dim=0)
    """
    p_per_track = torch.mean(
                        torch.count_nonzero(
                            torch.sum(
                                torch.sum(sample_bars,dim=4),
                                dim = 3
                            ), dim = 3
                        ).type(torch.float),
                        dim = 2
                    )
    
    avg_p_per_track = torch.mean(p_per_track,dim=0)
    upcs=[]

    if std:
        std_p_per_track = torch.std(p_per_track,dim=0)
        upcs_std=[]

    for i in range(s[1]):
        if track_names[i]=="Drums": continue
        upc = avg_p_per_track[i].item()
        if verbose:
            print("Used pitch classes for",track_names[i],":")
            print(upc)
        upcs.append(upc)

        if std:
            upc_std = std_p_per_track[i].item()
            if verbose:
                print("Standard dev.:",upc_std)
            upcs_std.append(upc_std)

    del p_per_track
    del avg_p_per_track
    del sample_bars
    del samples
    if std:
        del std_p_per_track
    torch.cuda.empty_cache()

    if std:
        return np.array(upcs), np.array(upcs_std)
    else:
        return np.array(upcs)

# ------

r1=1.0 ; r2=1.0 ; r3 = 0.5
tm = np.empty((6, 12), dtype=np.float32)
tm[0, :] = r1*np.sin(np.arange(12)*(7./6.)*np.pi)
tm[1, :] = r1*np.cos(np.arange(12)*(7./6.)*np.pi)
tm[2, :] = r2*np.sin(np.arange(12)*(3./2.)*np.pi)
tm[3, :] = r2*np.cos(np.arange(12)*(3./2.)*np.pi)
tm[4, :] = r3*np.sin(np.arange(12)*(2./3.)*np.pi)
tm[5, :] = r3*np.cos(np.arange(12)*(2./3.)*np.pi)

def to_chroma(bar):
  chroma = bar.reshape(bar.shape[0], 12, -1).sum(axis=2)
  return chroma

def tonal_dist(beat_chroma1, beat_chroma2):
  beat_chroma1 = beat_chroma1 / np.sum(beat_chroma1)
  c1 = np.matmul(tm, beat_chroma1)
  beat_chroma2 = beat_chroma2 / np.sum(beat_chroma2)
  c2 = np.matmul(tm, beat_chroma2)
  return np.linalg.norm(c1-c2)

def bar_tonal_distance(chroma1, chroma2):
  chr1 = np.sum(chroma1, axis=0)
  chr2 = np.sum(chroma2, axis=0)
  return tonal_dist(chr1,chr2)

def tonal_distance(samples_np,verbose=True,std=False):
  """
  Samples are supposed to be a numpy array of shape [n,i,t,p] where
  · n is the number of samples
  · i is the number of instruments (tracks)
  · t is the timestep resolution (beat_resolution*beats_per_measure*measures)
  · p is the pitch dimension
  """

  torch.cuda.empty_cache()

  s = samples_np.shape
  bars = samples_np.view().reshape((s[0],s[1],s[2]//measure_resolution,measure_resolution,s[3]))
  s = bars.shape

  """
  bars is now of shape [n,i,m,b,p] where
  · m is the number of measures per sample
  · b is the timesteps in a measure (beat_resolution*beats_per_measure)
  """
  avg_scores = np.zeros((s[1]-1)*(s[1]-2)//2)
  if std:
     bar_scores = []
     score_stds = np.zeros((s[1]-1)*(s[1]-2)//2)
  current_combo = 0

  for t1 in range(1,s[1]):
    for t2 in range(t1+1,s[1]):
      total_bars = s[0]*s[2]
      for n in range(s[0]):
        for m in range(s[2]):
          bar1 = bars[n,t1,m,:,:]
          chroma1 = to_chroma(bar1)
          if (np.sum(chroma1)==0.0):
            total_bars-=1
            continue
          bar2 = bars[n,t2,m,:,:]
          chroma2 = to_chroma(bar2)
          if (np.sum(chroma2)==0.0):
            total_bars-=1
            continue
          td = bar_tonal_distance(chroma1,chroma2)
          avg_scores[current_combo] += td
          if std:
             bar_scores.append(td)
        # end for m
      # end for n
      if (total_bars==0):
        avg_scores[current_combo]=0
        if std:
           score_stds[current_combo]=0
      else:
        avg_scores[current_combo]/=total_bars
        if std:
           score_stds[current_combo]=np.std(np.array(bar_scores))
           bar_scores = []
      if verbose:
        print("Tonal distance",track_names[t1],"-",track_names[t2],":",avg_scores[current_combo])
        if std:
           print("Standard dev.:",score_stds[current_combo])
      current_combo+=1
    # end for t2
  # end for t1

  if std:
     return avg_scores, score_stds

  return avg_scores

def sample_from_midi(path):
    sample = pypianoroll.read(path)
    sample.set_resolution(beat_resolution)
    sample = (sample.stack() > 0)
    sample = sample[:, :64, lowest_pitch:lowest_pitch + n_pitches]
    if sample.shape[1]<64:
       sample = np.pad(sample,((0,0),(0,64-sample.shape[1]),(0,0)))
    if sample.shape[0]<3:
        # If only one instrument appears, it's drums
        # If only two instruments appear, it's drums and bass
        # (Not an actual solution but it is true for our subjects)
        if sample.shape[0]==1:
            sample = np.pad(sample,((0,2),(0,0),(0,0)))
        elif sample.shape[0]==2:
            new_sample = np.zeros((3,64,72))
            new_sample[0] = sample[0]
            new_sample[2] = sample[1]
            sample = new_sample
    sample = np.array([sample])
    print(sample.shape)

    return sample