#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

def mel_to_bark(mel):
  return 6 * math.log10((mel / 600) + math.sqrt(1 + (mel / 600) ** 2))

def hertz_to_mel(frequency):
  return 2595 * math.log10(1 + (frequency / 700))

def hertz_to_bark(frequency):
  return 6 * math.log10((frequency / 600) + math.sqrt(1 + (frequency / 600) ** 2))

# Example conversion from Mel to Bark
hertz_list = [1322.618908, 1044.622915, 1262.0233, 1452.889701, 1043.055522, 1614.899394]

for hertz in hertz_list:
  bark_frequency = hertz_to_bark(hertz)
  print(bark_frequency)
